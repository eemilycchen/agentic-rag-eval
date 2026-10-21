"""Line-aware chunking and document search over the RapidFire AI docs.

Every chunk keeps the file name and the exact line span it came from, so retrieved
evidence can be scored against the golden source spans.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def chunk_lines(lines, file, chunk_tokens, overlap_tokens, count_tokens):
    '''
    pack whole lines into chunks of about chunk_tokens, overlapping consecutive chunks by about overlap_tokens
    '''
    # +1 for the newline that joins the lines back together
    sizes = [count_tokens(line) + 1 for line in lines]
    chunks, start = [], 0
    while start < len(lines):
        # grow the chunk line by line until the budget is used (always take at least one line)
        end, total = start, 0
        while end < len(lines) and (total + sizes[end] <= chunk_tokens or end == start):
            total += sizes[end]
            end += 1
        text = '\n'.join(lines[start:end]).strip()
        if text:
            chunks.append(Document(page_content=text, metadata={'file': file, 'lines': [start + 1, end]}))
        if end >= len(lines):
            break
        # step back from the end so the next chunk repeats about overlap_tokens of context
        next_start, overlap = end, 0
        while next_start > start + 1 and overlap + sizes[next_start - 1] <= overlap_tokens:
            next_start -= 1
            overlap += sizes[next_start]
        start = next_start
    return chunks


def load_chunks(corpus_dir, chunk_tokens=384, overlap_tokens=64):
    import tiktoken

    encoding = tiktoken.get_encoding('o200k_base')
    count_tokens = lambda text: len(encoding.encode(text))
    chunks = []
    for path in sorted(Path(corpus_dir).glob('**/*.rst')):
        lines = path.read_text(encoding='utf-8').split('\n')
        chunks.extend(chunk_lines(lines, path.name, chunk_tokens, overlap_tokens, count_tokens))
    return chunks


def load_files(corpus_dir):
    '''
    each doc file as one Document spanning all its lines, for the whole-corpus reference
    '''
    files = []
    for path in sorted(Path(corpus_dir).glob('**/*.rst')):
        text = path.read_text(encoding='utf-8')
        files.append(Document(page_content=text, metadata={'file': path.name, 'lines': [1, len(text.split('\n'))]}))
    return files


def span(doc):
    '''
    (file, start line, end line) for a retrieved chunk
    '''
    return (doc.metadata['file'], *doc.metadata['lines'])


def format_docs(docs):
    '''
    render retrieved chunks for the LLM, labeling each with its source span
    '''
    return '\n\n---\n\n'.join(
        f'[{file} lines {start}-{end}]\n{doc.page_content}'
        for doc, (file, start, end) in ((d, span(d)) for d in docs)
    )


class DocSearch:
    '''
    FAISS index over the docs with MMR search and optional cross-encoder reranking;
    the index is built once and cached on disk
    '''

    def __init__(self, corpus_dir, embeddings, index_dir='index', chunk_tokens=384, overlap_tokens=64,
                 rerank_model=None):
        path = Path(index_dir) / f'faiss-c{chunk_tokens}-o{overlap_tokens}'
        if path.exists():
            self.store = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
        else:
            self.store = FAISS.from_documents(load_chunks(corpus_dir, chunk_tokens, overlap_tokens), embeddings)
            self.store.save_local(str(path))
        self.files = load_files(corpus_dir)
        # load the reranker up front so concurrent searches don't race to load it
        self.reranker = None
        if rerank_model:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(rerank_model, device='cpu')

    def search(self, query, k=5, rerank=False):
        if not rerank:
            return self.store.max_marginal_relevance_search(query, k=k, fetch_k=4 * k)
        if self.reranker is None:
            raise ValueError('rerank=True needs DocSearch(rerank_model=...)')
        # over-fetch with MMR, then keep the k chunks the cross encoder scores highest
        candidates = self.store.max_marginal_relevance_search(query, k=3 * k, fetch_k=6 * k)
        scores = self.reranker.predict([(query, d.page_content) for d in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in ranked[:k]]
