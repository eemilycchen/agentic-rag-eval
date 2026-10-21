"""Shared settings: paths into the RAG project, API access, and model clients."""

import os
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# The original RapidFire AI RAG project 
# this repo reuses its corpus, golden Q&A pairs, and judge prompt
RAG_REPO = Path(__file__).resolve().parent.parent / 'rapidfireAI-rag-context-engineering'
CORPUS_DIR = RAG_REPO / 'data' / 'sourcedocs'
GOLDEN_QA = RAG_REPO / 'golden_qa_pairs.json'
JUDGE_PROMPT = RAG_REPO / 'metric' / 'judge_prompt.txt'


PROVIDERS = {
    # OpenRouter, pay-as-you-go 
    # get a key at https://openrouter.ai/keys
    'openrouter': {
        'base_url': 'https://openrouter.ai/api/v1',
        'embedding_model': 'qwen/qwen3-embedding-8b',
        'model': 'openai/gpt-oss-120b',
        'key_env': 'OPENROUTER_API_KEY',
        'key_file': '~/openrouter-api-key.txt',
    }
    # ... NRP
}
PROVIDER = os.environ.get('LLM_PROVIDER', 'openrouter')
if PROVIDER not in PROVIDERS:
    raise ValueError(f'LLM_PROVIDER must be one of {sorted(PROVIDERS)}, got {PROVIDER!r}')

BASE_URL = PROVIDERS[PROVIDER]['base_url']
EMBEDDING_MODEL = PROVIDERS[PROVIDER]['embedding_model']
DEFAULT_MODEL = PROVIDERS[PROVIDER]['model']


def load_api_key(path=None):
    '''
    read the API key from the provider's env var, falling back to the first line of a key file
    '''
    settings = PROVIDERS[PROVIDER]
    if os.environ.get(settings['key_env']):
        return os.environ[settings['key_env']]
    path = path or settings['key_file']
    return Path(path).expanduser().read_text(encoding='utf-8').splitlines()[0].strip()


def make_embeddings(api_key):
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL, api_key=api_key, base_url=BASE_URL, check_embedding_ctx_length=False
    )


def make_llm(model, api_key):
    return ChatOpenAI(
        model=model, api_key=api_key, base_url=BASE_URL, temperature=0, max_retries=3, timeout=180
    )
