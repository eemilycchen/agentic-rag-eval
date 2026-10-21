"""Tests that run without an API key: chunk spans, scoring, the agent's search budget, and escalation."""

import asyncio
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import BUDGET_MESSAGE, build_agent, run_adaptive, run_agent  # noqa: E402
from configs import ESCALATE, AgentConfig  # noqa: E402
from evaluate import bootstrap_ci, evidence_recall, paired_bootstrap_diff  # noqa: E402
from retriever import chunk_lines  # noqa: E402

count_words = lambda text: len(text.split())


def test_chunk_spans_point_at_the_chunk_text():
    lines = [f'line {i} ' + 'word ' * (i % 7) for i in range(1, 60)]
    chunks = chunk_lines(lines, 'doc.rst', chunk_tokens=30, overlap_tokens=8, count_tokens=count_words)
    assert len(chunks) > 1
    for chunk in chunks:
        start, end = chunk.metadata['lines']
        assert chunk.page_content == '\n'.join(lines[start - 1:end]).strip()
    # every line is covered and consecutive chunks overlap
    assert chunks[0].metadata['lines'][0] == 1 and chunks[-1].metadata['lines'][1] == len(lines)
    for a, b in zip(chunks, chunks[1:]):
        assert b.metadata['lines'][0] <= a.metadata['lines'][1]


def test_chunking_terminates_on_a_line_longer_than_the_budget():
    lines = ['short', 'word ' * 100, 'short']
    chunks = chunk_lines(lines, 'doc.rst', chunk_tokens=10, overlap_tokens=5, count_tokens=count_words)
    assert [c.metadata['lines'] for c in chunks] == [[1, 1], [2, 2], [3, 3]]


def test_evidence_recall_counts_overlapping_gold_spans():
    gold = [('a.rst', 10, 20), ('b.rst', 5, 8)]
    assert evidence_recall([('a.rst', 18, 30)], gold) == 0.5
    assert evidence_recall([('a.rst', 21, 30), ('c.rst', 5, 8)], gold) == 0.0
    assert evidence_recall([('a.rst', 1, 10), ('b.rst', 8, 9)], gold) == 1.0


def test_bootstrap_intervals():
    mean, low, high = bootstrap_ci([0, 1] * 50)
    assert mean == 0.5 and low < 0.5 < high
    mean, low, high = paired_bootstrap_diff([1] * 20, [0] * 20)
    assert (mean, low, high) == (1.0, 1.0, 1.0)


class AlwaysSearches(BaseChatModel):
    '''
    a scripted model that requests more searches on every turn unless tools are disabled; called
    without tools bound (the RAG step), it gives rag_reply
    '''
    tool_choice: str | None = None
    has_tools: bool = False
    calls_per_turn: int = 1
    rag_reply: str = 'rag answer'

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        return self.model_copy(update={'tool_choice': tool_choice, 'has_tools': True})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        usage = {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}
        if not self.has_tools:
            message = AIMessage(self.rag_reply, usage_metadata=usage)
        elif self.tool_choice == 'none':
            message = AIMessage('final answer', usage_metadata=usage)
        else:
            n = sum(isinstance(m, ToolMessage) for m in messages)
            message = AIMessage('', tool_calls=[
                {'name': 'search_docs', 'args': {'query': f'query {n + i}'}, 'id': f'call{n + i}'}
                for i in range(self.calls_per_turn)
            ])
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self):
        return 'always-searches'


class FakeSearch:
    def search(self, query, k=5, rerank=False):
        return [Document(query, metadata={'file': 'a.rst', 'lines': [i, i + 1]}) for i in range(1, k + 1)]


def test_agent_stops_at_the_search_budget_and_answers():
    cfg = AgentConfig('test', 'agent', 'fake', k=2, max_searches=3)
    graph = build_agent(AlwaysSearches(), FakeSearch(), cfg)
    row = asyncio.run(run_agent(graph, 'question?', cfg))
    assert row['answer'] == 'final answer'
    assert row['searches'] == 3 and row['hit_budget']
    assert row['queries'] == ['query 0', 'query 1', 'query 2', 'query 3']
    assert row['llm_calls'] == 5 and row['output_tokens'] == 2
    # each query returns spans (1,2) and (2,3) with that query's text, so 3 searches give 6 unique chunks
    assert len(row['retrieved_spans']) == 6
    assert BUDGET_MESSAGE not in row['retrieved_context']


def test_parallel_calls_over_budget_still_run_the_ones_that_fit():
    # three searches at once with a budget of two: two run, the third is refused, then the agent answers
    cfg = AgentConfig('test', 'agent', 'fake', k=1, max_searches=2)
    graph = build_agent(AlwaysSearches(calls_per_turn=3), FakeSearch(), cfg)
    row = asyncio.run(run_agent(graph, 'question?', cfg))
    assert row['searches'] == 2 and row['hit_budget']
    assert row['answer'] == 'final answer' and row['llm_calls'] == 2
    assert row['retrieved_context'] == 'query 0\n\nquery 1'


def test_adaptive_answers_with_rag_when_the_passages_suffice():
    cfg = AgentConfig('test', 'adaptive', 'fake', k=2, max_searches=3)
    llm = AlwaysSearches()
    row = asyncio.run(run_adaptive(llm, build_agent(llm, FakeSearch(), cfg), FakeSearch(), 'question?', cfg))
    assert not row['escalated']
    assert row['answer'] == 'rag answer' and row['searches'] == 1 and row['llm_calls'] == 1


def test_adaptive_escalates_and_the_rag_search_counts_toward_the_budget():
    cfg = AgentConfig('test', 'adaptive', 'fake', k=2, max_searches=3)
    llm = AlwaysSearches(rag_reply=ESCALATE)
    row = asyncio.run(run_adaptive(llm, build_agent(llm, FakeSearch(), cfg), FakeSearch(), 'question?', cfg))
    assert row['escalated'] and row['answer'] == 'final answer'
    # the RAG search plus two agent searches use the budget of three
    assert row['searches'] == 3 and row['hit_budget']
    assert row['queries'][:2] == ['question?', 'query 1']
    # the RAG call's tokens are counted along with the agent's
    assert row['llm_calls'] == 5 and row['input_tokens'] == 20
