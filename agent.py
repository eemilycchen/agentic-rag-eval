"""The tool-calling agent (a small LangGraph state machine), the adaptive pipeline, and the reference baselines.

Agent graph:

    START -> agent --(tool calls)------> tools --(all calls within budget)--> agent
                   --(no tool calls)--> END     --(any call over budget)----> answer -> END

The search budget is enforced in the graph rather than by the prompt: the 'tools' node runs as
many of the requested searches as the budget still allows and answers the rest with a stop notice,
then the 'answer' node makes the model answer with tools disabled.

Adaptive pipeline: plain RAG answers first; if the model says the passages are not enough, the
question escalates to the agent, which starts from the passages RAG already retrieved.
"""

import asyncio
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

from configs import ADAPTIVE_RAG_PROMPT, ESCALATE, FULL_CONTEXT_PROMPT, PROMPTS, RAG_PROMPT
from retriever import format_docs, span

BUDGET_MESSAGE = 'Search budget used up. Answer now using the passages you already have.'


def build_agent(llm, search, cfg):
    @tool(response_format='content_and_artifact')
    def search_docs(query: str):
        """Search the RapidFire AI documentation. Returns the most relevant passages, each labeled with its file and line range."""
        docs = search.search(query, k=cfg.k, rerank=cfg.rerank)
        # the artifact carries the spans and text for scoring; only the content goes to the LLM
        return format_docs(docs), [(*span(d), d.page_content) for d in docs]

    llm_with_tools = llm.bind_tools([search_docs])
    llm_answer_only = llm.bind_tools([search_docs], tool_choice='none')
    system = SystemMessage(PROMPTS[cfg.prompt])

    async def call_agent(state):
        return {'messages': [await llm_with_tools.ainvoke([system, *state['messages']])]}

    async def run_search(call):
        # gpt-oss sometimes sends its built-in browser tool's arguments ({'cursor': 0, 'id': 0}) instead
        # of a query; tell the model so it can retry, and count the call toward the budget (empty
        # artifact) so malformed calls cannot loop forever
        if not isinstance(call['args'].get('query'), str):
            return ToolMessage(f'Invalid arguments {call["args"]}: search_docs takes {{"query": "<text>"}}.',
                               tool_call_id=call['id'], artifact=[])
        return await search_docs.ainvoke({**call, 'type': 'tool_call'})

    async def run_tools(state):
        # run the calls the budget still covers; every other call gets a stop notice, since each
        # tool call needs a matching tool message
        calls = state['messages'][-1].tool_calls
        remaining = max(cfg.max_searches - searches_run(state['messages']), 0)
        results = await asyncio.gather(*(run_search(c) for c in calls[:remaining]))
        refused = [ToolMessage(BUDGET_MESSAGE, tool_call_id=c['id']) for c in calls[remaining:]]
        return {'messages': [*results, *refused]}

    async def force_answer(state):
        return {'messages': [await llm_answer_only.ainvoke([system, *state['messages']])]}

    def after_agent(state):
        return 'tools' if state['messages'][-1].tool_calls else END

    def after_tools(state):
        last = state['messages'][-1]
        return 'answer' if isinstance(last, ToolMessage) and last.content == BUDGET_MESSAGE else 'agent'

    graph = StateGraph(MessagesState)
    graph.add_node('agent', call_agent)
    graph.add_node('tools', run_tools)
    graph.add_node('answer', force_answer)
    graph.add_edge(START, 'agent')
    graph.add_conditional_edges('agent', after_agent, ['tools', END])
    graph.add_conditional_edges('tools', after_tools, ['agent', 'answer'])
    graph.add_edge('answer', END)
    return graph.compile()


def searches_run(messages):
    '''
    searches that actually ran (refused calls carry no artifact)
    '''
    return sum(1 for m in messages if isinstance(m, ToolMessage) and m.artifact is not None)


async def run_agent(graph, question, cfg, history=(), started=None):
    '''
    run the agent on a question; history lets the adaptive pipeline hand over the search it already did
    '''
    started = started or time.perf_counter()
    # a backstop on graph steps in case routing ever loops; the budget normally ends the run first
    state = await graph.ainvoke(
        {'messages': [HumanMessage(question), *history]}, {'recursion_limit': 2 * cfg.max_searches + 6}
    )
    messages = state['messages']
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    retrieved = [item for m in tool_messages if m.artifact for item in m.artifact]
    return summarize(
        messages, retrieved, started,
        searches=searches_run(messages),
        hit_budget=any(m.content == BUDGET_MESSAGE for m in tool_messages),
    )


async def run_rag(llm, search, question, cfg, prompt=RAG_PROMPT):
    started = time.perf_counter()
    docs = await asyncio.to_thread(search.search, question, cfg.k, cfg.rerank)
    messages = [
        SystemMessage(prompt),
        HumanMessage(f'Question: {question}\n\nDocumentation passages:\n{format_docs(docs)}'),
    ]
    messages.append(await llm.ainvoke(messages))
    retrieved = [(*span(d), d.page_content) for d in docs]
    return summarize(messages, retrieved, started, searches=1, hit_budget=False), docs, messages[-1]


async def run_adaptive(llm, graph, search, question, cfg):
    '''
    answer with plain RAG; escalate to the agent only when the model says the passages fall short
    '''
    started = time.perf_counter()
    row, docs, response = await run_rag(llm, search, question, cfg, prompt=ADAPTIVE_RAG_PROMPT)
    if not text_of(response).strip().startswith(ESCALATE):
        return {**row, 'escalated': False}
    # hand the RAG search to the agent as its first tool call, so it continues from those passages;
    # the call carries the RAG response's token usage so the escalation decision is counted in the cost
    call = {'name': 'search_docs', 'args': {'query': question}, 'id': 'rag-search'}
    history = [
        AIMessage('', tool_calls=[call], usage_metadata=response.usage_metadata),
        ToolMessage(format_docs(docs), tool_call_id=call['id'], artifact=[(*span(d), d.page_content) for d in docs]),
    ]
    return {**await run_agent(graph, question, cfg, history, started), 'escalated': True}


async def run_full_context(llm, search, question):
    '''
    reference ceiling: the whole corpus in one prompt, no retrieval
    '''
    started = time.perf_counter()
    messages = [
        SystemMessage(FULL_CONTEXT_PROMPT),
        HumanMessage(f'Question: {question}\n\nDocumentation:\n{format_docs(search.files)}'),
    ]
    messages.append(await llm.ainvoke(messages))
    retrieved = [(*span(d), d.page_content) for d in search.files]
    return summarize(messages, retrieved, started, searches=0, hit_budget=False)


def text_of(message):
    '''
    message content as a string (some providers return a list of content parts)
    '''
    if isinstance(message.content, str):
        return message.content
    return ''.join(part.get('text', '') for part in message.content if isinstance(part, dict))


def summarize(messages, retrieved, started, searches, hit_budget):
    '''
    one row of results: the answer plus the trajectory, token, and latency measurements
    '''
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    usage = [m.usage_metadata or {} for m in ai_messages]
    # the same chunk can come back from several searches; keep the first copy
    unique = list(dict.fromkeys(retrieved))
    return {
        'answer': text_of(ai_messages[-1]).strip() if ai_messages else '',
        'queries': [c['args'].get('query', '') for m in ai_messages for c in m.tool_calls],
        'searches': searches,
        'hit_budget': hit_budget,
        'llm_calls': len(ai_messages),
        'input_tokens': sum(u.get('input_tokens', 0) for u in usage),
        'output_tokens': sum(u.get('output_tokens', 0) for u in usage),
        'latency_s': time.perf_counter() - started,
        'retrieved_spans': [item[:3] for item in unique],
        'retrieved_context': '\n\n'.join(item[3] for item in unique),
    }
