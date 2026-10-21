"""The configuration space: each AgentConfig is one variant of the system to evaluate."""

from dataclasses import dataclass

PROMPTS = {
    # answer with whatever the search turns up
    'basic': """You answer questions about the RapidFire AI documentation.Use the search_docs tool to find relevant passages before answering. You may search more than once with different queries if the first results do not cover the question. Answer only from the retrieved passages; if they do not contain the answer, say so. Write the answer in plain text and end with the sources you used, as file names and line ranges.""",
    # decompose first, then search for each part
    'plan': """You answer questions about the RapidFire AI documentation. Before searching, break the question into the facts you need (for a comparison, each side; for a procedure, each step). Search for each one with a focused query using the search_docs tool, and check whether the results cover every part before answering. Avoid repeating a query you already ran. Answer only from the retrieved passages; if they do not contain the answer, say so. Write the answer in plain text and end with the sources you used, as file names and line ranges.""",
}

# the plain RAG baseline: one search with the raw question, one LLM call
RAG_PROMPT = """You answer questions about the RapidFire AI documentation using only the passages below. If they do not contain the answer, say so. Write the answer in plain text and end with the sources you used, as file names and line ranges."""

# the adaptive pipeline's first step: plain RAG that can ask for more searching instead of answering
ESCALATE = 'NEED_MORE_SEARCH'
ADAPTIVE_RAG_PROMPT = f"""You answer questions about the RapidFire AI documentation using only the passages below.
If the passages fully answer the question, write the answer in plain text and end with the sources you used, as file names and line ranges.
If they are missing anything the question asks for (a fact, a step, or one side of a comparison), reply with exactly {ESCALATE} and nothing else."""

# the long-context reference: the whole corpus in the prompt
FULL_CONTEXT_PROMPT = """You answer questions about the RapidFire AI documentation using only the documentation below.
If it does not contain the answer, say so. Write the answer in plain text and end with the sources you used, as file names and line ranges."""


@dataclass(frozen=True)
class AgentConfig:
    name: str
    # 'rag' retrieves once with the question; 'agent' lets the LLM decide when and what to search;
    # 'adaptive' runs rag and escalates to the agent when the passages fall short;
    # 'full' puts the whole corpus in the prompt
    mode: str
    model: str
    prompt: str = 'basic'
    # chunks returned per search
    k: int = 5
    # agent and adaptive: searches allowed (including adaptive's first RAG search) before answering is forced
    max_searches: int = 3
    rerank: bool = False


def short_name(model):
    # drop the provider prefix too ('openai/gpt-oss-120b' -> 'gpt-oss-120b'): config names become file names
    return model.removeprefix('api-').rsplit('/', 1)[-1]


# the agent settings shared by the always-agent and adaptive configs, fixed before any results were seen;
# k matches the RAG baseline so adaptive's first step retrieves exactly what the baseline does
AGENT_PROMPT, AGENT_K, AGENT_SEARCHES = 'plan', 8, 4


def default_grid(models, rerank=False):
    '''
    for each model: the RAG baseline, the always-agent and adaptive systems with the same agent
    settings, and two references (RAG with the agent's full context budget, and the whole corpus)
    '''
    configs = []
    for model in models:
        m = short_name(model)
        configs += [
            AgentConfig(f'rag-k{AGENT_K}-{m}', 'rag', model, k=AGENT_K, rerank=rerank),
            AgentConfig(f'agent-{m}', 'agent', model, prompt=AGENT_PROMPT, k=AGENT_K,
                        max_searches=AGENT_SEARCHES, rerank=rerank),
            AgentConfig(f'adaptive-{m}', 'adaptive', model, prompt=AGENT_PROMPT, k=AGENT_K,
                        max_searches=AGENT_SEARCHES, rerank=rerank),
            AgentConfig(f'rag-k{AGENT_K * AGENT_SEARCHES}-{m}', 'rag', model, k=AGENT_K * AGENT_SEARCHES,
                        rerank=rerank),
            AgentConfig(f'full-context-{m}', 'full', model),
        ]
    return configs
