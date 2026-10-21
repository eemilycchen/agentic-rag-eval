# Experimenting Adaptive RAG

A question-answering system over the [RapidFire AI](https://github.com/RapidFireAI/rapidfireai) documentation that answers most questions with one retrieval and hands the harder ones to a tool-calling agent that keeps searching until it has what it needs.

LLM Statement: Used claude code to plan and implement meaningful extensions to a previous [graduate course project](https://github.com/eemilycchen/rapidfireAI-rag-context-engineering). My role is defining the scope, verifying the work, and grab key takeaways from the experimentation.

## Question

Most documentation questions are lookups that a single retrieval. Maybe an agent that plans and runs several searches handles those better. However, the cost may go up.

**How much of the agent's quality can we get by escalating to it only when plain RAG's passages fall short, and at what fraction of its cost?**

## How it works

**Adaptive pipeline** (`run_adaptive` in `agent.py`):

```
question -> retrieve k chunks -> LLM: answer, or say NEED_MORE_SEARCH
                                   |-- answer -----------> done (one search, one LLM call)
                                   '-- NEED_MORE_SEARCH --> agent, starting from those chunks
```

The first step is plain RAG with one change to the prompt: if the passages miss anything the question asks for, the model replies `NEED_MORE_SEARCH` instead of answering. The question then goes to the agent, which gets the RAG search as its first tool call, so nothing is retrieved twice. That first search counts toward the agent's budget.

**Agent**: a LangGraph state machine with one tool, `search_docs`, and the `plan` prompt (break the question into the facts it needs, then run one focused search for each).

```
START -> agent --(tool calls)------> tools --(all calls within budget)--> agent
               --(no tool calls)--> END     --(any call over budget)----> answer -> END
```

The search budget is enforced by the graph, not by the prompt. The `tools` node runs as many of the requested searches as the budget still covers and answers the rest with a "budget used up" message, then the `answer` node calls the model again with tools disabled. This caps the cost and stops search loops.

**Retrieval** (`retriever.py`): line-aware chunking (about 384 tokens with 64 overlap) so every chunk carries its exact `file:start-end` span. FAISS with MMR search, plus optional `bge-reranker-v2-m3` reranking.

**Configs** (`configs.py`), per model. The agent settings (`plan` prompt, 8 chunks per search, budget of 4 searches) were fixed before any results were seen and are shared by the agent and adaptive configs, so the only difference between them is the escalation step.

| Config | What it is | Role |
|---|---|---|
| `rag-k8` | one search with the question, one LLM call | baseline |
| `agent` | always uses the agent | quality target |
| `adaptive` | RAG first, escalates to the agent when needed | the system |
| `rag-k32` | one search returning as many chunks as the agent can read in total | is it just more context? |
| `full-context` | the whole corpus (about 30k tokens) in one prompt | ceiling |

**Metrics** (`evaluate.py` and `run_experiments.py`), per question:
- **Quality**: correctness (0/1), faithfulness (0/1), and completeness (0–5) from the LLM judge, also split by **single-span** questions (lookups) and **multi-span** questions (several facts)
- **Retrieval**: evidence recall, the fraction of golden source spans overlapped by anything the system retrieved
- **Escalation**: how often the adaptive pipeline escalates, the share of plain RAG's wrong answers it escalated, and the share of escalations plain RAG actually needed
- **Cost**: searches, budget hits, LLM calls, input/output tokens, latency

Each config reports means with **95% bootstrap CIs**, and each config is compared to `rag-k8` with a **paired bootstrap** over questions (so question difficulty cancels out), alongside its token cost relative to the baseline.

Every LLM and tool call is traced in **MLflow** (one run per config), so trajectories can be inspected and compared side by side.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# I used open router
export OPENROUTER_API_KEY=...        

python run_experiments.py

# look at charts
mlflow ui
```
Nice to have for later:

Useful flags: `--models a b` sweeps several generation models, `--rerank` adds the cross-encoder, `--concurrency N` sets how many questions run in parallel, and `--no-judge` skips judging.

Results go to `results/<timestamp>/`: one CSV per config (answers, queries, spans, scores), `summary.csv`, `vs_baseline.csv`, and `escalation.csv`.

**MCP**: `mcp_server.py` serves the same `search_docs` tool over MCP, so any MCP client can search the docs, for example `claude mcp add rapidfire-docs -- python mcp_server.py`.

## Results

Ran it first time, and it was roughly as good as RAG in this set up. It costed about $0.55 for the run.