"""Run every config over the golden Q&A set, score it, and compare against the RAG baseline.

    python run_experiments.py --limit 3 --no-mlflow     # smoke test
    python run_experiments.py                           # full grid, logged to MLflow
"""

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI

import common
from agent import build_agent, run_adaptive, run_agent, run_full_context, run_rag
from configs import default_grid
from evaluate import bootstrap_ci, call_judge, evidence_recall, gold_spans, paired_bootstrap_diff
from retriever import DocSearch

RERANK_MODEL = 'BAAI/bge-reranker-v2-m3'


async def run_config(cfg, examples, search, api_key, concurrency):
    llm = common.make_llm(cfg.model, api_key)
    graph = build_agent(llm, search, cfg) if cfg.mode in ('agent', 'adaptive') else None
    limit = asyncio.Semaphore(concurrency)

    async def one(example):
        async with limit:
            try:
                question = example['question']
                if cfg.mode == 'agent':
                    row = await run_agent(graph, question, cfg)
                elif cfg.mode == 'adaptive':
                    row = await run_adaptive(llm, graph, search, question, cfg)
                elif cfg.mode == 'full':
                    row = await run_full_context(llm, search, question)
                else:
                    row, _, _ = await run_rag(llm, search, question, cfg)
            except Exception as e:
                # a crashed run still counts as an answered-wrong question
                row = {'answer': '', 'error': f'{type(e).__name__}: {e}', 'retrieved_spans': [],
                       'retrieved_context': ''}
            return {'config': cfg.name, 'question_id': example['question_id'], **row}

    return await asyncio.gather(*(one(e) for e in examples))


async def judge_rows(rows, examples, args, api_key):
    client = OpenAI(api_key=api_key, base_url=common.BASE_URL)
    template = Path(args.judge_prompt).read_text(encoding='utf-8')
    by_id = {e['question_id']: e for e in examples}
    limit = asyncio.Semaphore(args.concurrency)

    async def one(row):
        example = by_id[row['question_id']]
        async with limit:
            scores = await asyncio.to_thread(
                call_judge, client, args.judge_model, template, example['question'],
                example['reference_answer'], row['retrieved_context'], row['answer'],
            )
        row.update(scores)

    await asyncio.gather(*(one(r) for r in rows))


def summarize_config(df):
    summary = {}
    for metric in ['correctness', 'faithfulness', 'completeness', 'evidence_recall']:
        if metric in df:
            mean, low, high = bootstrap_ci(df[metric])
            summary[metric] = f'{mean:.2f} [{low:.2f}, {high:.2f}]'
            summary[f'{metric}_mean'] = mean
    # single-span questions are lookups; multi-span ones need several facts, where extra searching should pay off
    if 'correctness' in df:
        for kind, subset in [('single', df[~df['multi_span']]), ('multi', df[df['multi_span']])]:
            summary[f'correctness_{kind}'] = subset['correctness'].mean()
    if 'escalated' in df:
        summary['escalated'] = df['escalated'].eq(True).mean()
    summary['searches'] = df['searches'].mean()
    summary['hit_budget'] = df['hit_budget'].mean()
    summary['tokens'] = tokens(df)
    summary['latency_s_p50'] = df['latency_s'].median()
    summary['errors'] = int(df['error'].notna().sum()) if 'error' in df else 0
    return summary


def baseline_for(cfg, configs):
    '''
    the RAG baseline for a config's model: the first rag config in the grid
    '''
    return next((c for c in configs if c.mode == 'rag' and c.model == cfg.model), None)


def tokens(df):
    return (df['input_tokens'] + df['output_tokens']).mean()


def compare_to_baseline(results, configs):
    '''
    paired difference in correctness between each config and the RAG baseline for the same model,
    with the token cost relative to the baseline
    '''
    rows = []
    for cfg in configs:
        baseline = baseline_for(cfg, configs)
        if baseline is None or cfg == baseline or 'correctness' not in results[cfg.name]:
            continue
        a = results[cfg.name].set_index('question_id')['correctness']
        b = results[baseline.name].set_index('question_id')['correctness'].reindex(a.index)
        mean, low, high = paired_bootstrap_diff(a, b)
        rows.append({'config': cfg.name, 'vs': baseline.name, 'correctness_diff': f'{mean:+.2f} [{low:+.2f}, {high:+.2f}]',
                     'ci_excludes_0': low > 0 or high < 0,
                     'token_cost_x': tokens(results[cfg.name]) / tokens(results[baseline.name])})
    return pd.DataFrame(rows)


def escalation_report(results, configs):
    '''
    how well each adaptive config picks the questions to escalate, using the RAG baseline's failures
    as the questions that needed it
    '''
    rows = []
    for cfg in configs:
        baseline = baseline_for(cfg, configs)
        if cfg.mode != 'adaptive' or baseline is None or 'correctness' not in results[baseline.name]:
            continue
        escalated = results[cfg.name].set_index('question_id')['escalated'].eq(True)
        rag_failed = results[baseline.name].set_index('question_id')['correctness'].reindex(escalated.index) == 0
        caught = (escalated & rag_failed).sum()
        rows.append({
            'config': cfg.name,
            'escalation_rate': escalated.mean(),
            # of the questions plain RAG got wrong, the share sent to the agent
            'rag_failures_escalated': caught / rag_failed.sum() if rag_failed.any() else float('nan'),
            # of the escalated questions, the share plain RAG got wrong (the rest were extra cost)
            'escalations_needed': caught / escalated.sum() if escalated.any() else float('nan'),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa', default=common.GOLDEN_QA, help='golden Q&A pairs with source evidence')
    parser.add_argument('--corpus-dir', default=common.CORPUS_DIR)
    parser.add_argument('--judge-prompt', default=common.JUDGE_PROMPT)
    parser.add_argument('--models', nargs='+', default=[common.DEFAULT_MODEL], help='generation models to sweep')
    parser.add_argument('--judge-model', default=common.DEFAULT_MODEL)
    parser.add_argument('--configs', nargs='*', help='only run configs whose names contain one of these strings')
    parser.add_argument('--limit', type=int, help='only use the first N questions (for smoke tests)')
    parser.add_argument('--concurrency', type=int, default=4, help='questions in flight at once')
    parser.add_argument('--rerank', action='store_true', help=f'rerank search results with {RERANK_MODEL}')
    parser.add_argument('--no-judge', action='store_true')
    parser.add_argument('--no-mlflow', action='store_true')
    parser.add_argument('--api-key-file', help="defaults to the provider's key file in common.py")
    parser.add_argument('--out', default='results')
    args = parser.parse_args()

    api_key = common.load_api_key(args.api_key_file)
    examples = json.loads(Path(args.qa).read_text(encoding='utf-8'))[:args.limit]
    search = DocSearch(args.corpus_dir, common.make_embeddings(api_key),
                       rerank_model=RERANK_MODEL if args.rerank else None)
    configs = default_grid(args.models, rerank=args.rerank)
    if args.configs:
        configs = [c for c in configs if any(s in c.name for s in args.configs)]

    out_dir = Path(args.out) / datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir.mkdir(parents=True)

    if not args.no_mlflow:
        import mlflow

        mlflow.set_experiment('agentic-rag-eval')
        # records every LangChain/LangGraph call as a trace: tool calls, prompts, tokens, latency
        mlflow.langchain.autolog()

    # one event loop for every config: langchain-openai caches its async HTTP client, and
    # a fresh asyncio.run per config leaves it holding connections from a closed loop
    loop = asyncio.new_event_loop()
    results, summaries = {}, []
    for cfg in configs:
        print(f'running {cfg.name} on {len(examples)} questions')
        run = None if args.no_mlflow else mlflow.start_run(run_name=cfg.name)
        rows = loop.run_until_complete(run_config(cfg, examples, search, api_key, args.concurrency))
        if not args.no_judge:
            loop.run_until_complete(judge_rows(rows, examples, args, api_key))
        gold = {e['question_id']: gold_spans(e) for e in examples}
        for row in rows:
            row['evidence_recall'] = evidence_recall(row['retrieved_spans'], gold[row['question_id']])
            row['multi_span'] = len(gold[row['question_id']]) > 1

        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f'{cfg.name}.csv', index=False)
        results[cfg.name] = df
        summary = {'config': cfg.name, **summarize_config(df)}
        summaries.append(summary)

        if run is not None:
            mlflow.log_params(asdict(cfg))
            mlflow.log_metrics({k: float(v) for k, v in summary.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(str(out_dir / f'{cfg.name}.csv'))
            mlflow.end_run()

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / 'summary.csv', index=False)
    shown = [c for c in ['config', 'correctness', 'correctness_single', 'correctness_multi', 'completeness',
                         'evidence_recall', 'escalated', 'searches', 'hit_budget', 'tokens', 'latency_s_p50', 'errors']
             if c in summary_df]
    print('\n' + summary_df[shown].to_string(index=False, float_format='%.2f'))

    comparison = compare_to_baseline(results, configs)
    if not comparison.empty:
        comparison.to_csv(out_dir / 'vs_baseline.csv', index=False)
        print('\n' + comparison.to_string(index=False))

    escalation = escalation_report(results, configs)
    if not escalation.empty:
        escalation.to_csv(out_dir / 'escalation.csv', index=False)
        print('\n' + escalation.to_string(index=False, float_format='%.2f'))
    print(f'\nresults written to {out_dir}')


if __name__ == '__main__':
    main()
