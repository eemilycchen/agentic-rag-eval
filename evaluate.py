"""Scoring: evidence recall against the golden spans, an LLM judge, and bootstrap confidence intervals."""

import json
import re

import numpy as np


def overlaps(a, b):
    '''
    two (file, start, end) spans overlap if they are in the same file and their inclusive line ranges intersect
    '''
    return a[0] == b[0] and a[1] <= b[2] and b[1] <= a[2]


def evidence_recall(retrieved, gold):
    '''
    fraction of golden evidence spans that at least one retrieved chunk overlaps
    '''
    if not gold:
        return 0.0
    return sum(any(overlaps(r, g) for r in retrieved) for g in gold) / len(gold)


def gold_spans(example):
    return [(e['file'], e['lines'][0], e['lines'][1]) for e in example['source_evidence']]


JUDGE_KEYS = {'correctness': (0, 1), 'faithfulness': (0, 1), 'completeness': (0, 5)}


def call_judge(client, model, template, question, reference, context, answer, attempts=3):
    '''
    score one answer with the course judge prompt; a judge that keeps failing scores 0 so failures
    can't inflate a config's results
    '''
    prompt = template
    for placeholder, value in [('{question}', question), ('{reference_answer}', reference),
                               ('{retrieved_context}', context), ('{system_answer}', answer)]:
        prompt = prompt.replace(placeholder, value)

    error = None
    for _ in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model, messages=[{'role': 'user', 'content': prompt}], temperature=0
            )
            text = response.choices[0].message.content
            if text is None:
                raise ValueError(f'empty judge response ({response.choices[0].finish_reason})')
            parsed = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.MULTILINE))
            scores = {}
            for key, (low, high) in JUDGE_KEYS.items():
                entry = parsed[key]
                score = int(entry['score'] if isinstance(entry, dict) else entry)
                if not low <= score <= high:
                    raise ValueError(f'{key} out of range: {score}')
                scores[key] = float(score)
            return {**scores, 'judge_failed': False}
        except Exception as e:
            error = e
    return {**{key: 0.0 for key in JUDGE_KEYS}, 'judge_failed': True, 'judge_error': str(error)}


def bootstrap_ci(values, n_resamples=2000, seed=0):
    '''
    mean with a 95% percentile-bootstrap confidence interval
    '''
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return values.mean(), *np.percentile(means, [2.5, 97.5])


def paired_bootstrap_diff(a, b, n_resamples=2000, seed=0):
    '''
    mean of (a - b) over the same questions, with a 95% CI; resampling questions keeps the pairing,
    so question difficulty cancels out of the comparison
    '''
    return bootstrap_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), n_resamples, seed)
