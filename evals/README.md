# Situation quality evaluation

This directory contains the reproducible second-stage Cerebras quality check for
the situation-aware generation flow.

## Run

Run from the `TongChoo-AI` directory with `CEREBRAS_API_KEY` configured:

```bash
.venv/bin/python evals/evaluate_situations.py --concurrency 2
```

Use `--ids case_id ...` for a focused rerun or `--limit N` for a smoke test.
The evaluator records classification accuracy, sentence and character fit,
grounding violations, regeneration rate, and classification/total latency.

## 2026-07-14 result

The final run used 21 synthetic cases: seven each for LIGHT, NORMAL, and SERIOUS.
It included boundary cases such as a five-minute lunch delay with a team lead and
a friend's request that caused financial loss.

| Metric | Result | Target | Status |
| --- | ---: | ---: | --- |
| Severity accuracy | 95.2% | at least 90% | Pass |
| Returned result fit | 100% | at least 90% | Pass |
| Regeneration rate | 15.0% | at most 30% | Pass |
| SERIOUS with at least 3 sentences | 100% | at least 90% | Pass |
| LIGHT with at most 2 sentences | 100% | at least 90% | Pass |
| SERIOUS humor violations | 0 | 0 | Pass |
| Returned grounding violations | 0 | 0 | Pass |
| Average total latency | 1,479 ms | observation | — |
| P95 total latency | 2,166 ms | observation | — |

Twenty cases returned a valid result. One SERIOUS production-backup case was
rejected with `GROUNDING_QUALITY_REJECTED` after both candidates asserted service
errors or user impact that the input never stated. This is an intentional safety
rejection and is counted as a severity mismatch in the conservative 95.2% score.

The complete provider outputs and per-case metrics are stored in
`latest-situation-report.json`.
