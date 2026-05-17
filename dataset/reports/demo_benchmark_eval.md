# Benchmark eval — demo-gold-as-prediction

Public-benchmark scores for `demo-gold-as-prediction` vs Typhoon-2 / OpenThaiGPT-1.5 published baselines.

| benchmark | score | typhoon-2 | Δ vs typhoon-2 | openthaigpt-1.5 | Δ vs openthaigpt-1.5 |
|---|---:|---:|---:|---:|---:|
| aime24_th | 1.000 | not published | — | not published | — |
| math500_th | 1.000 | 0.490 | +0.510 | not published | — |
| ifeval_th_strict | 1.000 | not published | — | not published | — |
| openthaieval_overall | 1.000 | not published | — | 0.658 | +0.342 |

## Raw results

```json
{
  "aime24": {
    "accuracy": 1.0,
    "n": 3,
    "sympy": 0,
    "llm": 0
  },
  "math500": {
    "accuracy": 1.0,
    "n": 3,
    "sympy": 0,
    "llm": 0
  },
  "ifeval": {
    "strict_acc": 1.0,
    "loose_acc": 1.0,
    "n": 3
  },
  "openthaieval": {
    "overall": 1.0,
    "mcq": 1.0,
    "analytic": 0.0,
    "by_subject": {
      "thai_professional_qa": 1.0,
      "science_multiple_choice_source_backed": 1.0
    }
  }
}
```