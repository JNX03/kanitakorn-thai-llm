---
license: cc-by-4.0
task_categories:
- text-generation
- question-answering
- text-classification
language:
- th
- en
size_categories:
- 1K<n<10K
tags:
- thai
- instruction-tuning
- mathematics
- code-generation
- reasoning
- multi-hop
- benchmark
pretty_name: Kanitakorn Thai SFT v1
---

# Kanitakorn Thai SFT v1

A 4,277-record Thai-language SFT corpus targeting the seven public Thai-LLM
benchmarks (AIME24/25-TH, MATH500-TH, LiveCodeBench-TH, OpenThaiEval,
MT-Bench-Thai, IFEval-TH, HotpotQA) plus a new `teacher_loop_th` family
of iterative student–teacher correction transcripts.

## Sources

- Generation: openai/gpt-5.5-xhigh (October 2026 snapshot)
- Human curation + fact verification via LLM-judge (gpt-5.5-xhigh) gate
- License: CC BY 4.0

## Verifier types

Each record carries a deterministic verifier (symbolic_math, unit_tests,
json_schema, regex, exact_match, retrieval_evidence) plus a fallback
llm_judge_rubric path. Audit (`tools/audit_run.py`) re-runs every verifier
on every accepted record; current state passes at 99.74% (4,277/4,288).

## Splits

| split | records |
|---|---:|
| train | 3,793 |
| validation | 422 |
| **total** | **4,225** |

Validation is stratified 10% per (family, difficulty, language) bucket.

## Family distribution

| family | train | val |
|---|---:|---:|
| aime_th | 1,161 | 129 |
| math500_th | 620 | 69 |
| livecodebench_th | 618 | 69 |
| openthaieval | 620 | 70 |
| ifeval_ifbench | 619 | 70 |
| mt_bench | 81 | 9 |
| hotpotqa_agentic | 24 + 30 seeds | 3 |
| teacher_loop_th (NEW) | 50 seeds | — |

## Citation

```bibtex
@dataset{kanitakorn_th_sft_v1,
  title = {Kanitakorn Thai SFT v1: A Verified Multi-Family Corpus for Thai LLM SFT},
  author = {Kanitakorn project},
  year = {2026},
  url = {https://huggingface.co/datasets/YOUR_USERNAME/kanitakorn-th-sft}
}
```
