---
license: apache-2.0
language:
- th
- en
base_model: Qwen/Qwen3.6-35B-A3B-Instruct
pipeline_tag: text-generation
tags:
- thai
- sft
- instruction-tuning
- mathematics
- code-generation
---

# Qwen3.6-35B-A3B-TH-SFT

Fine-tuned from `Qwen/Qwen3.6-35B-A3B-Instruct` on the Kanitakorn Thai SFT
v1 corpus + the novel `teacher_loop_th` family (iterative student-teacher
correction transcripts).

## Training

- Base: Qwen/Qwen3.6-35B-A3B-Instruct
- Method: LoRA SFT (r=16, alpha=32, target q_proj/k_proj/v_proj/o_proj)
- Epochs: 1 over 3,793 train records + 50 teacher-loop seeds
- Hardware: 1× A100 SXM4 80GB
- Sampling: sqrt-balanced per family (see manifest)

## Evaluation

| benchmark | this model | Typhoon-2-8B | OpenThaiGPT-1.5 7B | delta |
|---|---:|---:|---:|---:|
| math500 | TBD | 0.490 | not published | TBD |
| ifeval-th | TBD | 0.726 | not published | TBD |
| mt-bench-th | TBD | 5.74 | not published | TBD |
| openthaieval | TBD | not published | 0.658 | TBD |
| ... | | | | |

Full report: `dataset/reports/benchmark_eval_<this-model>.md`.

## Intended use

- Thai academic tutoring (O-NET, A-Level, TGAT, TPAT)
- Thai math contest reasoning (AIME-style)
- Thai literature / prosody (กลอน, กาพย์, ฉันท์)
- Multi-hop reasoning with citation
- Instruction following in Thai + English

## Limitations

- Trained on 4k records; not a from-scratch Thai LM
- HotpotQA-agentic family is the weakest (54 records); multi-hop performance
  may lag dedicated retrieval-augmented systems
- Fact-grounding relies on the citation chain in `sources` field;
  hallucinations on out-of-distribution facts are possible

## Citation

```bibtex
@model{kanitakorn_qwen35b_th_v1,
  title = {Qwen3.6-35B-A3B-TH-SFT: Thai-Specialty Instruction-Tuned Qwen via Teacher-Loop Method},
  author = {Kanitakorn project},
  year = {2026},
  base_model = {Qwen/Qwen3.6-35B-A3B-Instruct},
  dataset = {kanitakorn-th-sft-v1}
}
```
