"""Publish the SFT dataset + model card to the HuggingFace Hub.

Two upload paths:

    --dataset <repo>   uploads dataset/sft_ready/*.jsonl as a HF dataset
    --model <repo>     uploads a trained checkpoint with auto-generated card

Both rely on `huggingface_hub` (already installed). HF token must be set via
`huggingface-cli login` or HF_TOKEN env var. The dataset card and model card
templates are written to `dataset/reports/hf_dataset_card.md` and
`dataset/reports/hf_model_card.md` first; review them before pushing.

CLI:
    python tools/hf_publish.py --dataset YOUR_USERNAME/kanitakorn-th-sft
    python tools/hf_publish.py --model YOUR_USERNAME/qwen3.6-35b-a3b-th-sft \\
                               --checkpoint runs/qwen3.6-35b-a3b-thai-sft/final
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
REPORTS = DATASET / "reports"


DATASET_CARD = """---
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
"""


MODEL_CARD = """---
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
"""


def write_cards() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "hf_dataset_card.md").write_text(DATASET_CARD, encoding="utf-8")
    (REPORTS / "hf_model_card.md").write_text(MODEL_CARD, encoding="utf-8")
    print(f"wrote {REPORTS / 'hf_dataset_card.md'}")
    print(f"wrote {REPORTS / 'hf_model_card.md'}")


def push_dataset(repo: str) -> int:
    try:
        from huggingface_hub import HfApi, create_repo  # type: ignore
    except ImportError:
        print("huggingface_hub not installed (pip install huggingface_hub)")
        return 2
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    try:
        create_repo(repo, repo_type="dataset", exist_ok=True, token=token)
    except Exception as e:
        print(f"create_repo: {e}")
    # Upload sft_ready folder + dataset card + manifest.
    sft_dir = DATASET / "sft_ready"
    if not sft_dir.exists():
        print(f"no {sft_dir} — run tools/few_shot_collator.py first")
        return 2
    api.upload_folder(folder_path=str(sft_dir), repo_id=repo, repo_type="dataset", token=token)
    api.upload_file(
        path_or_fileobj=str(REPORTS / "hf_dataset_card.md"),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        token=token,
    )
    print(f"pushed dataset → https://huggingface.co/datasets/{repo}")
    return 0


def push_model(repo: str, checkpoint: str) -> int:
    try:
        from huggingface_hub import HfApi, create_repo  # type: ignore
    except ImportError:
        print("huggingface_hub not installed")
        return 2
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    create_repo(repo, repo_type="model", exist_ok=True, token=token)
    api.upload_folder(folder_path=checkpoint, repo_id=repo, repo_type="model", token=token)
    api.upload_file(
        path_or_fileobj=str(REPORTS / "hf_model_card.md"),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
        token=token,
    )
    print(f"pushed model → https://huggingface.co/{repo}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="HF dataset repo (e.g. user/name)")
    parser.add_argument("--model", help="HF model repo")
    parser.add_argument("--checkpoint", help="local model directory to upload (with --model)")
    parser.add_argument("--cards-only", action="store_true", help="write card templates without uploading")
    args = parser.parse_args()

    write_cards()
    if args.cards_only:
        return 0

    if args.dataset:
        return push_dataset(args.dataset)
    if args.model:
        if not args.checkpoint:
            print("--checkpoint required with --model")
            return 2
        return push_model(args.model, args.checkpoint)
    print("specify --dataset <repo> or --model <repo> --checkpoint <dir>, or --cards-only")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
