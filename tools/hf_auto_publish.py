"""Auto-version HuggingFace publish for campaign adapters.

Pushes an adapter folder to Jnx03/kanitakorn-{base_short}-{stage}-{date} and
appends a structured row to VERSION_LOG.md.

Usage:
    python3 tools/hf_auto_publish.py \\
        --adapter-dir /workspace/kanitakorn/runs/campaign_v1/final \\
        --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
        --stage stage1 \\
        --dataset-sha <hex> \\
        --hyper '{"lr":1e-5,"epochs":3,"r":16}' \\
        --eval-json /workspace/.../eval_top3.json

If --eval-json is provided, the model card embeds the per-benchmark scores
and VERSION_LOG.md gets an actionable summary row.
"""
import argparse, json, os, hashlib, datetime, subprocess, sys
from pathlib import Path

from huggingface_hub import HfApi

PROJECT = Path(__file__).resolve().parents[1]
VERSION_LOG = PROJECT / "VERSION_LOG.md"

CAMPAIGN_TARGETS = {
    "thaiexam": 75.0, "math500": 82.0, "aime24": 25.0,
    "aime24_th": 15.0, "math500_th": 56.0,
    "livecodebench": 60.0, "livecodebench_th": 35.0,
    "openthaieval": 80.0, "hotpotqa": 46.0,
    "ifeval": 57.0, "mt_bench": 85.0, "ifeval_th": 82.0,
}

def base_short(base: str) -> str:
    """deepseek-ai/DeepSeek-R1-Distill-Qwen-14B -> r1d-qwen14b"""
    s = base.lower().split("/")[-1]
    s = (s.replace("deepseek-r1-distill-qwen-", "r1d-qwen")
           .replace("deepseek-r1-distill-", "r1d-")
           .replace("-instruct", "-it")
           .replace("-base", "")
           .replace("thaillm-", "thai-"))
    return s.strip("-")

def make_model_card(args, scores: dict, hyper: dict) -> str:
    today = datetime.date.today().isoformat()
    score_table = "\n".join(
        f"| {b} | {v:.2f} | {CAMPAIGN_TARGETS.get(b, '—')} | {'✅' if (CAMPAIGN_TARGETS.get(b) and v > CAMPAIGN_TARGETS[b]) else '❌' if CAMPAIGN_TARGETS.get(b) else '—'} |"
        for b, v in (scores or {}).items()
    ) or "| — | — | — | — |"
    return f"""---
library_name: peft
base_model: {args.base}
license: mit
tags:
- thai
- sft
- lora
- kanitakorn
language:
- th
- en
---

# {args.repo_id}

LoRA adapter from the Kanitakorn 7-day Thai LLM campaign.

- **Base**: `{args.base}`
- **Stage**: {args.stage}
- **Date**: {today}
- **Dataset SHA256**: `{args.dataset_sha or '—'}`

## Hyperparameters

```json
{json.dumps(hyper, indent=2)}
```

## Evaluation (top-3 benchmarks)

| Benchmark | Score | Target | Beat? |
|-----------|------:|-------:|:-----:|
{score_table}

## Recipe summary

- LoRA rank: {hyper.get('r', '—')}, alpha: {hyper.get('alpha', '—')}, dropout: {hyper.get('dropout', '—')}
- Learning rate: {hyper.get('lr', '—')}, epochs: {hyper.get('epochs', '—')}
- QLoRA 4-bit (nf4): {hyper.get('qlora', '—')}
- Mix Thai:EN: {hyper.get('mix', '—')}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "{args.base}"
adapter = "{args.repo_id}"

tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16", device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
```

## Campaign log

See [VERSION_LOG.md](https://github.com/Jnx03/kanitakorn/blob/main/VERSION_LOG.md) for context on this checkpoint vs siblings.

License: MIT (inherited from {args.base}).
"""

def append_version_log(args, scores, hyper, repo_url):
    VERSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not VERSION_LOG.exists():
        VERSION_LOG.write_text("""# Kanitakorn Version Log

One row per published adapter. Sorted oldest-first.

| Date | Repo | Stage | Base | LR | R | Epochs | THAIEXAM | MATH500 | AIME24 | Notes |
|------|------|-------|------|---:|--:|-------:|---------:|--------:|-------:|-------|
""", encoding="utf-8")
    row = (
        f"| {datetime.date.today()} "
        f"| [{args.repo_id.split('/')[-1]}]({repo_url}) "
        f"| {args.stage} | {base_short(args.base)} "
        f"| {hyper.get('lr','—')} | {hyper.get('r','—')} | {hyper.get('epochs','—')} "
        f"| {(scores or {}).get('thaiexam','—')} "
        f"| {(scores or {}).get('math500','—')} "
        f"| {(scores or {}).get('aime24','—')} "
        f"| {args.notes or '—'} |\n"
    )
    with VERSION_LOG.open("a", encoding="utf-8") as fh:
        fh.write(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True, help="Path to adapter folder (with adapter_model.safetensors)")
    ap.add_argument("--base", required=True, help="Base model HF id")
    ap.add_argument("--stage", required=True, help="Stage name (stage1, stage2-math, merge-7-3, final)")
    ap.add_argument("--dataset-sha", default=None)
    ap.add_argument("--hyper", default="{}", help="JSON dict of hyperparams")
    ap.add_argument("--eval-json", default=None, help="Path to eval results JSON (top-3+ benchmarks)")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--repo-id", default=None, help="Override auto-generated repo id")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = datetime.date.today().strftime("%Y%m%d")
    if not args.repo_id:
        args.repo_id = f"Jnx03/kanitakorn-{base_short(args.base)}-{args.stage}-{today}"

    hyper = json.loads(args.hyper)
    scores = {}
    if args.eval_json:
        try:
            e = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
            # accept either {bench: acc} or {benchmark: name, accuracy: x}
            if "benchmark" in e and "accuracy" in e:
                scores = {e["benchmark"]: round(e["accuracy"]*100 if e["accuracy"]<=1 else e["accuracy"], 2)}
            else:
                scores = {k: round(v*100 if v<=1 else v, 2) for k, v in e.items() if isinstance(v, (int, float))}
        except Exception as ex:
            print(f"[warn] could not read eval-json: {ex}")

    card = make_model_card(args, scores, hyper)
    adapter = Path(args.adapter_dir)
    if not adapter.exists():
        sys.exit(f"adapter dir missing: {adapter}")
    (adapter / "README.md").write_text(card, encoding="utf-8")

    if args.dry_run:
        print(f"[dry] would push {adapter} -> {args.repo_id}")
        print(card[:600] + "...")
        return

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo_id, private=args.private, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(adapter), repo_id=args.repo_id,
                      commit_message=f"{args.stage}: {args.notes or 'adapter checkpoint'}")
    repo_url = f"https://huggingface.co/{args.repo_id}"
    print(f"[hf] pushed {repo_url}")
    append_version_log(args, scores, hyper, repo_url)
    print(f"[log] appended to {VERSION_LOG}")

if __name__ == "__main__":
    main()
