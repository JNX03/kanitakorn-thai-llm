# Runbook — train + evaluate a Thai LLM that beats Typhoon-2 / OpenThaiGPT-1.5

This is the end-to-end recipe. The infrastructure is built and verified —
this runbook tells you the exact commands to produce the empirical numbers.

## Prerequisites

- Python 3.13+ with this project's deps already installed (`transformers`,
  `datasets`, `openai`, `pythainlp`, `jsonschema`, `sympy`)
- 1× H100 (or 4× A100 40GB) for full SFT on Qwen3.6-35B-A3B; a single 24GB
  GPU works for LoRA on Qwen2.5-7B as a faster smoke test
- `OPENAI_API_KEY` if you'll regenerate teacher-loop records or use the LLM
  judge on free-form items
- **Pause OneDrive/iCloud sync on `dataset\`** before any production run.
  This session observed silent file reverts three times. Either move
  `dataset/` outside the synced folder OR run `python tools/lock_state.py`
  which sets the files read-only.

## Step 0 — Lock the dataset

```
python tools/lock_state.py
```

Output: 4,225 records, val=422, teacher_loop_th=50 seed. A snapshot lives at
`tools/_locked_snapshot/`; `python tools/_locked_snapshot/restore.py`
restores it after any future revert.

## Step 1 — Validate

```
python tools/audit_run.py --no-prosody
python tools/dataset_quality_analysis.py
```

Expected: `overall: 4214/4225 (99.74%) — Phase-0 gate PASSED`.

`dataset_quality_analysis.py` writes `dataset/reports/dataset_quality_analysis.md`
— a statistical cross-reference of our corpus vs the 10,014 public-benchmark
inputs (coverage ratios, prompt-length distributions, lexical diversity, token
budget estimates, side-by-side prompt samples). Surfaces real gaps: today's
output shows livecodebench_th prompts are too short vs LCB-public (median 297
vs 1232 chars), aime_th is over-represented at 21×, hotpotqa is under-
represented at 0.004×.

## Step 1.5 — Establish a real baseline (optional, ~10 min on CPU)

```
python tools/run_tiny_baseline.py
```

Runs Qwen2.5-0.5B-Instruct (downloads ~1GB on first run) on a 50-record
subset across 5 benchmark families and writes
`dataset/reports/tiny_baseline_report.md` — real (non-gold) inference
scores. Use as the floor your trained 35B model must beat by a wide margin.

## Step 2 — Generate the teacher-loop corpus at scale (optional but recommended)

```
$env:OPENAI_API_KEY = "sk-..."
python tools/teacher_loop_generator.py --skill klon_4   --count 200
python tools/teacher_loop_generator.py --skill register --count 200
python tools/audit_run.py --no-prosody          # confirm new records pass
```

Target: 450 generated + 50 seed = 500 total `teacher_loop_th` records.
Diversity guard requires ≥30% with `n_corrections ≥ 2` (script enforces).

Cost estimate: ~$5–15 with `gpt5.5-xhigh` at the wired-in pricing
constants. Cap with `MAX_JUDGE_USD=20` env var if you want a hard limit.

## Step 3 — Package SFT data

```
python tools/few_shot_collator.py
python tools/build_train_manifest.py
```

Output: `dataset/sft_ready/{family}_{split}.jsonl` (Qwen chat template) +
`dataset/sft_ready/manifest.json` (sqrt-balanced weights).

## Step 4 — Train

This step is your training stack. Suggested setup (PEFT / LoRA):

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import json

manifest = json.load(open("dataset/sft_ready/manifest.json"))
ds = load_dataset("json", data_files={
    family: f"dataset/sft_ready/{family}_train.jsonl"
    for family in manifest["weights"]
})
# Apply manifest["weights"] in your sampler.

base = "Qwen/Qwen3.6-35B-A3B-Instruct"      # or "google/gemma-4-7b-instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"])
model = get_peft_model(model, lora)

cfg = SFTConfig(
    output_dir="runs/qwen3.6-35b-a3b-thai-sft",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    bf16=True,
    save_strategy="epoch",
)
trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=..., args=cfg)
trainer.train()
trainer.save_model("runs/qwen3.6-35b-a3b-thai-sft/final")
```

Expected runtime: ~12–18 hours on 1× H100 for 1 epoch over 3,793 train
records + 50 teacher-loop records.

## Step 5 — Run inference

The 10,579 public benchmark inputs are already exported at
`dataset/reports/benchmark_inputs.jsonl` (auto-loaded from your HF cache).

This now includes **ThaiExam** — the canonical Thai exam benchmark that
Typhoon-2 and OpenThaiGPT-1.5 publish on (5 subjects: ONET 162, IC 95,
TGAT 65, TPAT-1 116, A-Level 127 = 565 test rows). Apples-to-apples
comparison vs Typhoon-2-8B (0.5120) and OpenThaiGPT-1.5 7B (0.5204) is
now possible without any extra baseline runs.
If you need to refresh:

```
python tools/benchmark_eval.py --inputs-only dataset/reports/benchmark_inputs.jsonl --family all
```

Pipe them through your trained model:

```
python tools/run_inference.py --backend hf-local --model runs/qwen3.6-35b-a3b-thai-sft/final \
    --inputs dataset/reports/benchmark_inputs.jsonl \
    --out dataset/reports/predictions.jsonl
```

Expected runtime: ~3–6 hours on 1× H100 for 10K inputs at temperature 0.0.

## Step 6 — Score and compare

```
python tools/benchmark_eval.py --score-from dataset/reports/predictions.jsonl \
    --model qwen3.6-35b-a3b-thai-sft
```

Output: `dataset/reports/benchmark_eval_qwen3.6-35b-a3b-thai-sft.md` —
markdown table with Δ vs Typhoon-2 and Δ vs OpenThaiGPT-1.5 per family.

## Step 7 — Baseline pass for the unpublished benchmarks

Typhoon-2 / OpenThaiGPT-1.5 don't publish AIME24-TH, AIME25-TH,
LiveCodeBench-TH, or HotpotQA scores. To get apples-to-apples comparison
on those four:

```
# Download Typhoon-2-8B-Instruct
python tools/run_inference.py --backend hf-local --model scb10x/typhoon-2-8b-instruct \
    --inputs dataset/reports/benchmark_inputs.jsonl --out dataset/reports/typhoon2_predictions.jsonl

python tools/benchmark_eval.py --score-from dataset/reports/typhoon2_predictions.jsonl --model typhoon-2-8b
# This fills in the 'not published' rows with our own measurements.

# Then update the constants at the top of tools/benchmark_eval.py with the
# numbers from typhoon2_predictions to make all future delta reports honest.
```

## Step 8 — Verify "search every data" (optional)

The user explicitly asked for search-grounded validation:

```
$env:TAVILY_API_KEY = "..."        # or BRAVE_API_KEY
python tools/web_search_verifier.py --check-record dataset/train/train_aime_th_010.jsonl --limit 10
```

Reports which records' `final_answer` strings literally appear in top-5 web
search results — a soft signal that the answer is correct.

## Expected gates

| gate | metric | threshold |
|---|---|---|
| Phase 0 audit | per-family pass | ≥ 99% (gate at 99.74% today) |
| Validation split | val records / family | ≥ 10% (today: 422/4225 = 10.0%) |
| Teacher-loop scale | `teacher_loop_th` record count | ≥ 500 (today: 50 seed) |
| Teacher-loop diversity | % with n_corrections ≥ 2 | ≥ 30% |
| Beat Typhoon-2 (8B) | math500_th accuracy | > 0.49 |
| Beat Typhoon-2 (8B) | ifeval_th overall | > 0.73 |
| Beat Typhoon-2 (8B) | mt_bench_th overall | > 5.74 |
| Beat OpenThaiGPT-1.5 (7B) | openthaieval overall | > 0.66 |
| Beat Typhoon-2 (70B) | thai_exam | > 0.63 |

## Troubleshooting

- **"OPENAI_API_KEY not set"** in `teacher_loop_generator.py` — set it; the
  generator drives student + teacher both through gpt-5.5-xhigh.
- **OneDrive reverts files** — run `python tools/_locked_snapshot/restore.py`
  then `python tools/lock_state.py` again; or move `dataset/` out of sync.
- **"Dataset scripts are no longer supported"** in `datasets` library —
  the openthaieval path falls back to direct parquet reads from the HF
  cache; works without code changes if the parquet is cached.
- **CUDA OOM on Qwen3.6-35B** — drop to per-device batch size 1, gradient
  checkpointing, or use the Gemma-4 alternative.
- **livecodebench judge fails on Windows** — `resource.setrlimit` is
  POSIX-only; the judge degrades to timeout-only sandboxing on Windows.
  Run code-gen eval on Linux or in a Vercel Sandbox for full isolation.
