#!/usr/bin/env bash
# Bootstrap the Kanitakorn project on a fresh A100 machine (RunPod / Lambda /
# Vast / your own box). Run after `git clone` of this repo.
#
# Usage:
#   bash bootstrap_a100.sh
#
# Steps it performs:
#   1. Install Python deps (transformers, datasets, peft, trl, accelerate,
#      bitsandbytes, sympy, jsonschema, pythainlp, openai)
#   2. Verify CUDA + reasonable VRAM (warn if < 70GB available)
#   3. Re-emit the validation split (rebalance_validation_split.py)
#   4. Re-export the 10,579 public benchmark inputs from HF cache
#      (`benchmark_eval.py --inputs-only`)
#   5. Run audit_run.py to confirm Phase-0 gate passes
#   6. Print a "you're ready to train" status with the exact next commands

set -e

echo "=== 1. Install deps ==="
pip install -q --upgrade pip
pip install -q \
    "torch>=2.3" \
    "transformers>=4.45" \
    "datasets>=3.0" \
    "accelerate>=0.30" \
    "peft>=0.10" \
    "trl>=0.10" \
    "bitsandbytes>=0.43" \
    "huggingface_hub>=0.24" \
    "openai>=1.40" \
    "anthropic>=0.30" \
    "sympy>=1.13" \
    "jsonschema>=4.22" \
    "pythainlp>=5.0" \
    "sentencepiece" \
    "pyarrow"

echo ""
echo "=== 2. CUDA check ==="
python -c "
import torch
print(f'  torch       : {torch.__version__}')
print(f'  cuda avail  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / 1024**3
        print(f'  GPU {i}      : {p.name} — {gb:.1f} GB')
        if gb < 70:
            print(f'  WARN: GPU {i} has {gb:.1f} GB — Qwen3.6-35B-A3B needs ~70GB+ for full SFT. Use LoRA or smaller batch.')
"

echo ""
echo "=== 3. Re-emit validation split (stratified 10%) ==="
python tools/rebalance_validation_split.py

echo ""
echo "=== 4. Export 10,579 public benchmark inputs (from HF cache) ==="
python tools/benchmark_eval.py --inputs-only dataset/reports/benchmark_inputs.jsonl --family all
echo "  (this is the file gitignored — regenerated here from cached HF datasets)"

echo ""
echo "=== 5. Audit (Phase-0 gate) ==="
python tools/audit_run.py --no-prosody

echo ""
echo "=== 6. Lock state against any sync revert ==="
python tools/lock_state.py 2>/dev/null || echo "  (lock_state is a no-op on Linux; only matters on Windows/OneDrive)"

echo ""
echo "============================================================"
echo "BOOTSTRAP COMPLETE — ready to train + evaluate."
echo "============================================================"
echo ""
echo "Next steps (per RUNBOOK.md):"
echo ""
echo "  # If you want more teacher-loop records (set OPENAI_API_KEY first)"
echo "  export OPENAI_API_KEY=sk-..."
echo "  python tools/teacher_loop_generator.py --skill klon_4 --count 200"
echo "  python tools/teacher_loop_generator.py --skill register --count 200"
echo ""
echo "  # Package SFT corpus"
echo "  python tools/few_shot_collator.py"
echo "  python tools/build_train_manifest.py"
echo ""
echo "  # Train (see RUNBOOK.md step 4 for the training script template)"
echo "  # ... your trainer here ..."
echo ""
echo "  # Inference + scoring"
echo "  python tools/run_inference.py --backend hf-local --model runs/.../final \\"
echo "      --inputs dataset/reports/benchmark_inputs.jsonl \\"
echo "      --out dataset/reports/predictions.jsonl"
echo "  python tools/benchmark_eval.py --score-from dataset/reports/predictions.jsonl \\"
echo "      --model qwen3.6-35b-a3b-th-sft"
echo ""
echo "  # Push trained model to HF"
echo "  python tools/hf_publish.py --model YOUR_USERNAME/qwen3.6-35b-a3b-th-sft \\"
echo "      --checkpoint runs/.../final"
