#!/usr/bin/env bash
# Launch the 7-day campaign on 2x A100 40GB (vast.ai interruptible).
# Usage on remote instance:
#   bash fire_2xa100.sh
#
# This script is idempotent — safe to re-run after preemption. Training auto-resumes.

set -e
cd /workspace/kanitakorn

export HF_TOKEN="${HF_TOKEN:?must set HF_TOKEN}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
export HF_HOME=/workspace/.hf
mkdir -p "$HF_HOME"

# --- Step 0: env sanity ----------------------------------------------------
echo "=== nvidia-smi ==="
nvidia-smi -L
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "Driver: $DRIVER"

# --- Step 1: pin torch to driver's CUDA -----------------------------------
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "[setup] installing torch+cu124 (driver-matched)"
    pip install -q torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
}
python3 -c "import torch; print('torch=', torch.__version__, 'cuda_ok=', torch.cuda.is_available(), 'bf16=', torch.cuda.is_bf16_supported())"

# --- Step 2: deps ----------------------------------------------------------
pip install -q --upgrade transformers peft trl datasets safetensors huggingface_hub accelerate bitsandbytes sentencepiece scipy

# Unsloth (optional fast path) — install if requested
if [ "${SFT_USE_UNSLOTH:-0}" = "1" ]; then
    echo "[setup] installing unsloth fast-path"
    pip install -q --upgrade "unsloth[cu124]" "unsloth_zoo" || echo "[warn] unsloth install failed, will fall back to standard PEFT"
fi

# --- Step 3: smoke test the eval pipeline FIRST (5 min) --------------------
echo "=== eval smoke test ==="
python3 - <<'EOF'
import datasets
# Confirm we can hit all benchmark datasets from HF
for spec in [
    ("math-ai/aime24", None),
    ("HuggingFaceH4/MATH-500", None),
    ("scb10x/thai_exam", None),
]:
    try:
        ds = datasets.load_dataset(*[s for s in spec if s], split="train", streaming=True)
        print(spec[0], "OK", next(iter(ds)).keys())
    except Exception as e:
        print(spec[0], "FAIL:", str(e)[:120])
EOF

# --- Step 4: launch SFT in tmux ------------------------------------------
NGPU=$(nvidia-smi -L | wc -l)
echo "=== launching SFT on $NGPU GPU(s) ==="

tmux kill-session -t train 2>/dev/null || true

if [ "$NGPU" -ge 2 ]; then
    LAUNCH="torchrun --nproc_per_node=$NGPU --master_port=29501 train_2xa100_interruptible.py"
else
    LAUNCH="python3 train_2xa100_interruptible.py"
fi

tmux new -d -s train "
    export HF_TOKEN='$HF_TOKEN'
    export OPENROUTER_API_KEY='$OPENROUTER_API_KEY'
    export HF_HOME='$HF_HOME'
    export SFT_BASE='${SFT_BASE:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}'
    export SFT_OUT='${SFT_OUT:-/workspace/kanitakorn/runs/campaign_v1}'
    export SFT_EPOCHS='${SFT_EPOCHS:-3}'
    export SFT_LORA_R='${SFT_LORA_R:-16}'
    export SFT_MIX_EN='${SFT_MIX_EN:-1}'
    export SFT_QLORA='${SFT_QLORA:-auto}'
    export SFT_USE_UNSLOTH='${SFT_USE_UNSLOTH:-0}'
    cd /workspace/kanitakorn
    while true; do
        echo \"[\$(date)] starting training (resume-aware)\"
        $LAUNCH 2>&1 | tee -a /tmp/train_campaign.log
        ec=\${PIPESTATUS[0]}
        echo \"[\$(date)] exit=\$ec\"
        if [ \$ec -eq 0 ]; then
            echo \"[\$(date)] training COMPLETED — exiting loop\"
            break
        fi
        echo \"[\$(date)] training failed; sleeping 60s before retry\"
        sleep 60
    done
"

sleep 3
tmux ls
echo ""
echo "=== Launched. Monitor with: tail -F /tmp/train_campaign.log"
echo "===                 or:    tmux attach -t train"
