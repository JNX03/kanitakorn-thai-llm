#!/usr/bin/env bash
# Idempotent boot + train pipeline for vast.ai interruptible instances.
# Re-run after every restart. Skips steps already completed.
#
# Usage on remote:
#   bash /root/kanitakorn/boot_and_train.sh
set -e

cd /root/kanitakorn
# HF_TOKEN and OPENROUTER_API_KEY must be exported in the parent env (e.g. by tmux session creator)
export HF_TOKEN="${HF_TOKEN:?must export HF_TOKEN before running this script}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
export HF_HOME=/root/.hf

STATE_DIR=/root/kanitakorn/_state
mkdir -p "$STATE_DIR"

# Step 1: deps (idempotent — pip is fast if already installed)
if [ ! -f "$STATE_DIR/deps_ok" ]; then
    echo "=== [$(date)] STEP 1: deps ==="
    python3 -c "import torch, transformers, peft, trl, bitsandbytes, datasets, sentence_transformers, requests" 2>&1 || \
        pip install --break-system-packages -q --upgrade transformers peft trl datasets safetensors huggingface_hub accelerate bitsandbytes sentencepiece scipy jsonschema sentence-transformers requests
    touch "$STATE_DIR/deps_ok"
fi

# Step 2: model download (idempotent via HF cache)
if [ ! -f "$STATE_DIR/model_ok" ]; then
    echo "=== [$(date)] STEP 2: download R1-Distill-14B ==="
    python3 -c "
from huggingface_hub import snapshot_download
import os
p = snapshot_download(repo_id='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B', token=os.environ.get('HF_TOKEN'))
print('model at:', p)
"
    touch "$STATE_DIR/model_ok"
fi

# Step 3: rebuild SFT records for R1-Distill chat template
if [ ! -f "$STATE_DIR/rebuild_ok" ]; then
    echo "=== [$(date)] STEP 3: rebuild SFT records ==="
    python3 tools/rebuild_for_base.py \
        --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --in-glob "dataset/sft_ready/*_train.jsonl" \
        --out-dir dataset/sft_ready_r1d14b \
        --include-distill
    touch "$STATE_DIR/rebuild_ok"
fi

# Step 4: baseline eval (skip if already done)
mkdir -p reports
if [ ! -f "$STATE_DIR/baseline_ok" ]; then
    echo "=== [$(date)] STEP 4: baseline eval (top-3) ==="
    for bench in thaiexam math500 aime24; do
        if [ ! -f "reports/base_r1d14b_${bench}.json" ]; then
            n_limit=100
            [ "$bench" = "aime24" ] && n_limit=30
            echo "  [baseline] $bench (limit=$n_limit)"
            python3 eval_self_consistency.py \
                --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
                --benchmark $bench --n 1 --limit $n_limit \
                --out reports/base_r1d14b_${bench}.json 2>&1 | tail -5 || echo "  [baseline] $bench FAILED"
        fi
    done
    touch "$STATE_DIR/baseline_ok"
fi

# Step 5: full Stage B SFT training
if [ ! -f "$STATE_DIR/train_done" ]; then
    echo "=== [$(date)] STEP 5: Stage B SFT training ==="
    export SFT_BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    export SFT_OUT=/root/kanitakorn/runs/stage1_v1
    export SFT_EPOCHS=2
    export SFT_LORA_R=16
    export SFT_MIX_EN=1
    export SFT_QLORA=1
    export SFT_PROJECT=/root/kanitakorn
    export SFT_MANIFEST=/root/kanitakorn/dataset/sft_ready_r1d14b/manifest.json

    NGPU=$(nvidia-smi -L | wc -l)
    while [ ! -f "$STATE_DIR/train_done" ]; do
        echo "=== [$(date)] training attempt (ngpu=$NGPU) ==="
        if [ $NGPU -ge 2 ]; then
            torchrun --nproc_per_node=$NGPU --master_port=29501 train_2xa100_interruptible.py 2>&1 | tee -a /root/kanitakorn/train.log
        else
            python3 train_2xa100_interruptible.py 2>&1 | tee -a /root/kanitakorn/train.log
        fi
        ec=${PIPESTATUS[0]}
        if [ $ec -eq 0 ]; then
            touch "$STATE_DIR/train_done"
            echo "=== [$(date)] training COMPLETE ==="
        else
            echo "=== [$(date)] training failed ec=$ec, retry in 30s ==="
            sleep 30
        fi
    done
fi

# Step 6: post-train eval
if [ ! -f "$STATE_DIR/eval_done" ]; then
    echo "=== [$(date)] STEP 6: post-train eval ==="
    for bench in thaiexam math500 aime24; do
        n_limit=100
        [ "$bench" = "aime24" ] && n_limit=30
        python3 eval_self_consistency.py \
            --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
            --adapter runs/stage1_v1/final \
            --benchmark $bench --n 1 --limit $n_limit \
            --out reports/v1_${bench}.json 2>&1 | tail -5
    done
    touch "$STATE_DIR/eval_done"
fi

# Step 7: push to HF
if [ ! -f "$STATE_DIR/push_done" ]; then
    echo "=== [$(date)] STEP 7: push adapter to HF ==="
    # Combine top-3 scores into one json for the model card
    python3 -c "
import json
scores = {}
for b in ['thaiexam', 'math500', 'aime24']:
    try:
        d = json.load(open(f'reports/v1_{b}.json'))
        scores[b] = d['accuracy']
    except: scores[b] = None
json.dump(scores, open('reports/v1_combined.json', 'w'))
print('combined:', scores)
"
    python3 tools/hf_auto_publish.py \
        --adapter-dir runs/stage1_v1/final \
        --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --stage stage1-v1 \
        --hyper '{"lr":1e-5,"epochs":2,"r":16,"qlora":true,"mix":"30:70"}' \
        --eval-json reports/v1_combined.json \
        --notes "Stage B: R1-Distill-14B + Thai SFT + R1-distill mix"
    touch "$STATE_DIR/push_done"
fi

echo "=== [$(date)] ALL STAGES COMPLETE ==="
