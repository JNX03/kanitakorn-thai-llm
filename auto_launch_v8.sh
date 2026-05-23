#!/usr/bin/env bash
# Auto-launch v8 training when deps ready. Polls every 30s.
set -e
cd /root/kanitakorn
set -a; source .env.secrets; set +a
export HF_HOME=/root/.hf
export CUDA_VISIBLE_DEVICES=0,1

echo "[auto] waiting for deps..."
until python3 -c "import transformers, peft, trl, datasets, bitsandbytes" 2>/dev/null; do
    sleep 30
    echo "[auto] still waiting..."
done
echo "[auto] DEPS READY at $(date)"

# Build manifest
python3 << 'EOF'
import json, os
manifest = {
    "strategy": "weighted_v8_2xa40_ddp",
    "weights": {
        "thaiexam_synth": 0.25,
        "wangchan": 0.20,
        "typhoon_t1_thai": 0.10,
        "thai_reasoning_v2": 0.10,
        "thai_reasoning_v1": 0.05,
        "aime_th_rehearsal": 0.10,
        "math500_th_rehearsal": 0.07,
        "math_thai_v3": 0.05,
        "ifeval_th_v3": 0.04,
        "openthaieval_orig": 0.02,
        "teacher_loop_th": 0.02,
    },
    "sft_files": {
        "thaiexam_synth": {"train": "sft_ready_thai_v7/thaiexam_FINAL_v2_train.jsonl"},
        "wangchan": {"train": "sft_ready_external/wangchan_train.jsonl"},
        "typhoon_t1_thai": {"train": "sft_ready_external/typhoon_t1_thai_train.jsonl"},
        "thai_reasoning_v2": {"train": "train/train_thai_reasoning_v2.jsonl"},
        "thai_reasoning_v1": {"train": "train/train_thai_reasoning_v1.jsonl"},
        "aime_th_rehearsal": {"train": "sft_ready/aime_th_train.jsonl"},
        "math500_th_rehearsal": {"train": "sft_ready/math500_th_train.jsonl"},
        "math_thai_v3": {"train": "train/train_math_thai_v3.jsonl"},
        "ifeval_th_v3": {"train": "train/train_ifeval_th_v3.jsonl"},
        "openthaieval_orig": {"train": "sft_ready/openthaieval_train.jsonl"},
        "teacher_loop_th": {"train": "sft_ready/teacher_loop_th_train.jsonl"},
    }
}
os.makedirs("/root/kanitakorn/dataset/sft_ready_thai_v8", exist_ok=True)
with open("/root/kanitakorn/dataset/sft_ready_thai_v8/manifest_v8.json","w") as f:
    json.dump(manifest, f, indent=2)
print("v8 manifest written")
EOF

# Launch DDP training
rm -rf /root/kanitakorn/runs/r1d_kanitakorn_v8
rm -f /root/kanitakorn/_state/train_done

export SFT_BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
export SFT_OUT=/root/kanitakorn/runs/r1d_kanitakorn_v8
export SFT_EPOCHS=0.7
export SFT_BS=1
export SFT_GA=8
export SFT_MAX_LEN=2048
export SFT_SAVE_STEPS=75
export SFT_LORA_R=16
export SFT_LR=2e-6
export SFT_QLORA=1
export SFT_MIX_EN=0
export SFT_PROJECT=/root/kanitakorn
export SFT_MANIFEST=/root/kanitakorn/dataset/sft_ready_thai_v8/manifest_v8.json
export SFT_THAI_TOTAL=8000

echo "[auto] launching DDP train at $(date)"
torchrun --nproc_per_node=2 train_2xa100_interruptible.py 2>&1 | tee /root/kanitakorn/train_v8.log
echo "[auto] v8 DONE at $(date)"
