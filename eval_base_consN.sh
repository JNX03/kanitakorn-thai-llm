#!/usr/bin/env bash
# Fallback: skip SFT entirely. Use R1-Distill base with Cons@N self-consistency.
# Per DeepSeek paper: R1-Distill-Qwen-14B Cons@64 = 80% AIME24 (vs 69.7 single).
#
# Strategy: if SFT damages math reasoning, just use base with Maj@N voting.
# Tradeoff: ThaiExam will be ~30-45% (no Thai SFT), but math/aime crushed.
#
# Usage on instance:
#   bash eval_base_consN.sh
set -e
cd /root/kanitakorn
set -a; source .env.secrets; set +a
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/root/.hf

BASE="${BASE:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
N="${N:-16}"  # smaller N to save compute; can go to 64 for max
TAG="${TAG:-base-cons${N}}"

mkdir -p reports

for bench in math500 aime24 thaiexam; do
    out="reports/${TAG}_${bench}.json"
    if [ -f "$out" ]; then echo "[skip] $out exists"; continue; fi
    case "$bench" in
        aime24)  limit=30 ;;
        math500) limit=200 ;;  # subset for speed
        thaiexam) limit=200 ;;
    esac
    echo "=== [$(date)] $bench n=$N limit=$limit ==="
    python3 eval_vllm.py \
        --base "$BASE" \
        --benchmark $bench --n $N --limit $limit \
        --max-tokens 4096 --tp 1 \
        --out "$out" 2>&1 | tail -10
done

python3 -c "
import json, os
print(f'=== BASE Cons@${N} (no SFT) ===')
print(f\"{'Benchmark':15s}{'Score':>10s}{'Target':>10s}  Beat?\")
print('-'*50)
for b, t in [('math500', 82), ('aime24', 25), ('thaiexam', 75)]:
    p = f'reports/${TAG}_{b}.json'
    if not os.path.exists(p): continue
    d = json.load(open(p))
    v = d.get('accuracy', 0) * 100
    mark = '✅' if v > t else '❌'
    print(f'{b:15s}{v:>10.2f}{t:>10}   {mark}')
"
