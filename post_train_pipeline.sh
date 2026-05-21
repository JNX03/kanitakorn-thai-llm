#!/usr/bin/env bash
# Post-training pipeline: eval top-3 with vLLM, push to HF, update VERSION_LOG.
#
# Usage on instance (after training finishes):
#   bash post_train_pipeline.sh
set -e

cd /root/kanitakorn
[ -f .env.secrets ] && { set -a; source .env.secrets; set +a; }
export HF_HOME=/root/.hf
export CUDA_VISIBLE_DEVICES=0

ADAPTER="${ADAPTER:-/root/kanitakorn/runs/stage1_v1/final}"
BASE="${BASE:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
STAGE="${STAGE:-stage1-v1}"
TAG="${TAG:-v1}"

mkdir -p reports

if [ ! -d "$ADAPTER" ]; then
    echo "[err] adapter dir not found: $ADAPTER"
    echo "[err] available: $(ls runs/ 2>&1)"
    exit 1
fi

# Step 1: vLLM-based eval on top-3 (full test sets)
echo "=== [$(date)] STEP 1: vLLM eval ==="
for bench in math500 aime24 thaiexam; do
    out="reports/${TAG}_${bench}.json"
    if [ -f "$out" ]; then echo "  [skip] $out exists"; continue; fi
    case "$bench" in
        aime24)  limit=30 ;;
        math500) limit=500 ;;
        thaiexam) limit=565 ;;
    esac
    echo "  [eval] $bench (full $limit)"
    python3 eval_vllm.py \
        --base "$BASE" --adapter "$ADAPTER" \
        --benchmark $bench --n 1 --limit $limit \
        --max-tokens 4096 --tp 1 \
        --out "$out" 2>&1 | tail -10
done

# Step 2: Combine scores
python3 -c "
import json, os
scores = {}
for b in ['thaiexam','math500','aime24']:
    p = f'reports/${TAG}_{b}.json'
    if os.path.exists(p):
        d = json.load(open(p))
        v = d.get('accuracy')
        scores[b] = round(v*100 if v and v<=1 else v, 2) if v is not None else None
json.dump(scores, open('reports/${TAG}_combined.json','w'))
print('=== ${TAG} top-3 ===')
for b, v in scores.items():
    target = {'thaiexam':75,'math500':82,'aime24':25}[b]
    mark = '✅' if (v and v > target) else '❌'
    print(f'  {b:12s}: {v} {mark} (target {target})')
"

# Step 3: HF push
echo "=== [$(date)] STEP 3: HF push ==="
python3 tools/hf_auto_publish.py \
    --adapter-dir "$ADAPTER" \
    --base "$BASE" \
    --stage "$STAGE" \
    --hyper '{"lr":1e-5,"epochs":1,"r":16,"qlora":true,"mix":"30:70","bs":4,"ga":2,"max_len":1536}' \
    --eval-json reports/${TAG}_combined.json \
    --notes "Stage 1: R1-Distill-14B + 4528 TH SFT + 6000 EN mix, single GPU after ECC error on GPU 1"

echo "=== [$(date)] DONE ==="
