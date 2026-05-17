#!/usr/bin/env bash
# Push this repo to a HuggingFace Hub dataset repo. Recommended for fast
# A100 transfer because:
#   - HF natively supports large files (no LFS setup needed)
#   - A100 rentals usually have huggingface-cli pre-installed
#   - `git clone https://huggingface.co/datasets/USER/REPO` is fast
#   - You can publish trained model to a separate model repo via hf_publish.py
#
# Prereq: `huggingface-cli login` (gets token from huggingface.co/settings/tokens)
#
# Usage:
#   bash push_to_hf.sh YOUR_USERNAME/kanitakorn-th-sft-pipeline

set -e

REPO="${1:?Usage: bash push_to_hf.sh YOUR_USERNAME/repo-name}"

# Use the new `hf` CLI (huggingface-cli is deprecated).
HF_CLI="hf"
command -v $HF_CLI > /dev/null 2>&1 || HF_CLI="huggingface-cli"

# Sanity: must be logged in
if ! $HF_CLI auth whoami > /dev/null 2>&1; then
    echo "ERROR: not logged in to HF. Run: $HF_CLI auth login"
    exit 1
fi

echo "=== Creating HF dataset repo: $REPO ==="
$HF_CLI repo create "$REPO" --repo-type dataset --exist-ok

echo ""
echo "=== Initializing git remote ==="
if [ ! -d .git ]; then
    git init -b main
    git add -A
    git commit -m "Initial commit: Kanitakorn Thai SFT v1 pipeline + dataset"
fi
git remote remove hf 2>/dev/null || true
git remote add hf "https://huggingface.co/datasets/$REPO"

echo ""
echo "=== Configuring LFS for large jsonl files ==="
git lfs install 2>/dev/null || pip install -q hf-transfer
$HF_CLI lfs-enable-largefiles . 2>/dev/null || true

echo ""
echo "=== Pushing to HF ==="
git push hf main --force

echo ""
echo "=== DONE ==="
echo "Repo: https://huggingface.co/datasets/$REPO"
echo ""
echo "On the A100:"
echo "  git clone https://huggingface.co/datasets/$REPO kanitakorn"
echo "  cd kanitakorn"
echo "  bash bootstrap_a100.sh"
