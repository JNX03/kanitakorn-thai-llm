#!/usr/bin/env bash
# Push this repo to GitHub. Code-only-friendly. The 1.97GB benchmark_inputs
# file is .gitignored — it gets regenerated on the A100 via bootstrap_a100.sh.
#
# Prereq: `gh auth login` (or have a GitHub PAT set in GITHUB_TOKEN)
#
# Usage:
#   bash push_to_github.sh YOUR_GITHUB_USERNAME/repo-name [--public]

set -e

REPO="${1:?Usage: bash push_to_github.sh OWNER/repo [--public]}"
VIS="--private"
if [ "$2" = "--public" ]; then VIS="--public"; fi

# Sanity
if ! command -v gh > /dev/null 2>&1; then
    echo "ERROR: gh CLI not installed. https://cli.github.com"
    exit 1
fi
gh auth status > /dev/null 2>&1 || { echo "ERROR: gh auth login first"; exit 1; }

echo "=== Creating GitHub repo: $REPO ($VIS) ==="
gh repo create "$REPO" $VIS --source=. --remote=origin --push --description "Kanitakorn Thai LLM SFT pipeline + dataset" 2>&1 || {
    echo "  repo may already exist; falling back to add-remote-and-push"
    if [ ! -d .git ]; then
        git init -b main
    fi
    git remote remove origin 2>/dev/null || true
    git remote add origin "https://github.com/$REPO.git"
    git add -A
    git commit -m "Initial commit: Kanitakorn Thai SFT v1 pipeline" 2>/dev/null || true
    git branch -M main
    git push -u origin main --force
}

echo ""
echo "=== DONE ==="
echo "Repo: https://github.com/$REPO"
echo ""
echo "On the A100:"
echo "  git clone https://github.com/$REPO kanitakorn"
echo "  cd kanitakorn"
echo "  bash bootstrap_a100.sh"
