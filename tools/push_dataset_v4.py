"""Push distilled-augmented dataset to HF as kanitakorn-th-sft-v4-distill.

v3 was 8645 records (Thai SFT with CoT). v4 adds the R1-distilled traces
for math preservation + DeepSeek-format chat-template aware records.

Usage:
    HF_TOKEN=... python3 tools/push_dataset_v4.py
"""
import json, os, hashlib
from pathlib import Path
from huggingface_hub import HfApi
from datasets import Dataset

PROJECT = Path(__file__).resolve().parents[1]

def collect():
    """Combine sft_ready families + distill_r1_* traces."""
    out = []
    # Existing SFT-ready
    for fp in (PROJECT / "dataset/sft_ready").glob("*_train.jsonl"):
        for line in fp.open(encoding="utf-8"):
            if not line.strip(): continue
            r = json.loads(line)
            r["origin"] = "sft_ready_v3"
            out.append(r)
    # Distillation files
    distill_files = list((PROJECT / "dataset/train").glob("train_distill_r1_*.jsonl"))
    for fp in distill_files:
        family = "aime_th" if "aime" in fp.name else ("math500_th" if "math500" in fp.name else "openthaieval")
        for line in fp.open(encoding="utf-8"):
            if not line.strip(): continue
            r = json.loads(line)
            r["family"] = family
            r["origin"] = "distill_r1_v1"
            # Add empty `text` field; consumer rebuilds with their tokenizer
            if "text" not in r and "messages" in r:
                r["text"] = ""  # placeholder — must re-render
            out.append(r)
    return out

def main():
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    recs = collect()
    print(f"[collect] total {len(recs)} records")
    # Stats per family
    from collections import Counter
    counts = Counter(r.get("family","?") for r in recs)
    for f, c in counts.most_common():
        print(f"  {f}: {c}")

    # Compute SHA256
    h = hashlib.sha256()
    for r in recs[:10000]:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode())
    print(f"[sha256] {h.hexdigest()[:16]}...")

    # Save locally first
    out_fp = PROJECT / "dataset/v4_combined.jsonl"
    with out_fp.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[local] {out_fp}")

    # Push as HF dataset
    ds = Dataset.from_list(recs)
    repo = "Jnx03/kanitakorn-th-sft-v4-distill"
    api.create_repo(repo, private=False, repo_type="dataset", exist_ok=True)
    ds.push_to_hub(repo, token=os.environ.get("HF_TOKEN"))
    print(f"[hf] pushed {repo}")

if __name__ == "__main__":
    main()
