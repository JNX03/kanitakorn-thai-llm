"""Embedding-similarity contamination check.

Loads benchmark test sets (the actual public eval sets we'll use to score),
embeds them with paraphrase-multilingual-MiniLM, then compares each train
record. Anything with cosine ≥ 0.92 to any test record is quarantined.

This protects against accidental data leakage that would inflate our scores.

Usage:
    python3 tools/contamination_check.py \\
        --train-jsonl dataset/sft_ready/*.jsonl \\
        --out reports/contamination_$(date +%Y%m%d).json
"""
import argparse, json, glob, hashlib
from pathlib import Path

import torch
import numpy as np

from datasets import load_dataset

TEST_SETS = {
    "aime24": ("math-ai/aime24", "test", lambda r: r["problem"]),
    "math500": ("HuggingFaceH4/MATH-500", "test", lambda r: r["problem"]),
    "thaiexam": ("scb10x/thai_exam", "test", lambda r: r["question"]),
    # OpenThaiEval / IFEval-TH / etc. — add when test loaders confirmed
}

def load_test_texts():
    texts = []
    for tag, (repo, split, fn) in TEST_SETS.items():
        try:
            ds = load_dataset(repo, split=split)
            for r in ds:
                t = fn(r)
                if t: texts.append((tag, t))
            print(f"[load] {tag}: {len(ds)} test items from {repo}")
        except Exception as e:
            print(f"[skip] {tag} ({repo}): {e}")
    return texts

def extract_user_text(rec):
    """Pull user-message text from a training record."""
    if "text" in rec:
        # Try to find user message in chat template
        t = rec["text"]
        if "<|im_start|>user" in t:
            return t.split("<|im_start|>user")[-1].split("<|im_end|>")[0].strip()
        return t[:2000]
    if "messages" in rec:
        for m in rec["messages"]:
            if m.get("role") == "user":
                return m.get("content", "")
    return rec.get("prompt") or rec.get("question") or ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", nargs="+", default=["dataset/sft_ready/*_train.jsonl"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.92)
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    print(f"[model] loading {args.model}")
    enc = SentenceTransformer(args.model, device="cuda" if torch.cuda.is_available() else "cpu")

    test_texts = load_test_texts()
    test_vecs = enc.encode([t for _, t in test_texts], batch_size=64,
                            convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=True)
    print(f"[test] embedded {len(test_texts)} test items")

    quarantined = []
    n_total = 0
    for pat in args.train_jsonl:
        for fp in glob.glob(pat):
            print(f"[train] processing {fp}")
            recs = [json.loads(l) for l in Path(fp).open(encoding="utf-8") if l.strip()]
            texts = [extract_user_text(r) for r in recs]
            vecs = enc.encode(texts, batch_size=64, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
            sims = vecs @ test_vecs.T  # [N_train, N_test]
            max_idx = sims.argmax(axis=1)
            max_sim = sims.max(axis=1)
            n_total += len(recs)
            for i, s in enumerate(max_sim):
                if s >= args.threshold:
                    tag, ttext = test_texts[max_idx[i]]
                    quarantined.append({
                        "source_file": fp,
                        "rec_id": recs[i].get("id"),
                        "max_sim": float(s),
                        "matched_bench": tag,
                        "matched_test_text": ttext[:200],
                        "train_text": texts[i][:200],
                    })

    pct = 100 * len(quarantined) / max(n_total, 1)
    print(f"\n[contamination] {len(quarantined)} / {n_total} records ({pct:.2f}%) above threshold {args.threshold}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "threshold": args.threshold,
        "n_train": n_total,
        "n_quarantined": len(quarantined),
        "quarantined": quarantined,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {args.out}")

if __name__ == "__main__":
    main()
