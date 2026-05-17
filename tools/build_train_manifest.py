"""Phase 3.2 — weighted-sampling manifest for the SFT trainer.

Reads dataset/sft_ready/*.jsonl and writes dataset/sft_ready/manifest.json
with a per-family weight that prevents the bigger families (aime_th 1,287)
from drowning the smaller ones (mt_bench 87).

Weighting rule: balanced by sqrt(N). This is a softer balance than 1/N
(which over-weights tiny families) and harder than N (no balance).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SFT_DIR = ROOT / "dataset" / "sft_ready"


def count_records(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(SFT_DIR / "manifest.json"))
    parser.add_argument("--strategy", choices=["sqrt", "linear", "uniform"], default="sqrt")
    args = parser.parse_args()

    counts: Counter = Counter()
    for path in sorted(SFT_DIR.glob("*_train.jsonl")):
        family = path.name[: -len("_train.jsonl")]
        counts[family] = count_records(path)

    if not counts:
        print(f"no sft_ready files under {SFT_DIR}; run few_shot_collator.py first")
        return 1

    if args.strategy == "sqrt":
        raw = {f: math.sqrt(n) for f, n in counts.items() if n > 0}
    elif args.strategy == "linear":
        raw = {f: float(n) for f, n in counts.items() if n > 0}
    else:  # uniform
        raw = {f: 1.0 for f, n in counts.items() if n > 0}
    total = sum(raw.values())
    weights = {f: round(v / total, 6) for f, v in raw.items()}

    manifest = {
        "strategy": args.strategy,
        "train_counts": dict(counts),
        "weights": weights,
        "sft_files": {
            family: {
                "train": f"sft_ready/{family}_train.jsonl",
                "validation": f"sft_ready/{family}_validation.jsonl"
                if (SFT_DIR / f"{family}_validation.jsonl").exists()
                else None,
            }
            for family in counts
        },
    }
    Path(args.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"weight sum = {sum(weights.values()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
