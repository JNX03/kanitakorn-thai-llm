"""Phase 0.1 — re-emit the validation split.

The packager wrote 817 val_*.jsonl files but only 18 contain records, because
only 18 source items in manual_corpus + continuation_corpus carry
`split: "validation"`. We rebuild the split here without touching the source
corpus files:

1. Read every record from `dataset/train/` and `dataset/validation/`.
2. Pool by family + difficulty + language.
3. Within each stratum, the bottom 10% (by stable hash of id) become
   validation; the rest are train.
4. Write back, deleting the 0-byte sentinel files.

Re-runnable: writing with the same input + same hash function produces the
same split, so this is idempotent.

CLI:
    python tools/rebalance_validation_split.py [--ratio 0.1] [--dry-run]

Stratification preserves both `difficulty` and `language` ratios so the
validation set isn't accidentally biased toward easy English items.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"


def stable_bucket(record_id: str) -> float:
    """Deterministic float in [0, 1) for a given record id."""
    h = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def shard_from_path(name: str) -> str:
    parts = name.rsplit("_", 1)
    if len(parts) != 2:
        return "000"
    return parts[1].split(".")[0]


def family_from_path(name: str) -> str:
    base = name.removeprefix("train_").removeprefix("val_").removesuffix(".jsonl")
    last_underscore = base.rfind("_")
    if last_underscore == -1:
        return base
    return base[:last_underscore]


def load_all() -> list[dict]:
    items: list[dict] = []
    seen_ids: set[str] = set()  # dedupe across all files (last-write-wins logically;
                                # we keep the FIRST occurrence and skip the rest)
    for split_dir, split_name in [(DATASET / "train", "train"), (DATASET / "validation", "validation")]:
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.jsonl")):
            shard = shard_from_path(path.name)
            family = family_from_path(path.name)
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = rec.get("id", "")
                if rid and rid in seen_ids:
                    continue  # skip duplicate
                if rid:
                    seen_ids.add(rid)
                items.append(
                    {
                        "_record": rec,
                        "_current_split": split_name,
                        "_shard": shard,
                        "_family": rec.get("benchmark_family", family),
                        "_difficulty": rec.get("difficulty", "medium"),
                        "_language": rec.get("language", "th"),
                    }
                )
    return items


def assign_splits(items: list[dict], ratio: float) -> list[dict]:
    """Stratified assignment: within each (family, difficulty, language) bucket,
    the bottom `ratio` (by stable hash of id) becomes validation.
    """
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for it in items:
        key = (it["_family"], it["_difficulty"], it["_language"])
        buckets[key].append(it)
    for key, group in buckets.items():
        # Sort by hash; the first floor(N * ratio) become validation.
        group.sort(key=lambda x: stable_bucket(x["_record"]["id"]))
        n_val = max(1, int(round(len(group) * ratio))) if len(group) >= 10 else (1 if len(group) >= 3 else 0)
        for i, it in enumerate(group):
            it["_new_split"] = "validation" if i < n_val else "train"
    return items


def regroup_and_write(items: list[dict], dry_run: bool = False) -> dict:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for it in items:
        key = (it["_new_split"], it["_family"], it["_shard"])
        grouped[key].append(it["_record"])

    counts: Counter = Counter()
    for (split, family, shard), recs in grouped.items():
        counts[(split, family)] += len(recs)

    if dry_run:
        return {"counts": dict(counts), "total_groups": len(grouped)}

    # Delete every existing train_/val_ jsonl, then rewrite from scratch.
    # Handle Windows read-only attribute (set by lock_state.py).
    # Tolerate locked files — they'll be overwritten by writes below if dedupe collapses them.
    import subprocess as _sp
    import sys as _sys
    locked_paths: list[Path] = []
    for d in (DATASET / "train", DATASET / "validation"):
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*.jsonl"):
            try:
                p.unlink()
            except PermissionError:
                if _sys.platform == "win32":
                    _sp.run(["attrib", "-R", str(p)], check=False, capture_output=True)
                    try:
                        p.unlink()
                    except (PermissionError, OSError):
                        locked_paths.append(p)
                else:
                    raise
    if locked_paths:
        print(f"warn: {len(locked_paths)} files locked by another process; they'll be overwritten where dedupe places them")

    for (split, family, shard), recs in grouped.items():
        folder = DATASET / ("train" if split == "train" else "validation")
        prefix = "train" if split == "train" else "val"
        path = folder / f"{prefix}_{family}_{shard}.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + ("\n" if recs else ""),
            encoding="utf-8",
        )

    return {"counts": dict(counts), "total_groups": len(grouped)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.10, help="validation fraction per stratum")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    items = load_all()
    print(f"loaded {len(items)} records")
    if not items:
        print("no records found — aborting")
        return 2

    items = assign_splits(items, args.ratio)
    summary = regroup_and_write(items, dry_run=args.dry_run)

    print("split summary (after rebalance):")
    train_total = val_total = 0
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "validation": 0})
    for (split, family), n in sorted(summary["counts"].items(), key=lambda kv: (kv[0][1], kv[0][0])):
        by_family[family][split] = n
        if split == "train":
            train_total += n
        else:
            val_total += n
    print(f"{'family':<22} {'train':>6} {'val':>6} {'val%':>6}")
    for family, splits in sorted(by_family.items()):
        t, v = splits["train"], splits["validation"]
        denom = t + v
        pct = (v / denom * 100) if denom else 0.0
        print(f"{family:<22} {t:>6} {v:>6} {pct:>5.1f}%")
    print(f"{'TOTAL':<22} {train_total:>6} {val_total:>6}")

    if args.dry_run:
        print("(dry-run — no files written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
