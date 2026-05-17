"""Build 3 backup dataset variants for the A100 SFT runs.

Hedge against the "main dataset doesn't beat baseline" outcome by producing
three differently-weighted variants the user can train on if needed:

    1. conservative  — only LLM-verified records (highest quality, smallest)
                       useful if main dataset overfits or hallucinates
    2. tutor         — heavy weight on Thai exam-prep + teacher-loop
                       useful if the goal is "student tutor" specifically
    3. aggressive    — includes ALL records, heavier weight on teacher_loop
                       useful if novelty (teacher-loop) is the headline result

All variants share the same SFT-text format (Qwen chat template, hash-bucketed
few-shot). They differ only in:
    - which records are included
    - per-family weight in dataset/sft_ready/manifest_<variant>.json

Run:
    python tools/build_variants.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


VARIANTS = {
    "conservative": {
        "description": "Only LLM-verified records. Highest factual quality, smallest size.",
        "include": [
            # OK or WARN from LLM-fact-audit (FAIL records excluded)
            "lambda r: r['benchmark_family'] not in ('hotpotqa_agentic', 'teacher_loop_th')",
            "lambda r: r.get('quality_scores', {}).get('correctness', 0) >= 0.95",
        ],
        # Conservative weights — uniform across families that survived.
        "strategy": "uniform",
    },
    "tutor": {
        "description": "Heavy weight on Thai exam-prep + teacher-loop. For 'help students with exams' mission.",
        "include": [],  # all records
        "strategy": "tutor_weighted",
        "weights": {
            "openthaieval": 2.0,
            "teacher_loop_th": 2.0,
            "mt_bench": 1.5,
            "ifeval_ifbench": 1.5,
            "aime_th": 0.8,
            "math500_th": 1.0,
            "livecodebench_th": 0.5,
            "hotpotqa_agentic": 1.0,
        },
    },
    "aggressive": {
        "description": "All records, heavy teacher_loop weight to test the novel method.",
        "include": [],  # all
        "strategy": "aggressive_weighted",
        "weights": {
            "teacher_loop_th": 3.0,  # 3× the sqrt weight
            "openthaieval": 1.5,
            "mt_bench": 1.5,
            "ifeval_ifbench": 1.2,
            "aime_th": 1.0,
            "math500_th": 1.0,
            "livecodebench_th": 1.0,
            "hotpotqa_agentic": 1.5,
        },
    },
}


def load_records() -> list[dict]:
    """Load every train + val record into a single list with split tag."""
    records = []
    seen = set()
    for split in ("train", "validation"):
        for p in sorted((ROOT / "dataset" / split).glob("*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("id") in seen:
                    continue
                seen.add(r.get("id"))
                r["_split"] = split
                records.append(r)
    return records


def load_fact_audit_failures() -> set[str]:
    """IDs that LLM-fact-audit marked FAIL."""
    fail_ids = set()
    for p in (
        ROOT / "dataset" / "reports" / "fact_verification_llm.jsonl",
        ROOT / "dataset" / "reports" / "exam_fact_verification_llm.jsonl",
    ):
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("verdict") == "FAIL":
                fail_ids.add(r.get("id"))
    return fail_ids


def build_variant(name: str, records: list[dict], fact_fails: set[str]) -> dict:
    cfg = VARIANTS[name]
    # Apply include predicates.
    keep = []
    for r in records:
        if r.get("id") in fact_fails:
            continue
        ok = True
        for pred_str in cfg.get("include", []):
            pred = eval(pred_str)  # noqa: S307 — trusted source
            if not pred(r):
                ok = False
                break
        if ok:
            keep.append(r)

    by_family = Counter(r["benchmark_family"] for r in keep)
    train_counts = Counter(r["benchmark_family"] for r in keep if r["_split"] == "train")

    # Compute weights.
    if cfg["strategy"] == "uniform":
        raw = {f: 1.0 for f in train_counts}
    elif cfg["strategy"] in ("tutor_weighted", "aggressive_weighted"):
        base = {f: math.sqrt(n) for f, n in train_counts.items() if n > 0}
        raw = {f: base[f] * cfg["weights"].get(f, 1.0) for f in base}
    else:  # sqrt-balanced fallback
        raw = {f: math.sqrt(n) for f, n in train_counts.items() if n > 0}
    total_w = sum(raw.values())
    weights = {f: round(v / total_w, 6) for f, v in raw.items()}

    # Write variant SFT files (one per family per split).
    out_dir = ROOT / "dataset" / "sft_ready_variants" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    by_family_split: dict[tuple[str, str], list[dict]] = {}
    for r in keep:
        by_family_split.setdefault((r["benchmark_family"], r["_split"]), []).append(r)
    for (family, split), recs in by_family_split.items():
        path = out_dir / f"{family}_{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in recs:
                rr = {k: v for k, v in r.items() if k != "_split"}
                fh.write(json.dumps(rr, ensure_ascii=False) + "\n")

    # Write manifest.
    manifest = {
        "variant": name,
        "description": cfg["description"],
        "strategy": cfg["strategy"],
        "fact_audit_excluded": sorted(fact_fails),
        "train_counts": dict(train_counts),
        "weights": weights,
        "sft_files": {
            family: {
                "train": f"sft_ready_variants/{name}/{family}_train.jsonl",
                "validation": f"sft_ready_variants/{name}/{family}_validation.jsonl"
                if (out_dir / f"{family}_validation.jsonl").exists()
                else None,
            }
            for family in train_counts
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "variant": name,
        "total_kept": len(keep),
        "by_family": dict(by_family),
        "weights_sum": round(sum(weights.values()), 6),
    }


def main() -> int:
    records = load_records()
    fact_fails = load_fact_audit_failures()
    print(f"Loaded {len(records)} unique records; {len(fact_fails)} excluded due to LLM-fact-FAIL")

    summary = {}
    for name in VARIANTS:
        s = build_variant(name, records, fact_fails)
        summary[name] = s
        print(f"\n{name:<12}: kept {s['total_kept']}")
        for fam, n in sorted(s["by_family"].items()):
            print(f"   {fam:<22} {n:>6}")
    (ROOT / "dataset" / "reports" / "variants_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote dataset/reports/variants_summary.json")
    print(f"variants live under dataset/sft_ready_variants/{{conservative,tutor,aggressive}}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
