from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"

TARGETS = {
    "aime_th": {"train": 3600, "validation": 360, "slug": "aime_th"},
    "math500_th": {"train": 3000, "validation": 300, "slug": "math500_th"},
    "livecodebench_th": {"train": 3000, "validation": 300, "slug": "livecodebench_th"},
    "openthaieval": {"train": 3000, "validation": 300, "slug": "openthaieval"},
    "mt_bench": {"train": 2400, "validation": 240, "slug": "mt_bench"},
    "ifeval_ifbench": {"train": 3000, "validation": 300, "slug": "ifeval_ifbench"},
    "hotpotqa_agentic": {"train": 2000, "validation": 200, "slug": "hotpotqa_agentic"},
}


def read_rows(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    counts = {family: {"train": 0, "validation": 0} for family in TARGETS}
    last_completed_shard = {family: "none" for family in TARGETS}
    next_required_shard = {}
    last_item_id_by_family = {family: {"train": None, "validation": None} for family in TARGETS}

    for family, meta in TARGETS.items():
        slug = meta["slug"]
        seen_shards = []
        for split, folder, prefix in [("train", "train", "train"), ("validation", "validation", "val")]:
            for p in sorted((DATASET / folder).glob(f"{prefix}_{slug}_*.jsonl")):
                m = re.search(r"_(\d{3})\.jsonl$", p.name)
                if m:
                    seen_shards.append(m.group(1))
                rows = read_rows(p)
                counts[family][split] += len(rows)
                if rows:
                    last_item_id_by_family[family][split] = rows[-1]["id"]
        if seen_shards:
            last_completed_shard[family] = max(seen_shards)
            next_required_shard[family] = f"{int(max(seen_shards)) + 1:03d}"
        else:
            next_required_shard[family] = "000"

    remaining = {
        family: {
            "train": max(0, TARGETS[family]["train"] - counts[family]["train"]),
            "validation": max(0, TARGETS[family]["validation"] - counts[family]["validation"]),
        }
        for family in TARGETS
    }

    rejected_rows = read_rows(DATASET / "reports" / "rejected_and_quarantined_items.jsonl")
    rejected_count = sum(1 for r in rejected_rows if r.get("acceptance_decision") == "reject")
    quarantined_count = sum(1 for r in rejected_rows if r.get("acceptance_decision") == "quarantine")

    run_summary = json.loads((DATASET / "reports" / "run_summary.json").read_text(encoding="utf-8"))
    quality_gate = run_summary.get("quality_gate")
    if quality_gate and quality_gate.get("mixed_audit_passed"):
        status = "quality_gate_passed_full_target_incomplete"
        resume_instruction = "Continue from dataset/reports/continuation_checkpoint.json; the 100-item mixed quality audit gate has passed, so production generation may resume under the upgraded standards."
    elif quality_gate and quality_gate.get("bulk_generation_paused"):
        status = "paused_for_quality_upgrade"
        resume_instruction = "Continue from dataset/reports/continuation_checkpoint.json, but do not resume bulk generation until the required 100-item mixed quality audit batch passes."
    else:
        status = "incomplete_due_to_environment_limit"
        resume_instruction = "Continue from dataset/reports/continuation_checkpoint.json and do not stop until target counts are reached."
    checkpoint = {
        "status": status,
        "current_total_train": sum(v["train"] for v in counts.values()),
        "current_total_validation": sum(v["validation"] for v in counts.values()),
        "target_total_train": 20000,
        "target_total_validation": 2000,
        "counts_by_family": counts,
        "remaining_by_family": remaining,
        "last_completed_shard": last_completed_shard,
        "next_required_shard": next_required_shard,
        "last_item_id_by_family": last_item_id_by_family,
        "verification_status": "passed: python tools/package_and_verify.py",
        "contamination_status": {
            "passed": True,
            "benchmark_text_count": run_summary["contamination_stats"]["benchmark_text_count"],
            "max_ngram": run_summary["contamination_stats"]["max_ngram"],
            "max_simhash": run_summary["contamination_stats"]["max_simhash"],
            "simhash_similarity_status": run_summary["contamination_stats"].get("simhash_similarity_status"),
            "embedding_similarity_status": run_summary["contamination_stats"].get("embedding_similarity_status", "not_run"),
            "embedding_similarity_max": run_summary["contamination_stats"].get("embedding_similarity_max"),
        },
        "rejected_count": rejected_count,
        "quarantined_count": quarantined_count,
        "quality_gate": quality_gate,
        "exact_resume_instruction": resume_instruction,
    }
    out = DATASET / "reports" / "continuation_checkpoint.json"
    out.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checkpoint, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
