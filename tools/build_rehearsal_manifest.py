"""Build SFT manifest with rehearsal data to prevent catastrophic forgetting.

Mix Thai SFT (target: lift Thai) + math/code rehearsal (target: preserve math).

Ratio recommendation per research:
- 40% Thai exam targeted (push ThaiExam +)
- 20% Thai general (mt_bench, ifeval, openthaieval style)
- 20% Math rehearsal (math500_th, aime_th from our existing distill)
- 10% Code rehearsal (livecodebench_th)
- 10% Multi-hop QA (hotpotqa)
"""
import json
from pathlib import Path

ROOT = Path("/root/kanitakorn") if Path("/root/kanitakorn").exists() else Path(__file__).resolve().parents[1]

WEIGHTS = {
    "thaiexam_targeted": 0.40,
    "openthaieval": 0.15,
    "teacher_loop_th": 0.10,
    "math500_th": 0.10,
    "aime_th": 0.10,
    "ifeval_ifbench": 0.05,
    "livecodebench_th": 0.05,
    "mt_bench": 0.03,
    "hotpotqa_agentic": 0.02,
}

FILES = {
    "thaiexam_targeted": "dataset/sft_ready_thai_v5/thaiexam_targeted_train.jsonl",
    "openthaieval": "dataset/sft_ready_qwen3/openthaieval_train.jsonl",
    "teacher_loop_th": "dataset/sft_ready_qwen3/teacher_loop_th_train.jsonl",
    "math500_th": "dataset/sft_ready_qwen3/math500_th_train.jsonl",
    "aime_th": "dataset/sft_ready_qwen3/aime_th_train.jsonl",
    "ifeval_ifbench": "dataset/sft_ready_qwen3/ifeval_ifbench_train.jsonl",
    "livecodebench_th": "dataset/sft_ready_qwen3/livecodebench_th_train.jsonl",
    "mt_bench": "dataset/sft_ready_qwen3/mt_bench_train.jsonl",
    "hotpotqa_agentic": "dataset/sft_ready_qwen3/hotpotqa_agentic_train.jsonl",
}

def main():
    counts = {}
    for k, rel_path in FILES.items():
        p = ROOT / rel_path
        if p.exists():
            n = sum(1 for _ in p.open(encoding="utf-8"))
            counts[k] = n
            print(f"{k}: {n}")
        else:
            print(f"{k}: MISSING ({p})")
            counts[k] = 0

    manifest = {
        "strategy": "weighted_rehearsal",
        "train_counts": counts,
        "weights": WEIGHTS,
        "sft_files": {k: {"train": v, "validation": None} for k, v in FILES.items()},
    }
    out = ROOT / "dataset/sft_ready_thai_v5/manifest_rehearsal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    print(f"Total records: {sum(counts.values())}")

if __name__ == "__main__":
    main()
