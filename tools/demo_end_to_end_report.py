"""End-to-end demonstration that benchmark_eval emits a real comparison report.

We cannot run a trained 35B model from this session, but we CAN prove the
scoring + reporting pipeline produces a real markdown report against the real
Typhoon-2 / OpenThaiGPT-1.5 baselines wired into benchmark_eval.

Method: extract ~30 records per family from `dataset/validation/`, treat each
record's `final_answer` as if it were a perfect prediction, then score them
through the real judges and write the comparison report.

Output: `dataset/reports/demo_benchmark_eval.md` — shows the layout, baselines,
and deltas the user will see when they plug in real model predictions.

Limitations of this demo:
    - "Predictions" are the gold answers, so scores will be near-1.0 — this
      validates the WIRING (judge → report → delta), not model quality.
    - HotpotQA path needs prediction_sources; we synthesize from `sources`.
    - LiveCodeBench needs runnable code; we skip that family in the demo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_eval import score_predictions, write_report, TYPHOON_2_BASELINES  # noqa: E402


FAMILY_TO_BENCHMARK = {
    "aime_th": "aime24",   # treat AIME-TH gold as if it were AIME24 predictions
    "math500_th": "math500",
    "openthaieval": "openthaieval",   # MCQ items only
    "ifeval_ifbench": "ifeval",
    "hotpotqa_agentic": "hotpotqa",
    # Excluded from the offline demo because they require LLM tiebreaker:
    #   - mt_bench (pointwise grading is always LLM-driven)
    #   - livecodebench_th (needs runnable code, separate sandbox)
}


def _is_deterministic_openthaieval(answer: str) -> bool:
    """MCQ letter/digit OR XNLI 3-class — both bypass the LLM tiebreaker."""
    import re as _re
    a = answer.strip().lower()
    if _re.fullmatch(r"\(?[1-9]\)?", a) or _re.fullmatch(r"\(?[ก-ฮ]\)?", a):
        return True
    if a in {"entailment", "contradiction", "neutral"}:
        return True
    return False


def main() -> int:
    demo_preds: list[dict] = []
    for shard in sorted((ROOT / "dataset" / "validation").glob("val_*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            family = rec["benchmark_family"]
            target = FAMILY_TO_BENCHMARK.get(family)
            if not target:
                continue
            user_prompt = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
            gold = rec["final_answer"]
            base = {
                "family": target,
                "id": rec["id"],
                "prompt": user_prompt,
                "prediction": gold,  # gold-as-prediction: validates the scoring path
                "gold": gold,
            }
            if target == "openthaieval":
                # Demo handles MCQ + XNLI items (both deterministic).
                # Free-form analytic items would need the LLM tiebreaker.
                if not _is_deterministic_openthaieval(gold):
                    continue
                base["subject"] = rec.get("task_type", "unknown")
            if target == "ifeval":
                # Synthesize a single matching constraint so the deterministic
                # IFEval path passes on the gold answer.
                base["constraints"] = [{"type": "regex", "pattern": ".*"}]
            if target == "hotpotqa":
                base["prediction_text"] = gold
                base["gold_answer"] = gold
                base["prediction_sources"] = [s.get("url", "") for s in rec.get("sources", [])]
                base["gold_supporting_facts"] = base["prediction_sources"]
            demo_preds.append(base)

    demo_jsonl = ROOT / "dataset" / "reports" / "demo_predictions.jsonl"
    demo_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with demo_jsonl.open("w", encoding="utf-8") as fh:
        for p in demo_preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(demo_preds)} demo predictions → {demo_jsonl}")

    results = score_predictions(demo_jsonl)
    out_md = ROOT / "dataset" / "reports" / "demo_benchmark_eval.md"
    write_report(results, TYPHOON_2_BASELINES, "demo-gold-as-prediction", out_md)
    print(f"wrote report → {out_md}")

    # Per-family summary line so the smoke test is visible from the console.
    summary = []
    for fam, r in sorted(results.items()):
        if "accuracy" in r:
            summary.append(f"{fam}={r['accuracy']:.3f}")
        elif "strict_acc" in r:
            summary.append(f"{fam}=strict:{r['strict_acc']:.3f}/loose:{r['loose_acc']:.3f}")
        elif "overall" in r:
            summary.append(f"{fam}={r['overall']:.3f}")
        elif "answer_em" in r:
            summary.append(f"{fam}=em:{r['answer_em']:.3f}/f1:{r['sf_f1']:.3f}")
    print(" | ".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
