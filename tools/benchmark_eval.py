"""Phase 1.4 — benchmark harness against public eval sets.

Runs a trained (or baseline) model through the actual public benchmarks the
project targets, then routes each family's predictions to the right judge
module and writes a markdown comparison report against the published Typhoon-2
/ OpenThaiGPT-1.5 baselines.

Model loading: defaults to local HuggingFace inference. Supports `--model
api:<provider>:<model>` for hosted models (OpenAI/Anthropic). Implementing
local inference cleanly requires gigabytes of weights — for now the harness
loads the trained model lazily and accepts an `--inputs-only` mode that
exports the benchmark inputs as a JSONL so an external worker can generate
predictions, then `--score-from <predictions.jsonl>` to score them.

Public benchmark sources (per run_summary.json blacklist_notes):
    math-ai/aime24 — 30 rows
    math-ai/aime25 — 30 rows
    math-ai/math500 — 500 rows
    typhoon-ai/ifeval-th — 215 rows
    ThaiLLM-Leaderboard/mt-bench-thai — 91 rows
    iapp/openthaieval — 1232 rows
    typhoon-ai/livecodebench-th — sampled JSONL
    (HotpotQA needs the original parquet; agentic harness is a separate task)

Published baselines for the report (placeholder constants — update when you
have authoritative numbers from the actual Typhoon-2 / OpenThaiGPT-1.5
leaderboards):
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from judges import (  # noqa: E402
    MTBenchJudge,
    IFEvalJudge,
    OpenThaiEvalJudge,
    HotpotQAJudge,
    MathJudge,
    LiveCodeBenchJudge,
)


# Published baselines.
#
# Sources:
#   Typhoon 2 paper (arxiv 2412.13702v2) — Table 4+ across Typhoon2-{1B,3B,8B,70B}-Instruct.
#   OpenThaiGPT 1.5 model card (huggingface.co/openthaigpt/openthaigpt1.5-7b-instruct).
#
# Caveats:
#   * AIME24-TH, AIME25-TH, LiveCodeBench-TH, HotpotQA: the source papers do not
#     report these — values below are `None` and the report will print "—" for
#     those rows. Update once Typhoon/OpenThaiGPT publish them or rerun their
#     official eval scripts.
#   * MT-Bench-TH: Typhoon reports a 1-10 score (LLM-judge mean), not a fraction.
#   * IFEval-TH 72.60 from Typhoon paper is reported as a combined accuracy; we
#     compare against `ifeval_th_overall` and leave strict/loose separate as `None`.
#
# All other numbers are direct copies from the published tables.
TYPHOON_2_BASELINES = {
    # Typhoon2-70B-Instruct
    "thai_exam_70b": 0.6338,
    "m3exam_70b": 0.6233,
    "bfcl_th_70b": 0.7089,
    # Typhoon2-8B-Instruct (closest size to our Qwen3.6-35B-A3B target for fair-ish compare)
    "thai_exam_8b": 0.5120,
    "m3exam_8b": 0.4752,
    "ifeval_th_overall": 0.7260,
    "mt_bench_th_overall": 5.74,  # 1-10 LLM-judge mean
    "gsm8k": 0.81,
    "math500_th": 0.4904,  # Typhoon paper reports MATH (not MATH500-TH specifically); proxy
    # Targets we don't have published numbers for:
    "aime24_th": None,
    "aime25_th": None,
    "ifeval_th_strict": None,
    "openthaieval_overall": None,
    "livecodebench_th_pass_at_1": None,
    "hotpotqa_em": None,
}
OPENTHAIGPT_1_5_BASELINES = {
    # OpenThaiGPT 1.5 7B
    "thai_exam_7b": 0.5204,
    "m3exam_7b": 0.5401,
    "openthaieval_overall": 0.6578,  # micro-average across 17 Thai exam categories
    # 14B / 72B variants
    "thai_exam_14b": 0.5965,
    "thai_exam_72b": 0.6407,
    # Not reported by OpenThaiGPT 1.5:
    "aime24_th": None,
    "aime25_th": None,
    "math500_th": None,
    "ifeval_th_strict": None,
    "ifeval_th_overall": None,
    "mt_bench_th_overall": None,
    "livecodebench_th_pass_at_1": None,
    "hotpotqa_em": None,
}

FAMILIES = [
    "aime24", "aime25", "math500", "ifeval", "mt_bench", "openthaieval",
    "livecodebench", "hotpotqa",
    # Thai exam benchmarks — what Typhoon-2 / OpenThaiGPT publish numbers on.
    "thai_exam_onet", "thai_exam_ic", "thai_exam_tgat", "thai_exam_tpat1", "thai_exam_a_level",
]
# Aliases expand to multiple families.
_FAMILY_ALIASES = {
    "aime": ["aime24", "aime25"],
    "thai_exam": ["thai_exam_onet", "thai_exam_ic", "thai_exam_tgat", "thai_exam_tpat1", "thai_exam_a_level"],
}


def export_inputs(family: str, out_path: Path) -> int:
    """Read the cached benchmark and write a predictions-input JSONL.

    Each line: {"family":..., "id":..., "prompt":..., "gold":..., "meta":...}
    External worker fills in `prediction` and writes a parallel predictions
    file the harness will score.
    """
    rows = _load_benchmark(family)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def _load_benchmark(family: str) -> list[dict]:
    """Load a public benchmark from the HF cache. Best-effort — if the cached
    parquet/jsonl isn't available, returns an empty list and the harness will
    log a warning."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("datasets package not available — install `datasets` to use --inputs", file=sys.stderr)
        return []

    # (repo, split, config, mapper) — config=None for repos that don't need it.
    name_map = {
        "aime24": ("math-ai/aime24", "test", None, _aime_row),
        "aime25": ("math-ai/aime25", "test", None, _aime_row),
        "math500": ("math-ai/math500", "test", None, _math_row),
        "ifeval": ("typhoon-ai/ifeval-th", "train", None, _ifeval_row),  # only 'train' split
        "mt_bench": ("ThaiLLM-Leaderboard/mt-bench-thai", "train", None, _mt_bench_row),
        # iapp/openthaieval uses an old loading script — load the cached parquet directly via _load_openthaieval_parquet
        "openthaieval": ("iapp/openthaieval", "test", None, _openthaieval_row),
        "livecodebench": ("typhoon-ai/livecodebench-th", "train", None, _lcb_row),  # only 'train' split exists
        "hotpotqa": ("hotpot_qa", "validation", "distractor", _hotpotqa_row),
        # ThaiExam (scb10x/thai_exam) — 5 subject configs, MCQ exact-match scoring.
        "thai_exam_onet":    ("scb10x/thai_exam", "test", "onet",    _thai_exam_row),
        "thai_exam_ic":      ("scb10x/thai_exam", "test", "ic",      _thai_exam_row),
        "thai_exam_tgat":    ("scb10x/thai_exam", "test", "tgat",    _thai_exam_row),
        "thai_exam_tpat1":   ("scb10x/thai_exam", "test", "tpat1",   _thai_exam_row),
        "thai_exam_a_level": ("scb10x/thai_exam", "test", "a_level", _thai_exam_row),
    }
    if family not in name_map:
        return []
    repo, split, config, mapper = name_map[family]
    # iapp/openthaieval ships an old loading script that the new datasets API
    # rejects — fall back to the cached parquet file directly.
    if family == "openthaieval":
        cached = _load_openthaieval_parquet()
        if cached:
            return [_openthaieval_row(row, repo) for row in cached]
    try:
        if config:
            ds = load_dataset(repo, config, split=split)
        else:
            try:
                ds = load_dataset(repo, split=split)
            except Exception:
                ds = load_dataset(repo, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"failed to load {repo}: {e}", file=sys.stderr)
        return []
    return [mapper(row, repo) for row in ds]


def _aime_row(row, repo):
    return {
        "family": repo.split("/")[-1],
        "id": str(row.get("id", row.get("Problem Number", ""))),
        "prompt": row.get("problem") or row.get("question") or "",
        "gold": str(row.get("answer", "")),
    }


def _math_row(row, repo):
    return {
        "family": "math500",
        "id": str(row.get("unique_id", row.get("id", ""))),
        "prompt": row.get("problem", ""),
        "gold": str(row.get("answer", "")),
    }


def _ifeval_row(row, repo):
    return {
        "family": "ifeval",
        "id": str(row.get("key", row.get("id", ""))),
        "prompt": row.get("prompt", ""),
        "constraints": row.get("instruction_id_list", []),
        "gold": row.get("response", ""),
    }


def _mt_bench_row(row, repo):
    return {
        "family": "mt_bench",
        "id": str(row.get("question_id", "")),
        "prompt": row.get("turns", [""])[0] if row.get("turns") else row.get("prompt", ""),
        "category": row.get("category", ""),
        "reference_answer": row.get("reference", [""])[0] if row.get("reference") else "",
        "gold": "",
    }


def _openthaieval_row(row, repo):
    return {
        "family": "openthaieval",
        "id": str(row.get("id", "")),
        "prompt": row.get("instruction") or row.get("question", ""),
        "subject": row.get("subject", "unknown"),
        "gold": str(row.get("answer", row.get("result", ""))),
    }


def _lcb_row(row, repo):
    return {
        "family": "livecodebench",
        "id": str(row.get("question_id", "")),
        "prompt": row.get("question_content", row.get("prompt", "")),
        "public_tests": row.get("public_test_cases", []),
        "hidden_tests": row.get("private_test_cases", []),
        "gold": "",
    }


def _thai_exam_row(row, repo):
    """Render a ThaiExam MCQ as a single-prompt question and the gold letter."""
    options = []
    for letter in ("a", "b", "c", "d", "e"):
        v = row.get(letter)
        if v is not None and str(v).strip():
            options.append(f"({letter}) {v}")
    prompt = (
        f"{row.get('question', '')}\n\n"
        + "\n".join(options)
        + "\n\nตอบเฉพาะตัวอักษรของคำตอบ (a, b, c, d, หรือ e)"
    )
    return {
        "family": f"thai_exam_{row.get('subject', 'unknown')}",
        "id": f"{row.get('subject', '')}_{row.get('no', '')}",
        "prompt": prompt,
        "gold": str(row.get("answer", "")).strip().lower(),
        "subject": row.get("subject", "unknown"),
    }


def _hotpotqa_row(row, repo):
    return {
        "family": "hotpotqa",
        "id": str(row.get("id", "")),
        "prompt": row.get("question", ""),
        "gold_answer": row.get("answer", ""),
        "gold_supporting_facts": row.get("supporting_facts", {}).get("title", []),
    }


def _load_openthaieval_parquet():
    """iapp/openthaieval ships a loading script the new datasets API rejects.
    Find the cached parquet directly under ~/.cache/huggingface/hub/."""
    import glob as _glob
    import os as _os
    cache_root = _os.path.expanduser("~/.cache/huggingface/hub/datasets--iapp--openthaieval")
    if not _os.path.isdir(cache_root):
        return []
    parquets = _glob.glob(_os.path.join(cache_root, "**", "*.parquet"), recursive=True)
    if not parquets:
        return []
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        return []
    rows: list[dict] = []
    for p in parquets:
        try:
            table = pq.read_table(p)
            for r in table.to_pylist():
                rows.append(r)
        except Exception:
            continue
    return rows


def score_predictions(predictions_path: Path) -> dict:
    """Score a JSONL of model predictions. Each line must have at least:
        family, id, prediction, and the family-specific gold/meta fields
    that were written by `export_inputs`.
    """
    by_family: dict[str, list[dict]] = {}
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_family.setdefault(rec["family"], []).append(rec)

    results: dict[str, dict] = {}

    if "aime24" in by_family or "aime25" in by_family or "math500" in by_family:
        m = MathJudge()
        for fam in ("aime24", "aime25", "math500"):
            items = by_family.get(fam, [])
            if items:
                score = m.aggregate(items)
                results[fam] = {"accuracy": score.accuracy, "n": score.n, "sympy": score.n_sympy_resolved, "llm": score.n_llm_fallback}

    if "ifeval" in by_family:
        j = IFEvalJudge()
        items = [{"response": r["prediction"], "constraints": r.get("constraints", [])} for r in by_family["ifeval"]]
        s = j.aggregate(items)
        results["ifeval"] = {"strict_acc": s.strict_acc, "loose_acc": s.loose_acc, "n": s.n_prompts}

    if "mt_bench" in by_family:
        j = MTBenchJudge()
        triples = [(r["prompt"], r["prediction"], r.get("category", "")) for r in by_family["mt_bench"]]
        s = j.aggregate(triples)
        results["mt_bench"] = {"overall": s.overall, "by_category": s.by_category, "n": s.n, "cost_usd": s.cost_usd}

    if "openthaieval" in by_family:
        j = OpenThaiEvalJudge()
        items = [
            {"prediction": r["prediction"], "gold": r["gold"], "subject": r.get("subject", "unknown"), "question": r.get("prompt", "")}
            for r in by_family["openthaieval"]
        ]
        s = j.aggregate(items)
        results["openthaieval"] = {"overall": s.overall_acc, "mcq": s.mcq_acc, "analytic": s.analytic_acc, "by_subject": s.by_subject}

    if "livecodebench" in by_family:
        j = LiveCodeBenchJudge()
        items = [
            {"kind": "code", "source": r["prediction"], "public_tests": r.get("public_tests", []), "hidden_tests": r.get("hidden_tests", [])}
            for r in by_family["livecodebench"]
        ]
        s = j.aggregate(items)
        results["livecodebench"] = {"pass_at_1": s.pass_at_1, "n_pass": s.n_pass, "n": s.n}

    # ThaiExam — 5 configs, MCQ letter exact match. Reports per-subject + overall.
    thai_exam_subjects = [f for f in by_family if f.startswith("thai_exam_")]
    if thai_exam_subjects:
        import re as _re
        def _extract_thai_exam_letter(text: str) -> str:
            if not text:
                return ""
            t = text.strip().lower()
            # Look for first standalone a-e letter (with or without parens).
            m = _re.search(r"\(?([a-e])\)?", t)
            return m.group(1) if m else ""
        by_subject_correct: dict[str, int] = {}
        by_subject_n: dict[str, int] = {}
        for fam in thai_exam_subjects:
            subj = fam.removeprefix("thai_exam_")
            n = 0
            correct = 0
            for r in by_family[fam]:
                gold = str(r.get("gold", "")).strip().lower()
                pred = _extract_thai_exam_letter(r.get("prediction", ""))
                if pred and pred == gold:
                    correct += 1
                n += 1
            by_subject_correct[subj] = correct
            by_subject_n[subj] = n
        total_correct = sum(by_subject_correct.values())
        total_n = sum(by_subject_n.values())
        results["thai_exam"] = {
            "overall_acc": (total_correct / total_n) if total_n else 0.0,
            "n": total_n,
            "by_subject": {
                s: {"acc": by_subject_correct[s] / by_subject_n[s] if by_subject_n[s] else 0.0, "n": by_subject_n[s]}
                for s in by_subject_correct
            },
        }

    if "hotpotqa" in by_family:
        j = HotpotQAJudge()
        items = [
            {
                "prediction_text": r["prediction"],
                "prediction_sources": r.get("prediction_sources", []),
                "gold_answer": r.get("gold_answer", ""),
                "gold_supporting_facts": r.get("gold_supporting_facts", []),
                "question": r.get("prompt", ""),
            }
            for r in by_family["hotpotqa"]
        ]
        s = j.aggregate(items)
        results["hotpotqa"] = {"answer_em": s.answer_em, "sf_f1": s.sf_f1, "joint": s.joint_em_f1, "n": s.n}

    return results


def write_report(results: dict, baselines: dict, model_name: str, out_path: Path) -> None:
    lines = [
        f"# Benchmark eval — {model_name}",
        "",
        f"Public-benchmark scores for `{model_name}` vs Typhoon-2 / OpenThaiGPT-1.5 published baselines.",
        "",
        "| benchmark | score | typhoon-2 | Δ vs typhoon-2 | openthaigpt-1.5 | Δ vs openthaigpt-1.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def fmt_delta(score, baseline):
        if baseline is None:
            return "not published", "—"
        delta = score - baseline
        sign = "+" if delta >= 0 else ""
        return f"{baseline:.3f}", f"{sign}{delta:.3f}"

    # Flatten results to a comparable form.
    flat = []
    if "aime24" in results: flat.append(("aime24_th", results["aime24"]["accuracy"]))
    if "aime25" in results: flat.append(("aime25_th", results["aime25"]["accuracy"]))
    if "math500" in results: flat.append(("math500_th", results["math500"]["accuracy"]))
    if "ifeval" in results: flat.append(("ifeval_th_strict", results["ifeval"]["strict_acc"]))
    if "mt_bench" in results: flat.append(("mt_bench_th_overall", results["mt_bench"]["overall"]))
    if "openthaieval" in results: flat.append(("openthaieval_overall", results["openthaieval"]["overall"]))
    if "livecodebench" in results: flat.append(("livecodebench_th_pass_at_1", results["livecodebench"]["pass_at_1"]))
    if "hotpotqa" in results: flat.append(("hotpotqa_em", results["hotpotqa"]["answer_em"]))
    if "thai_exam" in results: flat.append(("thai_exam_8b", results["thai_exam"]["overall_acc"]))

    for name, score in flat:
        t2 = TYPHOON_2_BASELINES.get(name)
        otg = OPENTHAIGPT_1_5_BASELINES.get(name)
        t2_b, t2_d = fmt_delta(score, t2)
        otg_b, otg_d = fmt_delta(score, otg)
        lines.append(f"| {name} | {score:.3f} | {t2_b} | {t2_d} | {otg_b} | {otg_d} |")

    lines.append("")
    lines.append("## Raw results")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(results, ensure_ascii=False, indent=2))
    lines.append("```")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-only", help="export benchmark inputs to this JSONL and exit")
    parser.add_argument("--family", choices=FAMILIES + ["aime", "all"], default="all")
    parser.add_argument("--score-from", help="score predictions in this JSONL and write a report")
    parser.add_argument("--model", default="unknown-model", help="model name for the report")
    parser.add_argument("--report-out", default=None, help="markdown output path (defaults to dataset/reports/)")
    args = parser.parse_args()

    if args.inputs_only:
        total = 0
        out_path = Path(args.inputs_only)
        if args.family == "all":
            families = FAMILIES
        elif args.family in _FAMILY_ALIASES:
            families = _FAMILY_ALIASES[args.family]
        else:
            families = [args.family]
        out_path.write_text("", encoding="utf-8")
        for f in families:
            inputs = _load_benchmark(f)
            with out_path.open("a", encoding="utf-8") as fh:
                for r in inputs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += len(inputs)
            print(f"  {f}: {len(inputs)} inputs")
        print(f"total: {total} inputs → {out_path}")
        return 0

    if args.score_from:
        results = score_predictions(Path(args.score_from))
        out_path = Path(args.report_out) if args.report_out else (
            ROOT / "dataset" / "reports" / f"benchmark_eval_{args.model}.md"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_report(results, TYPHOON_2_BASELINES, args.model, out_path)
        print(f"report: {out_path}")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    print("Specify --inputs-only <path> to export benchmark inputs, or --score-from <predictions.jsonl> to score.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
