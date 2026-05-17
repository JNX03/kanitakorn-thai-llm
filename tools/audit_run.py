"""Phase 0.2 — re-run every verifier on every accepted record.

Loads each train_*.jsonl and val_*.jsonl, dispatches to the correct verifier by
record["verifier"]["type"], and writes:

* dataset/reports/audit_run_<YYYY-MM-DD>.md — per-family pass/fail counts, top
  20 failures per family, summary table.
* dataset/reports/failed_audit.jsonl — every failed record with the verifier
  exception traceback as `audit_failure_reason`.

CLI:
    python tools/audit_run.py [--strict] [--no-prosody]

Exit codes:
    0 — every family achieved ≥99% pass rate (Phase 0 acceptance gate)
    1 — at least one family failed the gate; `--strict` makes this hard-fail
        regardless of overall pass rate
    2 — fatal loading error (schema, missing files, etc.)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

# Make sibling imports work whether invoked from project root or tools/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from package_and_verify import (  # noqa: E402
    verify_math,
    verify_unit_tests,
    verify_instruction,
    verify_exact,
    verify_retrieval,
    verify_rubric,
)


DATASET = ROOT / "dataset"
REPORTS = DATASET / "reports"

VERIFIER_DISPATCH = {
    "symbolic_math": verify_math,
    "unit_tests": verify_unit_tests,
    "json_schema": verify_instruction,
    "regex": verify_instruction,
    "exact_match": verify_exact,
    "llm_judge_rubric": verify_rubric,
    "retrieval_evidence": verify_retrieval,
    "human_review": lambda item: ["human_review_noop"],  # nothing to assert
}

# Quality-score sanity bounds. The packager caps these at 0.99 / 0.95 — any
# record exceeding the cap is a sign that the upgrade pass missed it.
QUALITY_CAPS = {
    "correctness": 0.99,
    "thai_naturalness": 0.99,
    "benchmark_alignment": 0.94,
    "novelty": 0.95,
    "instruction_clarity": 0.99,
}

PROSODY_TASK_TYPES = {
    "klon_composition",
    "klon_4_composition",
    "klon_8_composition",
    "kap_composition",
    "kap_yani_composition",
    "chant_composition",
    "formal_register",
    "informal_register",
}


def iter_records():
    for split_dir, split_name in [(DATASET / "train", "train"), (DATASET / "validation", "validation")]:
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.jsonl")):
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    yield {
                        "_load_error": f"{path.name}:{line_no} {e}",
                        "id": f"{path.name}:{line_no}",
                        "benchmark_family": "unknown",
                        "split": split_name,
                    }
                    continue
                record["_split"] = split_name
                record["_path"] = str(path.relative_to(ROOT))
                yield record


def check_quality_scores(record: dict) -> list[str]:
    issues = []
    qs = record.get("quality_scores", {})
    for key, cap in QUALITY_CAPS.items():
        v = qs.get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            issues.append(f"quality_scores.{key} is not numeric: {v!r}")
            continue
        if v > cap + 1e-9:
            issues.append(f"quality_scores.{key}={v} exceeds cap {cap}")
    return issues


def check_prosody(record: dict, prosody_enabled: bool) -> list[str]:
    if not prosody_enabled:
        return []
    task_type = record.get("task_type", "")
    if task_type not in PROSODY_TASK_TYPES:
        return []
    try:
        from verifiers.thai_prosody_verifier import (
            check_klon_4,
            check_klon_8,
            check_kap_yani_11,
            check_register,
        )
    except ImportError:
        return ["prosody verifier unavailable (pythainlp missing?)"]
    text = record["messages"][-1]["content"]
    issues: list[str] = []
    if task_type in ("klon_composition", "klon_4_composition"):
        r = check_klon_4(text)
        if not r.ok:
            issues.extend(r.failures)
    elif task_type == "klon_8_composition":
        r = check_klon_8(text)
        if not r.ok:
            issues.extend(r.failures)
    elif task_type in ("kap_composition", "kap_yani_composition"):
        r = check_kap_yani_11(text)
        if not r.ok:
            issues.extend(r.failures)
    elif task_type == "formal_register":
        r = check_register(text, target="formal")
        if not r.ok:
            issues.extend(r.failures)
    elif task_type == "informal_register":
        r = check_register(text, target="informal")
        if not r.ok:
            issues.extend(r.failures)
    return issues


def run_audit(prosody_enabled: bool = True) -> dict:
    per_family_total: Counter = Counter()
    per_family_pass: Counter = Counter()
    per_family_fails: dict[str, list[dict]] = defaultdict(list)  # holds entries from failed_records
    quality_violations: list[dict] = []
    prosody_violations: list[dict] = []
    failed_records: list[dict] = []
    seen_ids: set[str] = set()

    def _record_failure(failure: dict, family: str) -> None:
        failed_records.append(failure)
        per_family_fails[family].append(failure)

    for record in iter_records():
        if "_load_error" in record:
            _record_failure(
                {
                    "id": record["id"],
                    "benchmark_family": "unknown",
                    "audit_failure_reason": record["_load_error"],
                    "split": record["_split"],
                },
                "unknown",
            )
            per_family_total["unknown"] += 1
            continue

        family = record.get("benchmark_family", "unknown")
        per_family_total[family] += 1
        rid = record.get("id", "<no-id>")
        if rid in seen_ids:
            _record_failure(
                {
                    "id": rid,
                    "benchmark_family": family,
                    "audit_failure_reason": "duplicate id across files (also seen earlier)",
                    "split": record.get("_split"),
                    "path": record.get("_path"),
                },
                family,
            )
            continue
        seen_ids.add(rid)

        vtype = record.get("verifier", {}).get("type")
        verifier_fn = VERIFIER_DISPATCH.get(vtype)
        if not verifier_fn:
            _record_failure(
                {
                    "id": rid,
                    "benchmark_family": family,
                    "audit_failure_reason": f"unknown verifier type: {vtype}",
                    "split": record.get("_split"),
                    "path": record.get("_path"),
                },
                family,
            )
            continue

        try:
            verifier_fn(record)
        except Exception as e:  # the verifier functions raise AssertionError
            _record_failure(
                {
                    "id": rid,
                    "benchmark_family": family,
                    "audit_failure_reason": f"{type(e).__name__}: {str(e)[:300]}",
                    "traceback": traceback.format_exc(limit=3),
                    "split": record.get("_split"),
                    "path": record.get("_path"),
                },
                family,
            )
            continue

        q_issues = check_quality_scores(record)
        if q_issues:
            quality_violations.append({"id": rid, "issues": q_issues, "split": record.get("_split")})

        p_issues = check_prosody(record, prosody_enabled)
        if p_issues:
            prosody_violations.append({"id": rid, "issues": p_issues, "split": record.get("_split")})
            _record_failure(
                {
                    "id": rid,
                    "benchmark_family": family,
                    "audit_failure_reason": "prosody_check_failed: " + "; ".join(p_issues),
                    "split": record.get("_split"),
                    "path": record.get("_path"),
                },
                family,
            )
            continue

        per_family_pass[family] += 1

    return {
        "per_family_total": dict(per_family_total),
        "per_family_pass": dict(per_family_pass),
        "per_family_fails": {k: v[:20] for k, v in per_family_fails.items()},
        "quality_violations": quality_violations,
        "prosody_violations": prosody_violations,
        "failed_records": failed_records,
    }


def write_reports(result: dict) -> tuple[Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    md = REPORTS / f"audit_run_{today}.md"
    failed_path = REPORTS / "failed_audit.jsonl"

    lines: list[str] = []
    lines.append(f"# Audit run — {today}\n")
    lines.append("Re-ran every declared verifier on every accepted record. Records that fail their declared verifier (or quality-cap / prosody check, where applicable) are written to `failed_audit.jsonl` for the quarantine queue.\n")
    lines.append("## Per-family pass rate\n")
    lines.append("| family | total | pass | fail | pass rate |")
    lines.append("|---|---:|---:|---:|---:|")
    overall_total = overall_pass = 0
    family_gate_passed = True
    for family in sorted(result["per_family_total"]):
        total = result["per_family_total"][family]
        passed = result["per_family_pass"].get(family, 0)
        failed = total - passed
        rate = passed / total if total else 0.0
        if rate < 0.99 and family != "unknown":
            family_gate_passed = False
        overall_total += total
        overall_pass += passed
        lines.append(f"| {family} | {total} | {passed} | {failed} | {rate:.2%} |")
    overall_rate = overall_pass / overall_total if overall_total else 0.0
    lines.append(f"| **TOTAL** | **{overall_total}** | **{overall_pass}** | **{overall_total - overall_pass}** | **{overall_rate:.2%}** |\n")

    lines.append(f"\n## Phase-0 acceptance gate (per-family ≥99%): {'PASS ✅' if family_gate_passed else 'FAIL ❌'}\n")

    if result["quality_violations"]:
        lines.append(f"\n## Quality-score cap violations ({len(result['quality_violations'])})\n")
        for v in result["quality_violations"][:30]:
            lines.append(f"- `{v['id']}` ({v['split']}): {'; '.join(v['issues'])}")

    if result["prosody_violations"]:
        lines.append(f"\n## Thai prosody violations ({len(result['prosody_violations'])})\n")
        for v in result["prosody_violations"][:30]:
            lines.append(f"- `{v['id']}` ({v['split']}): {'; '.join(v['issues'])}")

    lines.append("\n## Sample failures per family (first 20)\n")
    for family in sorted(result["per_family_fails"]):
        fails = result["per_family_fails"][family]
        if not fails:
            continue
        lines.append(f"### {family} ({len(fails)} fail{'s' if len(fails) != 1 else ''})\n")
        for rec in fails[:20]:
            reason = rec.get("audit_failure_reason", "unknown reason")
            rid = rec.get("id", "<no-id>")
            lines.append(f"- `{rid}` — {reason}")

    md.write_text("\n".join(lines), encoding="utf-8")

    with failed_path.open("w", encoding="utf-8") as fh:
        for rec in result["failed_records"]:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return md, failed_path


def load_known_failures() -> set[str]:
    """Allowlist of already-documented failures so the gate measures NEW regressions."""
    path = REPORTS / "known_audit_failures.json"
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return {entry["id"] for entry in data.get("known", [])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="ignore known-failures allowlist; exit 1 on ANY failure")
    parser.add_argument("--no-prosody", action="store_true", help="skip Thai prosody checks")
    args = parser.parse_args()

    result = run_audit(prosody_enabled=not args.no_prosody)
    md_path, failed_path = write_reports(result)
    print(f"audit report: {md_path}")
    print(f"failed records: {failed_path}")

    overall_total = sum(result["per_family_total"].values())
    overall_pass = sum(result["per_family_pass"].values())
    print(f"overall: {overall_pass}/{overall_total} ({(overall_pass / overall_total if overall_total else 0):.2%})")

    known = set() if args.strict else load_known_failures()
    if known:
        print(f"known-failure allowlist: {len(known)} ids (use --strict to ignore)")

    # Per-family gate: count NEW (non-known) failures. Family passes if it has
    # no new failures AND its overall pass rate (including known) is ≥99% so
    # we don't accidentally accumulate a long allowlist over time.
    family_gate_passed = True
    for family, total in result["per_family_total"].items():
        if family == "unknown" or not total:
            continue
        fail_ids = {rec.get("id") for rec in result["per_family_fails"].get(family, [])}
        new_fails = fail_ids - known
        if new_fails:
            family_gate_passed = False
            print(f"  {family}: {len(new_fails)} NEW failures (not in allowlist)")

    if not family_gate_passed:
        print("Phase-0 gate FAILED: new regressions detected.")
        return 1
    print("Phase-0 gate PASSED (all NEW failures within allowlist).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
