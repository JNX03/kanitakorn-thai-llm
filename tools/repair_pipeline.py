"""Phase 4 — repair pipeline orchestrator.

Runs the validate-and-repair phases in order, stopping on the first failure:

    Phase 0.1 — rebalance_validation_split.py
    Phase 0.2 — audit_run.py (with prosody)
    Phase 3   — few_shot_collator.py
    Phase 3.2 — build_train_manifest.py
    Phase 1.4 — benchmark_eval.py (export inputs only, since model is external)

Each step writes its own report under dataset/reports/pipeline_<YYYY-MM-DD>/.
Skips can be requested via --skip-phase 0.1, --skip-phase 0.2, etc.

CLI:
    python tools/repair_pipeline.py [--dry-run] [--skip-phase 0.1] [--skip-phase 0.2] ...
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, cmd: list[str], log_path: Path) -> bool:
    print(f"\n=== {name} ===")
    print("  cmd:", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        fh.write(f"# {name}\n## cmd\n{' '.join(cmd)}\n## stdout\n{proc.stdout}\n## stderr\n{proc.stderr}\n## returncode\n{proc.returncode}\n")
    print(f"  exit={proc.returncode}, log={log_path}")
    if proc.returncode != 0:
        # Surface the tail of stderr.
        tail = "\n".join(proc.stderr.splitlines()[-10:])
        print(f"  stderr (tail):\n{tail}")
    return proc.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report-only mode; tools won't mutate disk")
    parser.add_argument("--skip-phase", action="append", default=[], help="skip a phase id (e.g. 0.1, 0.2, 3, 1.4)")
    args = parser.parse_args()

    today = dt.date.today().isoformat()
    out_dir = ROOT / "dataset" / "reports" / f"pipeline_{today}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, str, list[str]]] = [
        ("0.1", "rebalance validation split",
         [sys.executable, "tools/rebalance_validation_split.py"]
         + (["--dry-run"] if args.dry_run else [])),
        ("0.2", "audit_run",
         [sys.executable, "tools/audit_run.py", "--no-prosody"]),
        ("3",   "few_shot_collator",
         [sys.executable, "tools/few_shot_collator.py"]),
        ("3.2", "build_train_manifest",
         [sys.executable, "tools/build_train_manifest.py"]),
        ("1.4", "benchmark_eval (inputs export)",
         [sys.executable, "tools/benchmark_eval.py", "--inputs-only", str(out_dir / "benchmark_inputs.jsonl"), "--family", "all"]),
    ]

    results: list[dict] = []
    halted = False
    for phase_id, name, cmd in plan:
        if phase_id in args.skip_phase:
            print(f"\n=== {phase_id} {name}: SKIPPED ===")
            results.append({"phase": phase_id, "name": name, "status": "skipped"})
            continue
        if halted:
            print(f"\n=== {phase_id} {name}: NOT RUN (prior failure) ===")
            results.append({"phase": phase_id, "name": name, "status": "not_run"})
            continue
        log_path = out_dir / f"step_{phase_id.replace('.', '_')}.md"
        ok = run_step(f"phase {phase_id}: {name}", cmd, log_path)
        results.append({"phase": phase_id, "name": name, "status": "pass" if ok else "fail", "log": str(log_path.relative_to(ROOT))})
        if not ok:
            halted = True

    summary = {
        "date": today,
        "dry_run": args.dry_run,
        "skipped": args.skip_phase,
        "results": results,
        "overall": "PASS" if all(r["status"] in ("pass", "skipped") for r in results) else "FAIL",
    }
    (out_dir / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Pipeline summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
