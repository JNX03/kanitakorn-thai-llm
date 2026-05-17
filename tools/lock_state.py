"""Lock the dataset into a state that survives an external file-sync revert.

Observed behavior in this project: `Desktop\\kanitakornv2\\dataset\\` is in a
synced folder (OneDrive / iCloud / Dropbox), and the sync layer reverts new
files within seconds. The verifier scripts run correctly, but their writes
keep getting undone.

This script:
    1. Re-runs `rebalance_validation_split.py` to produce the canonical split
    2. Re-runs `seed_teacher_loop.py` to produce the canonical teacher_loop seed
    3. Copies the result into `tools/_locked_snapshot/` (NOT in the synced
       dataset/ folder — tools/ has been observed to persist)
    4. On Windows: sets the canonical files to read-only via `attrib +R` so
       the sync layer can't silently overwrite them. On POSIX: `chmod 444`.
    5. Writes `tools/_locked_snapshot/restore.py` — a tiny script the user
       can run after disabling sync to copy the snapshot back if anything
       gets clobbered again.

Run once:
    python tools/lock_state.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
SNAPSHOT = ROOT / "tools" / "_locked_snapshot"


def run(cmd: list[str]) -> int:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def make_readonly(path: Path) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["attrib", "+R", str(path)], check=False)
        else:
            os.chmod(path, 0o444)
    except Exception as e:
        print(f"  warn: couldn't make {path} read-only: {e}")


def write_restore_script(snapshot_dir: Path) -> None:
    restore = snapshot_dir / "restore.py"
    restore.write_text(
        '''"""Restore the locked snapshot back into dataset/ folder.

Run after disabling OneDrive/iCloud sync on the project folder.
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE.parents[1] / "dataset"

for snap_dir in ("train", "validation"):
    src = HERE / snap_dir
    dst = DATASET / snap_dir
    if not src.exists():
        continue
    dst.mkdir(parents=True, exist_ok=True)
    # Delete existing files (they may be stale).
    for f in dst.glob("*.jsonl"):
        try:
            f.unlink()
        except PermissionError:
            import subprocess, sys
            if sys.platform == "win32":
                subprocess.run(["attrib", "-R", str(f)], check=False)
                f.unlink()
    for f in src.glob("*.jsonl"):
        shutil.copy2(f, dst / f.name)
    print(f"restored {snap_dir}/ ({sum(1 for _ in src.glob('*.jsonl'))} files)")
print("done — re-run tools/audit_run.py to verify")
''',
        encoding="utf-8",
    )
    print(f"  wrote {restore}")


def main() -> int:
    # Seed first so rebalance sees all records (avoids duplicates from
    # records already promoted to val on a prior run).
    print("=== Step 1a: seed teacher loop ===")
    rc = run([sys.executable, "tools/seed_teacher_loop.py"])
    if rc != 0:
        print("seed_teacher_loop failed — aborting")
        return rc
    print("\n=== Step 1b: seed hotpotqa ===")
    rc = run([sys.executable, "tools/seed_hotpotqa.py"])
    if rc != 0:
        print("seed_hotpotqa failed — continuing")
    print("\n=== Step 1c: seed Thai exam ===")
    rc = run([sys.executable, "tools/seed_thai_exam.py"])
    if rc != 0:
        print("seed_thai_exam failed — continuing")

    print("\n=== Step 2: rebalance validation split ===")
    rc = run([sys.executable, "tools/rebalance_validation_split.py"])
    if rc != 0:
        print("rebalance failed — aborting")
        return rc

    print("\n=== Step 3: snapshot to non-synced location ===")
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    for sub in ("train", "validation"):
        src = DATASET / sub
        dst = SNAPSHOT / sub
        if dst.exists():
            shutil.rmtree(dst)
        if src.exists():
            shutil.copytree(src, dst)
            n = sum(1 for _ in dst.glob("*.jsonl"))
            print(f"  snapshotted {sub}/ ({n} files)")

    print("\n=== Step 4: make canonical files read-only ===")
    n_locked = 0
    for sub in ("train", "validation"):
        for f in (DATASET / sub).glob("*.jsonl"):
            make_readonly(f)
            n_locked += 1
    print(f"  locked {n_locked} files")

    print("\n=== Step 5: write restore.py ===")
    write_restore_script(SNAPSHOT)

    print("\n=== DONE ===")
    print(f"Snapshot: {SNAPSHOT}")
    print("If the sync layer reverts dataset/ again:")
    print("  1) pause OneDrive/iCloud sync")
    print(f"  2) python {SNAPSHOT / 'restore.py'}")
    print("  3) python tools/audit_run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
