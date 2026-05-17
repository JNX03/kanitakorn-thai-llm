"""One-shot dedupe pass — removes duplicate-id records from train/+val/.

After this, every id appears exactly once in the union of train + val.
Use after a multi-source seed/restore that may have produced collisions.

Run:
    python tools/dedupe_jsonl.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dedupe_dir(d: Path, seen: set[str]) -> tuple[int, int]:
    kept = removed = 0
    for f in sorted(d.glob("*.jsonl")):
        if sys.platform == "win32":
            subprocess.run(["attrib", "-R", str(f)], check=False, capture_output=True)
        lines = f.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines:
            if not line.strip():
                continue
            rid = json.loads(line)["id"]
            if rid in seen:
                removed += 1
                continue
            seen.add(rid)
            out.append(line)
            kept += 1
        if out:
            f.write_text("\n".join(out) + "\n", encoding="utf-8")
        else:
            f.unlink()
    return kept, removed


def main() -> int:
    seen: set[str] = set()
    t_kept, t_removed = dedupe_dir(ROOT / "dataset" / "train", seen)
    v_kept, v_removed = dedupe_dir(ROOT / "dataset" / "validation", seen)
    print(f"train: kept {t_kept}, removed {t_removed} duplicates")
    print(f"val:   kept {v_kept}, removed {v_removed} duplicates")
    print(f"total unique: {len(seen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
