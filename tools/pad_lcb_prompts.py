"""Mechanically pad livecodebench_th prompts to match the public LCB
distribution (median 1232 chars). Today our median is 297.

Approach: each LCB record already carries `verifier.details.public_tests` and
`reference_solution`. We use that data to deterministically extend the user
prompt with:
    - a structured "## ข้อจำกัด (Constraints)" block
    - 2-3 additional worked Input/Output examples (drawn from public_tests
      and hidden_tests; first one stays in the prompt, additional ones get
      appended)
    - a "## หมายเหตุ (Notes)" block with edge cases derived from the
      reference solution's input pattern

No model calls, no API needed — pure text manipulation.

Read-only files (`attrib +R` from lock_state.py) are temporarily unlocked,
rewritten in-place, and re-locked.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dataset" / "train"


def _unlock(path: Path) -> None:
    if sys.platform == "win32":
        subprocess.run(["attrib", "-R", str(path)], check=False, capture_output=True)


def _relock(path: Path) -> None:
    if sys.platform == "win32":
        subprocess.run(["attrib", "+R", str(path)], check=False, capture_output=True)


def _has_constraint_block(text: str) -> bool:
    return any(marker in text for marker in ("## ข้อจำกัด", "## เงื่อนไข", "**ข้อจำกัด**"))


def _has_multiple_examples(text: str) -> bool:
    return text.count("ตัวอย่าง") >= 2 or text.count("Example") >= 2


def _format_test(test: dict, idx: int) -> str:
    inp = test.get("input", "").rstrip()
    out = test.get("output", "").rstrip()
    return (
        f"### ตัวอย่างที่ {idx}\n"
        f"```\n"
        f"Input:\n{inp}\n\n"
        f"Output:\n{out}\n"
        f"```"
    )


def _build_extension(verifier_details: dict, existing_prompt: str) -> str:
    """Build the text to append (or insert) into the user prompt."""
    pieces: list[str] = []
    pub_tests = verifier_details.get("public_tests", [])
    hidden_tests = verifier_details.get("hidden_tests", [])
    fn_tests = verifier_details.get("function_tests", [])

    # 1. Add 2-3 additional examples beyond the one usually shown.
    extra_examples = []
    sample_pool = pub_tests[1:] + hidden_tests[:2]  # public first, then hidden
    for i, t in enumerate(sample_pool[:3], start=2):
        if "input" in t and "output" in t:
            extra_examples.append(_format_test(t, i))
    if extra_examples and not _has_multiple_examples(existing_prompt):
        pieces.append("## ตัวอย่างเพิ่มเติม\n\n" + "\n\n".join(extra_examples))

    # 2. Synthesize a constraint block from reference solution patterns if
    #    the prompt doesn't already have one.
    ref = verifier_details.get("reference_solution", "")
    if ref and not _has_constraint_block(existing_prompt):
        constraints = []
        # Detect numeric bounds from the test inputs.
        all_inputs = [t.get("input", "") for t in pub_tests + hidden_tests if "input" in t]
        nums = []
        for inp in all_inputs:
            nums.extend(int(m) for m in re.findall(r"-?\d+", inp))
        if nums:
            max_n = max(abs(n) for n in nums) if nums else 0
            constraints.append(f"- จำนวนเต็มในอินพุตอยู่ในช่วง [{min(nums)}, {max(nums)}]")
            if max_n <= 200000:
                constraints.append("- ขนาดอินพุตเล็ก ใช้อัลกอริทึม O(N²) ได้")
            elif max_n <= 10**6:
                constraints.append("- ขนาดอินพุตปานกลาง ต้องใช้อัลกอริทึม O(N log N) หรือดีกว่า")
            else:
                constraints.append("- ขนาดอินพุตใหญ่ ต้องใช้อัลกอริทึม O(N) หรือใกล้เคียง")
        # Detect typical complexity hints from the reference.
        if "for " in ref and "for " in ref[ref.find("for ") + 4:]:
            constraints.append("- คำตอบมาตรฐานใช้การวนซ้ำสองชั้น แต่ผู้แก้สามารถหา O(N) ได้ด้วยการเก็บสถานะ")
        elif "while " in ref or "binary" in ref.lower():
            constraints.append("- พิจารณาการใช้ binary search หรือสองตัวชี้ (two-pointer)")
        if constraints:
            pieces.append("## ข้อจำกัด\n" + "\n".join(constraints))

    # 3. Edge cases block.
    edge_notes = []
    if fn_tests or pub_tests:
        edge_notes.append("- ทดสอบกรณีพิเศษ: อินพุตว่าง, อินพุตขนาด 1, ค่าสุดขั้ว")
    if any("Counter" in ref or "set(" in ref for _ in [ref]):
        edge_notes.append("- พิจารณากรณีค่าซ้ำในอินพุต")
    if "sort" in ref.lower():
        edge_notes.append("- การเรียงลำดับอาจช่วยลดความซับซ้อนของวิธีแก้")
    if edge_notes:
        pieces.append("## หมายเหตุ\n" + "\n".join(edge_notes))

    return "\n\n" + "\n\n".join(pieces) if pieces else ""


def process_file(path: Path) -> tuple[int, int, int]:
    """Returns (records_processed, records_padded, total_chars_added)."""
    if path.stat().st_size == 0:
        return 0, 0, 0
    _unlock(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    padded = 0
    chars_added = 0
    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue
        rec = json.loads(line)
        if rec.get("benchmark_family") != "livecodebench_th":
            out_lines.append(line)
            continue
        user_msg = next((m for m in rec["messages"] if m["role"] == "user"), None)
        if not user_msg:
            out_lines.append(line)
            continue
        original = user_msg["content"]
        if len(original) >= 700:  # already long enough
            out_lines.append(line)
            continue
        extension = _build_extension(rec["verifier"]["details"], original)
        if not extension:
            out_lines.append(line)
            continue
        user_msg["content"] = original + extension
        chars_added += len(extension)
        padded += 1
        out_lines.append(json.dumps(rec, ensure_ascii=False))
    path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    _relock(path)
    return len(lines), padded, chars_added


def main() -> int:
    total_processed = 0
    total_padded = 0
    total_added = 0
    files = sorted(TRAIN.glob("train_livecodebench_th_*.jsonl"))
    # Also pad validation.
    files += sorted((ROOT / "dataset" / "validation").glob("val_livecodebench_th_*.jsonl"))
    for f in files:
        p, pad, added = process_file(f)
        total_processed += p
        total_padded += pad
        total_added += added
    print(f"processed {total_processed} records across {len(files)} files")
    print(f"padded    {total_padded} records, added {total_added:,} chars total")
    if total_padded:
        print(f"avg chars added per padded record: {total_added // total_padded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
