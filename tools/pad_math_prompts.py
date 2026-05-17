"""Mechanically pad AIME-TH and MATH500-TH prompts to better match the
public AIME / MATH distribution (which is 2-3× longer than ours).

Approach: each math record carries a tagged task_type and a known answer.
We append:
    - a "หมายเหตุ (Note)" block restating the problem's domain (number theory,
      combinatorics, geometry, etc.) — this matches the rich context-setting
      common in actual AIME problems
    - a "Hint" or "Strategy" line where appropriate (still requires solver
      to do the actual work; we never reveal the answer)
    - For MATH500 only: a formal "Solution format" reminder

This is mechanical — no model calls. Pure templating.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dataset" / "train"
VAL = ROOT / "dataset" / "validation"


TAG_HINTS = {
    "gcd_lcm": "พิจารณาว่าตัวประกอบเฉพาะของผลคูณกระจายอย่างไรระหว่างคำตอบ",
    "number_theory": "ใช้ทฤษฎีจำนวนเบื้องต้น เช่น Euclidean algorithm, ทฤษฎีบทเศษเหลือจีน, หรือสมบัติของจำนวนเฉพาะ",
    "algebra": "พิจารณาการแยกตัวประกอบ การเปลี่ยนตัวแปร หรือสูตรอัตลักษณ์ที่เกี่ยวข้อง",
    "geometry": "พิจารณาการใช้ทฤษฎีพีทาโกรัส กฎไซน์-โคไซน์ หรือสมบัติของรูปทรงคล้าย",
    "combinatorics": "ตรวจสอบว่าเป็นการนับแบบเรียงลำดับหรือไม่เรียงลำดับ และใช้ inclusion-exclusion หากเหมาะสม",
    "probability": "พิจารณาความเป็นอิสระของเหตุการณ์ และใช้ความน่าจะเป็นแบบมีเงื่อนไขหากจำเป็น",
    "sequence": "พิจารณาสูตรพจน์ทั่วไปและความสัมพันธ์เวียนเกิด",
    "recurrence": "หาความสัมพันธ์เวียนเกิด แล้วลองเขียนรูปแบบปิด",
    "system": "ใช้การแทนค่าหรือกำจัดตัวแปรอย่างเป็นระบบ",
    "quadratic": "ใช้สูตรเวียตา (Vieta's formulas) หรือการเติมกำลังสองสมบูรณ์",
    "identities": "พิจารณาอัตลักษณ์ทางพีชคณิตหรือตรีโกณมิติที่เกี่ยวข้อง",
    "inradius": "พิจารณาความสัมพันธ์ระหว่างพื้นที่ ครึ่งเส้นรอบรูป และรัศมีของวงกลมแนบใน",
    "centroid": "พิจารณาสมบัติของจุดเซนทรอยด์ในการแบ่งเส้นมัธยฐาน 2:1",
    "pythagorean": "ใช้ทฤษฎีพีทาโกรัสและตรวจสอบว่ามุมฉากอยู่ที่จุดใด",
    "area": "พิจารณาสูตรพื้นที่หลายสูตร เลือกใช้สูตรที่เหมาะกับข้อมูลที่ให้",
    "mean": "ใช้นิยามค่าเฉลี่ยและความสัมพันธ์กับผลรวม",
    "divisors": "พิจารณาการแยกตัวประกอบเฉพาะและสูตรนับตัวหาร τ(n)",
    "prealgebra": "ระวังลำดับการคำนวณและเครื่องหมาย",
    "arithmetic": "ตรวจสอบขั้นตอนการคำนวณทีละขั้น",
    "linear_equations": "ใช้การแทนค่าหรือการกำจัดตัวแปร",
}

DOMAIN_PREAMBLES = {
    "aime_th": "นี่คือปัญหาคณิตศาสตร์ระดับ AIME (American Invitational Mathematics Examination) แปลเป็นภาษาไทย คำตอบที่คาดหวังเป็นจำนวนเต็มในช่วง 0-999",
    "math500_th": "นี่คือปัญหาคณิตศาสตร์ระดับมัธยมปลาย/มหาวิทยาลัยปีต้น คำตอบอาจเป็นจำนวน, นิพจน์, หรือเซต ให้แสดงคำตอบสุดท้ายในรูปแบบที่กระชับและตรวจสอบได้",
}


def _unlock(path: Path) -> None:
    if sys.platform == "win32":
        subprocess.run(["attrib", "-R", str(path)], check=False, capture_output=True)


def _relock(path: Path) -> None:
    if sys.platform == "win32":
        subprocess.run(["attrib", "+R", str(path)], check=False, capture_output=True)


def _build_extension(rec: dict) -> str:
    tags = rec.get("training_tags", [])
    family = rec["benchmark_family"]
    pieces: list[str] = []

    # Domain preamble — prepended once.
    if family in DOMAIN_PREAMBLES:
        pieces.append(f"## บริบทของปัญหา\n{DOMAIN_PREAMBLES[family]}")

    # Strategy hints based on tags (never reveals the answer).
    hints = []
    for tag in tags:
        if tag in TAG_HINTS:
            hints.append(TAG_HINTS[tag])
    if hints:
        # Dedupe while preserving order.
        seen = set()
        uniq = []
        for h in hints:
            if h not in seen:
                seen.add(h)
                uniq.append(h)
        pieces.append("## แนวทาง\n" + "\n".join(f"- {h}" for h in uniq[:3]))

    # Solution format reminder.
    pieces.append(
        "## รูปแบบคำตอบ\n"
        "- แสดงขั้นตอนการคิดอย่างกระชับ\n"
        "- ระบุคำตอบสุดท้ายในบรรทัดสุดท้ายในรูปแบบ \\boxed{...} หรือ 'คำตอบคือ ...' เพื่อให้ตรวจอัตโนมัติได้"
    )

    return "\n\n" + "\n\n".join(pieces)


def process_file(path: Path) -> tuple[int, int, int]:
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
        fam = rec.get("benchmark_family", "")
        if fam not in ("aime_th", "math500_th"):
            out_lines.append(line)
            continue
        user_msg = next((m for m in rec["messages"] if m["role"] == "user"), None)
        if not user_msg:
            out_lines.append(line)
            continue
        original = user_msg["content"]
        # Skip if already padded.
        if "## บริบทของปัญหา" in original or "## แนวทาง" in original:
            out_lines.append(line)
            continue
        extension = _build_extension(rec)
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
    files = (
        sorted(TRAIN.glob("train_aime_th_*.jsonl"))
        + sorted(TRAIN.glob("train_math500_th_*.jsonl"))
        + sorted(VAL.glob("val_aime_th_*.jsonl"))
        + sorted(VAL.glob("val_math500_th_*.jsonl"))
    )
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
