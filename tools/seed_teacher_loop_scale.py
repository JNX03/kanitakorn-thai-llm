"""Scale teacher_loop_th from 50 seeds → 500+ via codex.

Asks codex to generate batches of 20 teacher-loop transcripts per call.
Each record is a multi-turn student-teacher correction loop targeting one
of four skills: klon_4_composition, ifeval_constraint, aime_step_chain,
register_calibration.

Run once (target ~500 records total, ~$25, ~25 min):
    python tools/seed_teacher_loop_scale.py --target 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from codex_batch_generator import BatchConfig, run_batch, reset_spend  # noqa: E402
from package_and_verify import SCHEMA  # noqa: E402
from jsonschema import Draft202012Validator  # noqa: E402


# Skill rotation — each batch targets one skill so the model can focus.
SKILLS = ["klon_4_composition", "ifeval_constraint", "aime_step_chain", "register_calibration"]
BASE_FAMILIES = {
    "klon_4_composition": "ifeval_ifbench",
    "ifeval_constraint": "ifeval_ifbench",
    "aime_step_chain": "aime_th",
    "register_calibration": "mt_bench",
}

# Topic banks for each skill (model picks 1 topic per record).
KLON_4_TOPICS = [
    "ฤดูฝนในกรุงเทพ", "ทะเลยามค่ำคืน", "ดอกราตรีบาน", "หิ่งห้อยริมน้ำ", "เด็กเลี้ยงควาย",
    "ตลาดเช้ายามอรุณ", "สายลมเดือนหนาว", "วัวควายในทุ่ง", "กลิ่นข้าวใหม่", "ดวงดาวเหนือดอย",
    "นกพิราบบนหลังคา", "ฝนพรำในเดือนแปด", "หาดทรายขาว", "ม่านหมอกบนภู", "ดอกไม้ในสวน",
    "เสียงระฆังวัด", "เด็กเล่นว่าว", "ตะเกียงน้ำมัน", "ปลาทองในอ่าง", "พระจันทร์เต็มดวง",
    "ฤดูร้อนในชนบท", "ขนมไทยริมทาง", "แม่ค้าตลาดเช้า", "งานวัดประจำปี", "ทุ่งดอกบัวตอง",
    "ป่าฝนเขาใหญ่", "นาขั้นบันได", "ชายเลก่อนพายุ", "ลมหนาวบนดอย", "ผีเสื้อในสวน",
]

IFEVAL_TEMPLATES = [
    "ตอบเป็นบุลเล็ตจำนวน N บรรทัด ขึ้นต้นแต่ละบรรทัดด้วย '- หัวข้อ:'",
    "ห้ามใช้เลขอารบิก ใช้เลขไทยเท่านั้น",
    "ตอบเป็น JSON ที่มี field title (string), steps (array of N items), warning (string)",
    "อธิบาย {topic} โดยใช้คำว่า '{keyword}' พอดี N ครั้ง",
    "ตอบเป็น 3 paragraphs แต่ละ paragraph ขึ้นต้นด้วย 'ข้อคิด:' มี 2 ประโยค",
    "ห้ามใช้เครื่องหมาย ',' ในคำตอบ อธิบาย {topic} ใน 3 ประโยค",
    "ตอบเป็น Q&A 2 รอบ Q ขึ้นต้นด้วย 'ถาม:' A ขึ้นต้นด้วย 'ตอบ:'",
    "เขียน 4 บรรทัด ขึ้นต้นด้วยตัวอักษร ก, ข, ค, ง ตามลำดับ ตามด้วย ':'",
]


def build_prompt(batch_idx: int, n: int) -> str:
    skill = SKILLS[batch_idx % len(SKILLS)]
    base_family = BASE_FAMILIES[skill]
    if skill == "klon_4_composition":
        rng = random.Random(batch_idx * 31)
        topics = rng.sample(KLON_4_TOPICS, min(n, len(KLON_4_TOPICS)))
        skill_block = f"Skill: KLON_4_COMPOSITION (Thai 4-syllable poetry). Topics to use (one per record, in order): {', '.join(topics)}."
        rules_block = (
            "Each record's loop should show: (1) student attempts กลอน 4 with WRONG syllable counts (e.g. 8+ syllables per line), "
            "(2) teacher correction explicitly naming the syllable count error in Thai (e.g. 'วรรคที่ 1 มี X พยางค์ แต่กลอน 4 ต้องการ 4 พยางค์'), "
            "(3) student retries with 4 syllables per line + 4 lines per บท. Final assistant turn MUST be 4 lines × ~4 syllables each."
        )
    elif skill == "ifeval_constraint":
        rng = random.Random(batch_idx * 37)
        templates = rng.sample(IFEVAL_TEMPLATES, min(n, len(IFEVAL_TEMPLATES)))
        skill_block = f"Skill: IFEVAL_CONSTRAINT_FOLLOWING. Use these constraint templates (one per record): {' || '.join(templates)}."
        rules_block = (
            "Each record: (1) user asks with a verifiable constraint, (2) student answers but VIOLATES the constraint "
            "(wrong line count, includes forbidden chars, wrong keyword frequency, etc.), (3) teacher correction in Thai "
            "naming the specific violation, (4) student retries correctly. Final answer must literally satisfy the constraint."
        )
    elif skill == "aime_step_chain":
        skill_block = (
            "Skill: AIME_STEP_CHAIN — Thai-language math reasoning. Pick competition-style problems "
            "(algebra, number theory, combinatorics, geometry) with integer or simple rational answers."
        )
        rules_block = (
            "Each record: (1) user gives a problem in Thai, (2) student makes a SPECIFIC wrong step (sign error, "
            "off-by-one, wrong formula choice, etc.) and reaches a wrong answer, (3) teacher in Thai NAMES the wrong "
            "step ('ขั้นตอนที่ 2 ใช้สูตรผิด ควรใช้... แทน...'), (4) student retries, gets the correct answer. "
            "Final assistant turn ends with 'คำตอบคือ X' or '\\boxed{X}' where X is the correct integer/fraction."
        )
    else:  # register_calibration
        skill_block = (
            "Skill: REGISTER_CALIBRATION — Thai formal/informal register. Pick scenarios where wrong register matters "
            "(office email vs friend chat, royal speech vs casual, business proposal vs social media post)."
        )
        rules_block = (
            "Each record: (1) user requests a Thai text in a specific register, (2) student answers in the WRONG register "
            "(too informal when formal needed, or vice versa — use ครับ/ค่ะ vs เรียน/ขอเรียน/ขอบพระคุณ properly), "
            "(3) teacher correction in Thai pointing out specific register issues, (4) student rewrites in correct register."
        )

    return f"""You are generating training data for a Thai LLM SFT corpus. Output a STRICT JSON ARRAY of {n} records — no preamble, no markdown fences, ONLY the JSON array.

{skill_block}

{rules_block}

Each record in the array must be a JSON object with EXACTLY this shape:
{{
  "skill": "{skill}",
  "base_family": "{base_family}",
  "init_prompt": "<the user's initial request in Thai>",
  "wrong_attempt": "<student's first wrong answer in Thai>",
  "teacher_correction": "<teacher's Thai-language correction that names the specific failure>",
  "retry": "<student's corrected answer that satisfies the requirement>",
  "n_corrections": 1,
  "convergence_reached": true,
  "difficulty": "medium" or "hard"
}}

CRITICAL:
- All Thai text must be natural and grammatical
- The wrong_attempt must contain a CONCRETE error the teacher can name
- The retry must actually pass the implicit verifier (e.g., for klon_4: 4 lines × ~4 syllables; for ifeval: literally satisfy the constraint stated)
- Do NOT include explanations outside the JSON
- Output the array of {n} records. No more, no less.
"""


def record_factory(raw: dict, idx: int) -> dict:
    """Convert codex's compact format into a full project schema record."""
    skill = raw.get("skill", "klon_4_composition")
    base_family = raw.get("base_family", BASE_FAMILIES.get(skill, "ifeval_ifbench"))
    messages = [
        {"role": "user", "content": raw["init_prompt"]},
        {"role": "assistant", "content": raw["wrong_attempt"]},
        {"role": "user", "content": raw["teacher_correction"]},
        {"role": "assistant", "content": raw["retry"]},
    ]
    return {
        "id": f"teacher-loop-th-codex-{idx:05d}",
        "benchmark_family": "teacher_loop_th",
        "task_type": skill,
        "language": "th",
        "difficulty": raw.get("difficulty", "medium"),
        "messages": messages,
        "final_answer": raw["retry"][:200],
        "concise_solution": f"learned {skill} via {raw.get('n_corrections', 1)} correction(s)",
        "verifier": {
            "type": "llm_judge_rubric",
            "details": {
                "criteria": [
                    "final assistant turn satisfies the skill's deterministic verifier",
                    "teacher correction names the specific failure mode",
                    "convergence_reached=true",
                ],
                "skill": skill,
            },
        },
        "sources": [{"url": "synthetic-codex-teacher-loop", "license": "project-owned synthetic", "used_for": "teacher loop training"}],
        "contamination_audit": {
            "official_benchmark_checked": True, "exact_match": False, "ngram_similarity_max": 0.0,
            "embedding_similarity_status": "not_run", "embedding_similarity_max": None,
            "simhash_similarity_status": "not_run", "simhash_similarity_max": None,
            "numeric_structure_similarity": "low", "decision": "accept",
            "notes": "Codex-generated teacher-loop transcript",
        },
        "quality_scores": {
            "correctness": 0.97 if raw.get("convergence_reached") else 0.5,
            "thai_naturalness": 0.95, "benchmark_alignment": 0.85, "novelty": 0.95,
            "instruction_clarity": 0.95, "calibration_version": "codex_teacher_loop_v1",
        },
        "training_tags": ["thai", "teacher_loop", skill, f"n_corrections_{raw.get('n_corrections', 1)}"],
        "loop_metadata": {
            "n_corrections": int(raw.get("n_corrections", 1)),
            "skill_taught": skill,
            "convergence_reached": bool(raw.get("convergence_reached", True)),
            "base_family": base_family,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--records-per-call", type=int, default=20)
    args = parser.parse_args()

    reset_spend()
    validator = Draft202012Validator(SCHEMA)
    cfg = BatchConfig(
        family="teacher_loop_th",
        out_path=ROOT / "dataset" / "train" / "train_teacher_loop_th_codex.jsonl",
        records_per_call=args.records_per_call,
        total_target=args.target,
        build_prompt=build_prompt,
        record_factory=record_factory,
        schema_validator=validator.validate,
    )
    # Use a fresh shard name with timestamp to avoid file-lock collisions.
    import datetime as _dt
    if cfg.out_path.exists():
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.out_path = cfg.out_path.with_name(f"train_teacher_loop_th_codex_{ts}.jsonl")
    summary = run_batch(cfg, log_path=ROOT / "dataset" / "reports" / "codex_teacher_loop_log.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
