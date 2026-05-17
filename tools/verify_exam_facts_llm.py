"""LLM-judge fact verification for openthaieval Thai exam-prep records.

Different from verify_facts_llm.py (which checks `required_supporting_facts`
against fetched source pages). Exam records use `exact_match` verifier with
`accepted_answers` — there's a question, an answer, and a Thai explanation.

This judge asks codex: "Given the question, is the claimed answer factually
correct?" — no URL fetching needed because the questions are about
well-established facts (Thai history, literature, prosody, math, science).

Run:
    python tools/verify_exam_facts_llm.py \\
        --check dataset/train/train_openthaieval_thai_exam_seed.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "dataset" / "reports"


def _find_codex() -> str | None:
    found = shutil.which("codex") or shutil.which("codex.cmd")
    if found:
        return found
    for c in [
        Path(os.environ.get("APPDATA", "")) / "npm" / "codex.cmd",
        Path.home() / "AppData" / "Roaming" / "npm" / "codex.cmd",
    ]:
        if c.exists():
            return str(c)
    return None


_CODEX = _find_codex()

PROMPT_TEMPLATE = """\
You are a strict fact-verification auditor for a Thai exam-prep dataset. Decide if the claimed answer to the Thai question is factually correct, based on widely-accepted educational sources (Thai history textbooks, Royal Institute Dictionary, encyclopedias, well-known facts).

Return STRICT JSON only — no preamble, no markdown.

## Record id
{record_id}

## Question (Thai)
{question}

## Claimed answer
{answer}

## Claimed explanation (Thai)
{explanation}

## Accepted alternative answers
{accepted}

Return JSON:
{{
  "verdict": "OK" | "WARN" | "FAIL",
  "is_correct": true | false,
  "rationale": "<one short sentence in Thai or English>",
  "suggested_correction": "<if FAIL: the actually correct answer, else null>"
}}

Use FAIL when the claimed answer is factually WRONG (e.g., wrong king, wrong year, wrong term).
Use WARN when the claimed answer is partially correct or imprecise (e.g., year off by 1, term used loosely).
Use OK when the claimed answer is fully correct and the explanation supports it.
"""


def codex_call(prompt: str, timeout_s: float = 180.0) -> str:
    if not _CODEX:
        return ""
    try:
        proc = subprocess.run(
            [_CODEX, "exec", "--skip-git-repo-check", "-"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(ROOT),
        )
        return proc.stdout
    except subprocess.TimeoutExpired:
        return ""


def extract_json(text: str) -> dict:
    last_open = text.rfind("{")
    last_close = text.rfind("}")
    if last_open < 0 or last_close < last_open:
        return {"verdict": "FAIL", "rationale": "no JSON in codex output"}
    try:
        return json.loads(text[last_open : last_close + 1])
    except json.JSONDecodeError:
        return {"verdict": "FAIL", "rationale": "unparseable JSON"}


def verify_one(rec: dict) -> dict:
    question = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
    answer_msg = next((m["content"] for m in rec["messages"] if m["role"] == "assistant"), "")
    # Extract just the explanation part of the answer message.
    explanation = answer_msg
    if "หลักฐาน:" in answer_msg:
        explanation = answer_msg.split("หลักฐาน:", 1)[1].strip()
    elif "\n\n" in answer_msg:
        explanation = answer_msg.split("\n\n", 1)[1].strip()
    prompt = PROMPT_TEMPLATE.format(
        record_id=rec.get("id", "<no-id>"),
        question=question,
        answer=rec.get("final_answer", ""),
        explanation=explanation,
        accepted=", ".join(rec.get("verifier", {}).get("details", {}).get("accepted_answers", [])),
    )
    raw = codex_call(prompt)
    out = extract_json(raw)
    out["id"] = rec.get("id", "<no-id>")
    out["question_preview"] = question[:100]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-md", default=str(REPORTS / "exam_fact_verification_llm.md"))
    parser.add_argument("--out-jsonl", default=str(REPORTS / "exam_fact_verification_llm.jsonl"))
    args = parser.parse_args()

    p = Path(args.check)
    records: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("benchmark_family") != "openthaieval":
            continue
        records.append(rec)
        if args.limit and len(records) >= args.limit:
            break

    results: list[dict] = []
    for i, rec in enumerate(records, start=1):
        print(f"[{i}/{len(records)}] {rec.get('id')}...", flush=True)
        r = verify_one(rec)
        v = r.get("verdict", "?")
        rationale = (r.get("rationale") or "")[:120]
        print(f"  -> {v}  {rationale}", flush=True)
        results.append(r)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_jsonl).open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {"OK": 0, "WARN": 0, "FAIL": 0}
    for r in results:
        counts[r.get("verdict", "FAIL")] = counts.get(r.get("verdict", "FAIL"), 0) + 1

    lines = [
        "# Thai exam-prep fact verification (LLM-as-judge)",
        "",
        f"Judge: codex (gpt-5.5 xhigh). Checks if claimed answers are factually correct based on widely-accepted educational sources.",
        "",
        f"**OK: {counts.get('OK', 0)} · WARN: {counts.get('WARN', 0)} · FAIL: {counts.get('FAIL', 0)} · TOTAL: {len(results)}**",
        "",
        "| id | verdict | question | rationale | suggested correction |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        rid = r.get("id", "")
        v = r.get("verdict", "?")
        q = (r.get("question_preview") or "")[:60]
        rat = (r.get("rationale") or "")[:120]
        sug = (r.get("suggested_correction") or "—")[:60] if r.get("suggested_correction") else "—"
        lines.append(f"| {rid} | {v} | {q} | {rat} | {sug} |")
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out_md}")
    return 0 if counts.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
