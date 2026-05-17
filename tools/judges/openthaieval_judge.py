"""OpenThaiEval judge.

The iapp/openthaieval dataset has two flavors:
    * MCQ (O-NET / A-level / IC / TGAT / TPAT-1 / TGAT) — answered with a
      letter (1)/(2)/(3)/(4) or ก/ข/ค/ง. Scored by exact match.
    * Analytic — short-answer; scored by LLM equivalence judge.

The judge auto-detects MCQ vs analytic by whether the gold answer parses as a
letter-in-parens token.

Used by:
    * tools/benchmark_eval.py — runs against iapp/openthaieval cached parquet
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_judge import GPT55Judge, default_judge  # noqa: E402
from answer_formatter import extract_openthaieval  # noqa: E402


_MCQ_PARENS_RE = re.compile(r"\(([1-9]|[ก-ง])\)")
_MCQ_STANDALONE_DIGIT_RE = re.compile(r"\b([1-9])\b")
_MCQ_THAI_LETTER_RE = re.compile(r"(?:^|\s)([ก-ง])(?:[\s.,)]|$)")

# XNLI-style 3-class outputs that appear in openthaieval. Deterministic
# exact-match after normalization (lowercase, strip).
_XNLI_CLASSES = {"entailment", "contradiction", "neutral"}

# Thai entailment vocabulary in case the model answered in Thai.
_THAI_TO_XNLI = {
    "สรุปได้": "entailment", "ตามมา": "entailment", "เป็นไปตาม": "entailment",
    "ขัดแย้ง": "contradiction", "ไม่สอดคล้อง": "contradiction",
    "เป็นกลาง": "neutral", "ไม่เกี่ยว": "neutral", "ไม่แน่ชัด": "neutral",
}
_ANALYTIC_RUBRIC = (
    "Decide whether the candidate answer is equivalent in meaning to the gold "
    "answer for this Thai academic-exam question. Numbers and dates must "
    "match exactly. Synonyms and word-order differences are OK. Return JSON: "
    "{\"score\": 1 if equivalent else 0, \"rationale\": <short>}."
)


@dataclass
class OpenThaiEvalScore:
    mcq_acc: float
    analytic_acc: float
    overall_acc: float
    by_subject: dict[str, float]
    n_mcq: int
    n_analytic: int


class OpenThaiEvalJudge:
    def __init__(self, judge: GPT55Judge | None = None) -> None:
        self.judge = judge or default_judge()

    def score_one(self, prediction: str, gold: str, question: str = "") -> bool:
        # Run the canonical extractor first so messy model output gets cleaned.
        cleaned = extract_openthaieval(prediction)
        if self._is_mcq(gold):
            return self._extract_mcq(cleaned) == self._extract_mcq(gold)
        # XNLI-style 3-class — deterministic
        gold_xnli = self._extract_xnli(gold)
        if gold_xnli:
            pred_xnli = self._extract_xnli(cleaned)
            return pred_xnli == gold_xnli
        # Fall back to LLM judge for free-form analytic answers.
        r = self.judge.score_pointwise(
            prompt=f"Question: {question}\nGold answer: {gold}",
            response=prediction,
            rubric=_ANALYTIC_RUBRIC,
        )
        return r.score >= 1

    def aggregate(self, items: list[dict]) -> OpenThaiEvalScore:
        """Items: [{prediction, gold, subject (optional), question (optional)}]."""
        per_subject: dict[str, list[int]] = {}
        mcq_correct = mcq_n = analytic_correct = analytic_n = 0
        for it in items:
            gold = it["gold"]
            pred = it["prediction"]
            subject = it.get("subject", "unknown")
            ok = self.score_one(pred, gold, it.get("question", ""))
            per_subject.setdefault(subject, []).append(1 if ok else 0)
            if self._is_mcq(gold):
                mcq_n += 1
                if ok:
                    mcq_correct += 1
            else:
                analytic_n += 1
                if ok:
                    analytic_correct += 1
        n = mcq_n + analytic_n
        total_correct = mcq_correct + analytic_correct
        return OpenThaiEvalScore(
            mcq_acc=mcq_correct / mcq_n if mcq_n else 0.0,
            analytic_acc=analytic_correct / analytic_n if analytic_n else 0.0,
            overall_acc=total_correct / n if n else 0.0,
            by_subject={k: sum(v) / len(v) for k, v in per_subject.items()},
            n_mcq=mcq_n,
            n_analytic=analytic_n,
        )

    @staticmethod
    def _is_mcq(answer: str) -> bool:
        stripped = answer.strip()
        return bool(re.fullmatch(r"\(?[1-9]\)?", stripped) or re.fullmatch(r"\(?[ก-ฮ]\)?", stripped))

    @staticmethod
    def _extract_xnli(text: str) -> str:
        """Return one of {entailment, contradiction, neutral} or empty string."""
        if not text:
            return ""
        t = text.lower().strip()
        # Direct English label match.
        for cls in _XNLI_CLASSES:
            if t == cls or cls in t.split():
                return cls
        # Thai cue words.
        for cue, cls in _THAI_TO_XNLI.items():
            if cue in text:
                return cls
        return ""

    @staticmethod
    def _extract_mcq(text: str) -> str:
        # Prefer parens-wrapped choice (most explicit).
        m = _MCQ_PARENS_RE.search(text)
        if m:
            return m.group(1)
        # Fall back to standalone digit (e.g. "answer: 2").
        m = _MCQ_STANDALONE_DIGIT_RE.search(text)
        if m:
            return m.group(1)
        # Fall back to a Thai letter ก-ง surrounded by whitespace/punctuation.
        m = _MCQ_THAI_LETTER_RE.search(text)
        if m:
            return m.group(1)
        return ""
