"""Math judge — sympy first, LLM tiebreaker.

For AIME24-TH, AIME25, MATH500-TH. Sympy handles the overwhelming majority of
cases (latex parsing, expression equivalence). The LLM judge fires only when
sympy throws or can't normalize (e.g. "the answer is 8" vs "8" plain text).

Used by:
    * tools/benchmark_eval.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_judge import GPT55Judge, default_judge  # noqa: E402
from answer_formatter import extract_math  # noqa: E402


_EQ_RUBRIC = (
    "Decide whether the candidate's final answer is mathematically equivalent "
    "to the gold answer. Strip prose, latex, and dollar signs; compare numbers "
    "and symbolic expressions only. Return JSON: "
    "{\"score\": 1 if equivalent else 0, \"rationale\": <short>}."
)


@dataclass
class MathScore:
    accuracy: float
    n: int
    n_sympy_resolved: int
    n_llm_fallback: int


def _extract_final_answer(text: str) -> str:
    """Heuristically pull the final answer out of a chain-of-thought response."""
    if not text:
        return ""
    # Common AIME / MATH formatting: \boxed{...}
    m = re.search(r"\\boxed\{([^{}]+)\}", text)
    if m:
        return m.group(1).strip()
    # Look for "answer is X" or "คำตอบคือ X" near the end
    last_500 = text[-500:]
    for pat in (r"answer\s*(?:is|=|:)\s*([^\n.]+)", r"คำตอบคือ\s*([^\n.]+)", r"คำตอบ:\s*([^\n.]+)"):
        m = re.search(pat, last_500, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".")
    # Last numeric/algebra token of the response
    m = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", last_500)
    if m:
        return m[-1]
    return text.strip().splitlines()[-1] if text.strip() else ""


class MathJudge:
    def __init__(self, judge: GPT55Judge | None = None) -> None:
        self.judge = judge or default_judge()

    def score_one(self, prediction: str, gold: str, question: str = "") -> tuple[int, str]:
        """Return (1 if equivalent else 0, method_used)."""
        # Use the canonical extractor first (handles \boxed{}, "the answer is",
        # Thai "คำตอบคือ", etc.); fall back to the local _extract_final_answer
        # for old call sites that pre-normalized.
        pred_ans = extract_math(prediction) or _extract_final_answer(prediction)
        gold_ans = extract_math(gold) or _extract_final_answer(gold) or gold.strip()
        # Fast path: textual equality after stripping.
        if self._strict_norm(pred_ans) == self._strict_norm(gold_ans):
            return 1, "exact_match"
        # Sympy path.
        try:
            from sympy import simplify, sympify
            from sympy.parsing.latex import parse_latex  # type: ignore
            try:
                a = parse_latex(pred_ans)
            except Exception:
                a = sympify(pred_ans, rational=True)
            try:
                b = parse_latex(gold_ans)
            except Exception:
                b = sympify(gold_ans, rational=True)
            if simplify(a - b) == 0:
                return 1, "sympy"
            return 0, "sympy"
        except Exception:
            pass
        # LLM tiebreaker.
        r = self.judge.score_pointwise(
            prompt=f"Question: {question}\nGold final answer: {gold_ans}",
            response=pred_ans,
            rubric=_EQ_RUBRIC,
        )
        return (1 if r.score >= 1 else 0), "llm"

    def aggregate(self, items: list[dict]) -> MathScore:
        correct = 0
        n_sympy = 0
        n_llm = 0
        for it in items:
            ok, method = self.score_one(it["prediction"], it["gold"], it.get("question", ""))
            if ok:
                correct += 1
            if method == "sympy":
                n_sympy += 1
            elif method == "llm":
                n_llm += 1
        n = len(items)
        return MathScore(
            accuracy=correct / n if n else 0.0,
            n=n,
            n_sympy_resolved=n_sympy,
            n_llm_fallback=n_llm,
        )

    @staticmethod
    def _strict_norm(s: str) -> str:
        return re.sub(r"\s+", "", s.replace("$", "").replace("\\,", "").lower())
