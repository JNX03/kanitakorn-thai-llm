"""MT-Bench judge (Zheng et al. 2023 — arxiv 2306.05685).

Two modes:
    score(prompt, response)                  — single-answer grading, 1-10
    compare(prompt, response_a, response_b)  — pairwise vs reference, with
                                                position-swap

Categories: writing, roleplay, reasoning, math, coding, extraction, stem,
humanities (per Zheng et al. Table 1). Each record carries a `category` field;
the judge picks a category-specific rubric.

Used by:
    * tools/benchmark_eval.py — score a trained model on ThaiLLM-Leaderboard/mt-bench-thai
    * tools/package_and_verify.py::verify_rubric (when --use-llm-judge)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_judge import GPT55Judge, PointwiseResult, PairwiseResult, default_judge  # noqa: E402
from answer_formatter import extract_mt_bench  # noqa: E402


# Rubrics keyed by MT-Bench category. The Thai-language version reuses the
# same axes since the protocol is language-agnostic; the judge handles Thai
# naturally because gpt-5.5-xhigh is multilingual.
_BASE_RUBRIC = (
    "Score the response on a 1-10 scale. Consider:\n"
    "1. Helpfulness — does it answer the prompt?\n"
    "2. Correctness — are facts and reasoning correct?\n"
    "3. Depth — is it thorough?\n"
    "4. Fluency in the target language (Thai or English as appropriate).\n"
    "5. Multi-turn coherence (if multi-turn).\n"
    "If the response is empty, refuses without justification, or is wildly off-topic, give 1-3.\n"
    "If correct and helpful but average, give 5-7. If excellent and complete, give 8-10."
)

_CATEGORY_HINTS = {
    "math": "For math, the FINAL ANSWER and the worked steps must both be correct.",
    "coding": "For coding, evaluate whether the code would actually compile and pass the stated tests.",
    "reasoning": "For reasoning, the conclusion must follow from the premises.",
    "writing": "For writing, evaluate creativity, structure, and prose quality.",
    "extraction": "For extraction, the output must exactly match the requested fields.",
    "stem": "For STEM, technical correctness outweighs fluency.",
    "humanities": "For humanities, evaluate accuracy of claims and breadth.",
    "roleplay": "For roleplay, evaluate persona consistency and the depth of in-character response.",
}


@dataclass
class MTBenchScore:
    overall: float
    by_category: dict[str, float]
    n: int
    cost_usd: float


class MTBenchJudge:
    """Run MT-Bench scoring against a corpus of (prompt, response, category) tuples."""

    def __init__(self, judge: GPT55Judge | None = None) -> None:
        self.judge = judge or default_judge()

    def score(self, prompt: str, response: str, category: str = "") -> PointwiseResult:
        rubric = _BASE_RUBRIC
        if category and category in _CATEGORY_HINTS:
            rubric = rubric + "\n\nCategory hint: " + _CATEGORY_HINTS[category]
        response = extract_mt_bench(response)
        return self.judge.score_pointwise(prompt, response, rubric)

    def compare(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        category: str = "",
    ) -> PairwiseResult:
        rubric = _BASE_RUBRIC
        if category and category in _CATEGORY_HINTS:
            rubric = rubric + "\n\nCategory hint: " + _CATEGORY_HINTS[category]
        return self.judge.score_pairwise(prompt, response_a, response_b, rubric, swap=True)

    def aggregate(
        self,
        triples: list[tuple[str, str, str]],
    ) -> MTBenchScore:
        """Score a list of (prompt, response, category) → return aggregate."""
        by_cat_scores: dict[str, list[float]] = {}
        for prompt, response, category in triples:
            r = self.score(prompt, response, category)
            by_cat_scores.setdefault(category or "uncategorized", []).append(r.score)
        by_category = {c: sum(s) / len(s) for c, s in by_cat_scores.items() if s}
        flat = [v for s in by_cat_scores.values() for v in s]
        return MTBenchScore(
            overall=(sum(flat) / len(flat)) if flat else 0.0,
            by_category=by_category,
            n=len(flat),
            cost_usd=self.judge.spend.total_usd,
        )
