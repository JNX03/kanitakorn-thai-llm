"""HotpotQA judge — answer-equivalence + supporting-fact F1.

Two scores per question:
    answer_em  — 1 if model answer is semantically equivalent to gold; else 0
    sf_f1      — F1 between model-cited sources and gold supporting facts

Used by:
    * tools/benchmark_eval.py — multi-hop agentic harness
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_judge import GPT55Judge, default_judge  # noqa: E402
from answer_formatter import extract_hotpotqa  # noqa: E402


_EQ_RUBRIC = (
    "Decide whether the candidate answer matches the gold answer for this "
    "multi-hop question. Numbers/dates/named entities must match. Synonyms and "
    "paraphrases of the same entity are OK. Return JSON: "
    "{\"score\": 1 if equivalent else 0, \"rationale\": <short>}."
)


@dataclass
class HotpotQAScore:
    answer_em: float
    sf_f1: float
    joint_em_f1: float
    n: int


class HotpotQAJudge:
    def __init__(self, judge: GPT55Judge | None = None) -> None:
        self.judge = judge or default_judge()

    def score_one(
        self,
        prediction_text: str,
        prediction_sources: list[str],
        gold_answer: str,
        gold_supporting_facts: list[str],
        question: str = "",
    ) -> tuple[int, float]:
        """Return (answer_em, sf_f1) for a single question."""
        # If sources list is empty, attempt to extract URLs/cites from the
        # prediction text itself (the canonical extractor does both).
        extracted = extract_hotpotqa(prediction_text)
        if isinstance(extracted, dict):
            prediction_text = extracted.get("answer") or prediction_text
            if not prediction_sources:
                prediction_sources = extracted.get("sources", [])
        if self._fast_match(prediction_text, gold_answer):
            answer_em = 1
        else:
            r = self.judge.score_pointwise(
                prompt=f"Question: {question}\nGold answer: {gold_answer}",
                response=prediction_text,
                rubric=_EQ_RUBRIC,
            )
            answer_em = int(r.score >= 1)
        sf_f1 = self._set_f1(prediction_sources, gold_supporting_facts)
        return answer_em, sf_f1

    def aggregate(self, items: list[dict]) -> HotpotQAScore:
        em_sum = 0
        f1_sum = 0.0
        joint = 0.0
        for it in items:
            em, f1 = self.score_one(
                it["prediction_text"],
                it.get("prediction_sources", []),
                it["gold_answer"],
                it.get("gold_supporting_facts", []),
                it.get("question", ""),
            )
            em_sum += em
            f1_sum += f1
            joint += em * f1
        n = len(items)
        return HotpotQAScore(
            answer_em=em_sum / n if n else 0.0,
            sf_f1=f1_sum / n if n else 0.0,
            joint_em_f1=joint / n if n else 0.0,
            n=n,
        )

    @staticmethod
    def _fast_match(prediction: str, gold: str) -> bool:
        norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
        return norm(prediction) == norm(gold)

    @staticmethod
    def _set_f1(predicted: list[str], gold: list[str]) -> float:
        if not gold and not predicted:
            return 1.0
        if not gold or not predicted:
            return 0.0
        ps = {str(x).strip() for x in predicted}
        gs = {str(x).strip() for x in gold}
        if not ps or not gs:
            return 0.0
        tp = len(ps & gs)
        if tp == 0:
            return 0.0
        precision = tp / len(ps)
        recall = tp / len(gs)
        return 2 * precision * recall / (precision + recall)
