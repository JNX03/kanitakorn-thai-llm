"""IFEval / IFBench judge (Zhou et al. 2311.07911).

IFEval is dominated by *verifiable* constraints (line count, keyword frequency,
JSON schema, …). Those are scored deterministically — the LLM-judge handles
only the "soft" constraints regex can't capture (register, tone, semantic
fidelity).

Reports two metrics, matching the paper:
    strict_acc — fraction of prompts where EVERY constraint passes
    loose_acc  — average per-constraint pass rate

Used by:
    * tools/benchmark_eval.py — typhoon-ai/ifeval-th + the original ifeval
    * tools/teacher_loop_generator.py — verifier signal for the ifeval skill
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_judge import GPT55Judge, default_judge  # noqa: E402
from answer_formatter import extract_ifeval  # noqa: E402


_SOFT_RUBRIC = (
    "Decide whether the response satisfies the SOFT constraint below. "
    "A SOFT constraint cannot be checked by regex alone (e.g., register, tone, "
    "semantic fidelity, factual accuracy). Return JSON: "
    "{\"satisfied\": true|false, \"rationale\": <short>}."
)


@dataclass
class IFEvalScore:
    strict_acc: float
    loose_acc: float
    n_prompts: int
    per_constraint_pass_rate: dict[str, float] = field(default_factory=dict)


class IFEvalJudge:
    """Evaluates IFEval-style constraint sets.

    Each prompt carries a list of constraints. Deterministic constraints
    (everything we can codify) are run by `_check_deterministic`. Whatever
    remains is sent to the LLM judge in batch.
    """

    def __init__(self, judge: GPT55Judge | None = None) -> None:
        self.judge = judge or default_judge()

    def evaluate(self, response: str, constraints: list[dict]) -> tuple[bool, list[bool], list[str]]:
        """Return (strict_pass, per_constraint_pass, rationales)."""
        # Strip conversational preamble before checking constraints.
        response = extract_ifeval(response)
        per: list[bool] = []
        rationales: list[str] = []
        for constraint in constraints:
            ctype = constraint.get("type", "")
            result, rationale = self._check_one(response, ctype, constraint)
            per.append(result)
            rationales.append(rationale)
        return all(per), per, rationales

    def aggregate(self, items: list[dict]) -> IFEvalScore:
        """Items: [{response, constraints: [...]}]. Returns IFEval-paper metrics."""
        strict_passes = 0
        loose_sum = 0.0
        per_constraint: dict[str, list[int]] = {}
        for it in items:
            strict, per, _ = self.evaluate(it["response"], it["constraints"])
            if strict:
                strict_passes += 1
            loose_sum += (sum(per) / len(per)) if per else 0.0
            for c, p in zip(it["constraints"], per):
                per_constraint.setdefault(c.get("type", "unknown"), []).append(1 if p else 0)
        n = len(items)
        return IFEvalScore(
            strict_acc=strict_passes / n if n else 0.0,
            loose_acc=loose_sum / n if n else 0.0,
            n_prompts=n,
            per_constraint_pass_rate={k: sum(v) / len(v) for k, v in per_constraint.items()},
        )

    # ---- internals ----

    def _check_one(self, response: str, ctype: str, c: dict) -> tuple[bool, str]:
        if ctype == "keyword_frequency":
            word = c["keyword"]
            n = c["count"]
            return response.count(word) == n, f"count('{word}')={response.count(word)} vs {n}"
        if ctype == "exact_line_count":
            return len(response.splitlines()) == c["count"], f"lines={len(response.splitlines())}"
        if ctype == "starts_with":
            return all(line.startswith(c["prefix"]) for line in response.splitlines() if line.strip()), "prefix"
        if ctype == "ends_with":
            return response.rstrip().endswith(c["suffix"]), "suffix"
        if ctype == "forbid_substring":
            return c["forbidden"] not in response, f"forbidden present={c['forbidden'] in response}"
        if ctype == "no_arabic_digits":
            return not re.search(r"[0-9]", response), "arabic digits present" if re.search(r"[0-9]", response) else ""
        if ctype == "regex":
            pat = re.compile(c["pattern"], flags=re.DOTALL if c.get("dotall") else 0)
            return bool(pat.search(response)), "regex match"
        if ctype == "json_schema":
            try:
                from jsonschema import Draft202012Validator
                import json as _json
                _json.loads(response)
                Draft202012Validator(c["schema"]).validate(_json.loads(response))
                return True, "json valid"
            except Exception as e:
                return False, f"json invalid: {type(e).__name__}"
        if ctype == "language":
            target = c["language"]
            if target == "thai":
                return any("฀" <= ch <= "๿" for ch in response), "no thai chars"
            if target == "english":
                return bool(re.search(r"[A-Za-z]", response)) and not any(
                    "฀" <= ch <= "๿" for ch in response
                ), "thai chars present"
        # Soft constraint — defer to LLM judge.
        if ctype in {"register", "tone", "style", "semantic", "soft"}:
            prompt = (
                f"Soft constraint: {c.get('description', c.get('type', ''))}\n\n"
                f"Response to evaluate:\n{response}"
            )
            r = self.judge.score_pointwise(prompt=prompt, response=response, rubric=_SOFT_RUBRIC)
            satisfied = (r.score >= 7.0) or "satisfied" in r.rationale.lower() and "not satisfied" not in r.rationale.lower()
            return satisfied, r.rationale[:120]
        return False, f"unknown constraint type: {ctype}"
