"""LLM-as-judge core (Phase 1.1).

Wraps the OpenAI gpt-5.5-xhigh model (the same model that generates the
corpus). Per-family judge modules under tools/judges/ compose this class with
their own rubrics.

Self-preference mitigations:
    * temperature 0.0
    * position-swap for pairwise: run both A-vs-B and B-vs-A, report tie if
      the orderings disagree
    * rubric template explicitly forbids referencing model identity

Budget / reliability:
    * MAX_JUDGE_USD env var caps total spend per process
    * Per-call retry with exponential backoff on 429/5xx
    * On-disk cache keyed by sha256(prompt + response + rubric + mode) so
      reruns are free
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    from openai import OpenAI, BadRequestError, RateLimitError, APIError
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("openai SDK is required for llm_judge") from exc


_DEFAULT_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.5-xhigh")
_CACHE_DIR = Path(os.getenv("LLM_JUDGE_CACHE", str(Path(__file__).resolve().parent.parent / ".judge_cache")))
_MAX_USD = float(os.getenv("MAX_JUDGE_USD", "50.0"))
# Approximate $ per 1k input / output tokens for gpt-5.5-xhigh (placeholder; the
# real numbers come from OpenAI's pricing page — override via env if needed).
_USD_PER_1K_INPUT = float(os.getenv("JUDGE_USD_PER_1K_INPUT", "0.005"))
_USD_PER_1K_OUTPUT = float(os.getenv("JUDGE_USD_PER_1K_OUTPUT", "0.015"))


@dataclass
class JudgeSpend:
    """Running budget tracker shared across a single process."""

    total_usd: float = 0.0
    calls: int = 0
    cached_hits: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        cost = (input_tokens / 1000) * _USD_PER_1K_INPUT + (output_tokens / 1000) * _USD_PER_1K_OUTPUT
        self.total_usd += cost
        self.calls += 1
        if self.total_usd > _MAX_USD:
            raise RuntimeError(
                f"LLM judge budget exceeded: ${self.total_usd:.2f} > MAX_JUDGE_USD=${_MAX_USD:.2f}"
            )


@dataclass
class PointwiseResult:
    score: float
    rationale: str
    raw: dict = field(default_factory=dict)


@dataclass
class PairwiseResult:
    winner: Literal["A", "B", "tie"]
    rationale: str
    swap_consistent: bool
    raw: dict = field(default_factory=dict)


_POINTWISE_SYSTEM = (
    "You are an impartial benchmark judge. Score the assistant response on a "
    "1-10 scale using ONLY the rubric criteria provided. Do not reveal the "
    "model that produced the response. Do not consider model identity. "
    "Return strictly valid JSON: {\"score\": <int 1-10>, \"rationale\": <short string>}."
)

_PAIRWISE_SYSTEM = (
    "You are an impartial benchmark judge. Choose which response better "
    "satisfies the rubric. Do not reveal model identity. If responses are "
    "comparable, return tie. Return strictly valid JSON: "
    "{\"winner\": \"A\" | \"B\" | \"tie\", \"rationale\": <short string>}."
)


class GPT55Judge:
    """LLM-as-judge adapter for OpenAI gpt-5.5-xhigh.

    Stateless aside from on-disk cache and an injected `JudgeSpend` tracker.
    """

    def __init__(
        self,
        model: str | None = None,
        spend: JudgeSpend | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.model = model or _DEFAULT_MODEL
        self.spend = spend or JudgeSpend()
        self.cache_dir = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: OpenAI | None = None  # lazy — only init on first non-cached call

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    # ---- public ----

    def score_pointwise(self, prompt: str, response: str, rubric: str) -> PointwiseResult:
        cache_key = self._cache_key("pointwise", prompt, response, rubric)
        cached = self._cache_get(cache_key)
        if cached:
            return PointwiseResult(**cached)
        user = self._format_pointwise_user(prompt, response, rubric)
        raw = self._chat(_POINTWISE_SYSTEM, user)
        parsed = self._parse_json(raw["text"])
        result = PointwiseResult(
            score=float(parsed.get("score", 0)),
            rationale=str(parsed.get("rationale", "")),
            raw={"text": raw["text"], "usage": raw["usage"]},
        )
        self._cache_put(cache_key, {
            "score": result.score,
            "rationale": result.rationale,
            "raw": result.raw,
        })
        return result

    def score_pairwise(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        rubric: str,
        swap: bool = True,
    ) -> PairwiseResult:
        forward = self._pairwise_single(prompt, response_a, response_b, rubric, label_a="A", label_b="B")
        if not swap:
            return PairwiseResult(
                winner=forward["winner"], rationale=forward["rationale"], swap_consistent=True, raw=forward
            )
        # Swap positions: present B first, A second. Translate the winner label
        # back to the original A/B reference.
        reverse = self._pairwise_single(prompt, response_b, response_a, rubric, label_a="A", label_b="B")
        reverse_translated = {"A": "B", "B": "A", "tie": "tie"}[reverse["winner"]]
        if forward["winner"] == reverse_translated:
            winner = forward["winner"]
            consistent = True
        else:
            winner = "tie"  # position bias detected → safer to call it a tie
            consistent = False
        return PairwiseResult(
            winner=winner,
            rationale=forward["rationale"],
            swap_consistent=consistent,
            raw={"forward": forward, "reverse": reverse},
        )

    # ---- internals ----

    def _pairwise_single(
        self,
        prompt: str,
        first: str,
        second: str,
        rubric: str,
        *,
        label_a: str,
        label_b: str,
    ) -> dict:
        cache_key = self._cache_key("pairwise", prompt, first + "\n||\n" + second, rubric)
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        user = self._format_pairwise_user(prompt, first, second, rubric, label_a, label_b)
        raw = self._chat(_PAIRWISE_SYSTEM, user)
        parsed = self._parse_json(raw["text"])
        out = {
            "winner": str(parsed.get("winner", "tie")).strip(),
            "rationale": str(parsed.get("rationale", "")),
            "raw_text": raw["text"],
            "usage": raw["usage"],
        }
        if out["winner"] not in {"A", "B", "tie"}:
            out["winner"] = "tie"
        self._cache_put(cache_key, out)
        return out

    def _chat(self, system: str, user: str) -> dict:
        client = self._get_client()
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                )
                msg = resp.choices[0].message.content or ""
                usage = resp.usage
                input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                self.spend.add(input_tokens, output_tokens)
                return {"text": msg, "usage": {"input": input_tokens, "output": output_tokens}}
            except BadRequestError:
                raise  # don't retry malformed requests
            except (RateLimitError, APIError) as e:
                last_err = e
                sleep = (2 ** attempt) + random.random()
                time.sleep(min(sleep, 30.0))
        assert last_err is not None
        raise last_err

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Salvage attempt: find the first {...} block.
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            return {"score": 0, "winner": "tie", "rationale": f"unparseable: {text[:200]}"}

    @staticmethod
    def _format_pointwise_user(prompt: str, response: str, rubric: str) -> str:
        return (
            f"## Rubric\n{rubric.strip()}\n\n"
            f"## Original prompt\n{prompt.strip()}\n\n"
            f"## Assistant response\n{response.strip()}\n\n"
            f"Return JSON only."
        )

    @staticmethod
    def _format_pairwise_user(
        prompt: str, first: str, second: str, rubric: str, label_a: str, label_b: str
    ) -> str:
        return (
            f"## Rubric\n{rubric.strip()}\n\n"
            f"## Original prompt\n{prompt.strip()}\n\n"
            f"## Response {label_a}\n{first.strip()}\n\n"
            f"## Response {label_b}\n{second.strip()}\n\n"
            f"Return JSON only."
        )

    def _cache_key(self, mode: str, *parts: str) -> str:
        h = hashlib.sha256()
        h.update(self.model.encode("utf-8"))
        h.update(mode.encode("utf-8"))
        for p in parts:
            h.update(b"\x1e")
            h.update(p.encode("utf-8"))
        return h.hexdigest()

    def _cache_get(self, key: str) -> dict | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            self.spend.cached_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_put(self, key: str, payload: dict) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# Module-level singleton for convenience; per-family judges grab this.
_DEFAULT_JUDGE: GPT55Judge | None = None


def default_judge() -> GPT55Judge:
    global _DEFAULT_JUDGE
    if _DEFAULT_JUDGE is None:
        _DEFAULT_JUDGE = GPT55Judge()
    return _DEFAULT_JUDGE
