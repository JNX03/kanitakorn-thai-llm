"""Format-extractor layer that normalizes raw model output into the exact
shape each benchmark's verifier expects.

Without this layer a model emitting "Let me solve this step by step ... so the
answer is 42" scores 0 against gold "42" even though it's correct. Each
benchmark has its own canonical answer surface — this module knows them all.

Public API:
    extract(family: str, raw: str, meta: dict | None = None) -> str
        Returns the cleaned prediction string the per-family judge expects.

The judges in tools/judges/ are wired to call `extract()` automatically as
their first step; existing tests still pass because gold answers are already
normalized.
"""

from __future__ import annotations

import re
from typing import Callable


# ---------- Math (AIME / MATH500) ----------

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_DOLLAR_BOXED_RE = re.compile(r"\$\\boxed\{([^{}]+)\}\$")
_FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer|the\s+answer\s+is|answer\s*[:=]|คำตอบคือ|คำตอบ\s*[:：])\s*([^\n.]{1,200})",
    flags=re.IGNORECASE,
)
_NUMERIC_TAIL_RE = re.compile(r"([-+]?\d+(?:\.\d+)?(?:/\d+)?)\s*$")


def extract_math(raw: str, meta: dict | None = None) -> str:
    """Extract the final numeric / symbolic answer.

    Priority order:
        1. last \\boxed{...} in the response (AIME / MATH convention)
        2. "the answer is X" / "คำตอบคือ X" phrase
        3. trailing numeric on the last non-blank line
    """
    if not raw:
        return ""
    boxed = _BOXED_RE.findall(raw) or _DOLLAR_BOXED_RE.findall(raw)
    if boxed:
        return boxed[-1].strip().rstrip(".")
    tail = raw[-1000:]  # answers are at the end
    m = _FINAL_ANSWER_RE.search(tail)
    if m:
        return m.group(1).strip().rstrip(".")
    # Fall back to the last numeric token on the last non-empty line.
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if not line:
            continue
        m = _NUMERIC_TAIL_RE.search(line)
        if m:
            return m.group(1)
        return line  # last non-empty line as last resort
    return raw.strip()


# ---------- MCQ / OpenThaiEval ----------

_MCQ_PARENS_RE = re.compile(r"\(([1-9]|[ก-ง])\)")
_MCQ_AFTER_KEYWORD_RE = re.compile(
    r"(?:ตอบ|คำตอบ|answer|choice|select)\s*(?:คือ|is|:|=)?\s*\(?([1-9]|[ก-ง])\)?",
    flags=re.IGNORECASE,
)
_MCQ_STANDALONE_DIGIT_RE = re.compile(r"\b([1-9])\b")

_XNLI_KEYWORDS = {
    "entailment": "entailment",
    "contradiction": "contradiction",
    "neutral": "neutral",
    "สรุปได้": "entailment",
    "ตามมา": "entailment",
    "เป็นไปตาม": "entailment",
    "ขัดแย้ง": "contradiction",
    "ไม่สอดคล้อง": "contradiction",
    "เป็นกลาง": "neutral",
    "ไม่เกี่ยว": "neutral",
    "ไม่แน่ชัด": "neutral",
}


def extract_openthaieval(raw: str, meta: dict | None = None) -> str:
    """Extract MCQ choice or XNLI class from a Thai academic-exam answer."""
    if not raw:
        return ""
    text = raw.strip()
    # XNLI first (case-insensitive).
    low = text.lower()
    for cue in ("entailment", "contradiction", "neutral"):
        if cue in low.split() or low.startswith(cue):
            return cue
    for cue, mapped in _XNLI_KEYWORDS.items():
        if cue in text:
            return mapped
    # MCQ parens (most explicit).
    m = _MCQ_PARENS_RE.search(text)
    if m:
        return f"({m.group(1)})"
    # MCQ after "ตอบ:" / "คำตอบ:" / "answer:".
    m = _MCQ_AFTER_KEYWORD_RE.search(text)
    if m:
        return f"({m.group(1)})"
    # Last-resort: first standalone digit at the start of the response.
    if text and text[0] in "123456789":
        return f"({text[0]})"
    m = _MCQ_STANDALONE_DIGIT_RE.search(text[:30])  # search first 30 chars
    if m:
        return f"({m.group(1)})"
    return text


# ---------- IFEval ----------


_IFEVAL_PREAMBLE_PATTERNS = [
    re.compile(r"^(?:certainly|sure|of course|okay|ok|alright|here'?s?\s+(?:is|are|the))\b.*?[:.](\s*|$)", re.IGNORECASE),
    re.compile(r"^(?:ครับ|ค่ะ|ได้ครับ|ได้ค่ะ)\s*[,，]?\s*", re.IGNORECASE),
    re.compile(r"^(?:นี่คือ|นี้คือ|นี่คือคำตอบ|นี้คือคำตอบ|คำตอบ)\s*[:：,，]?\s*", re.IGNORECASE),
    re.compile(r"^(?:answer|response|output)\s*[:=]\s*", re.IGNORECASE),
]

_IFEVAL_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n(.*?)\n```$", re.DOTALL)


def extract_ifeval(raw: str, meta: dict | None = None) -> str:
    """Strip a conversational preamble; if the response is fenced, unwrap it."""
    if not raw:
        return ""
    text = raw.strip()
    # If wholly wrapped in a code fence, unwrap.
    m = _IFEVAL_CODE_FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    # Strip preamble lines iteratively — handle stacked "ครับ\nนี่คือคำตอบ:".
    for _ in range(4):
        before = text
        for pat in _IFEVAL_PREAMBLE_PATTERNS:
            text = pat.sub("", text, count=1).strip()
        if text == before:
            break
    return text


# ---------- LiveCodeBench ----------


_LCB_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\n(.*?)\n```", re.DOTALL)
_LCB_BARE_CODE_HINT_RE = re.compile(r"^\s*(?:def |class |from |import |#|\"\"\"|''')", re.MULTILINE)


def extract_livecodebench(raw: str, meta: dict | None = None) -> str:
    """Pull the largest Python code block from a markdown response.

    Falls back to "looks like code" heuristics for fence-free responses.
    """
    if not raw:
        return ""
    blocks = _LCB_CODE_FENCE_RE.findall(raw)
    if blocks:
        # Take the longest block — usually the full solution.
        return max(blocks, key=len).strip()
    if _LCB_BARE_CODE_HINT_RE.search(raw):
        return raw.strip()
    return raw.strip()


# ---------- HotpotQA ----------


_HOTPOT_SOURCE_PATTERNS = [
    re.compile(r"https?://[^\s)\]]+"),
    re.compile(r"\[(?:source|แหล่ง|wiki|wikipedia)\s*[:：]\s*([^\]]+)\]", re.IGNORECASE),
]


def extract_hotpotqa(raw: str, meta: dict | None = None) -> dict:
    """Return {"answer": str, "sources": list[str]}.

    HotpotQA needs both the named-entity answer and the cited supporting
    sources (URLs). We extract the answer using the "answer is X" pattern and
    pull every URL or [source: X] mention as the source set.
    """
    if not raw:
        return {"answer": "", "sources": []}
    text = raw.strip()
    answer = ""
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        answer = m.group(1).strip().rstrip(".")
    else:
        # First short line (≤ 200 chars) before any source citation.
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(("http", "[source", "[wiki", "[แหล่ง")):
                break
            if len(line) <= 200:
                answer = line.split(":", 1)[-1].strip().rstrip(".")
                break
    sources: list[str] = []
    for pat in _HOTPOT_SOURCE_PATTERNS:
        sources.extend(pat.findall(text))
    return {"answer": answer, "sources": list(dict.fromkeys(sources))}  # dedupe


# ---------- MT-Bench ----------


def extract_mt_bench(raw: str, meta: dict | None = None) -> str:
    """MT-Bench has no canonical answer — the LLM-judge grades fluency etc.

    We only strip outer code fences and trailing system tokens.
    """
    if not raw:
        return ""
    text = raw.strip()
    m = _IFEVAL_CODE_FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    # Strip Qwen/Gemma chat-template residue.
    for tag in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        text = text.replace(tag, "")
    return text.strip()


# ---------- dispatcher ----------


_EXTRACTORS: dict[str, Callable] = {
    "aime24": extract_math,
    "aime25": extract_math,
    "math500": extract_math,
    "math500_th": extract_math,
    "aime_th": extract_math,
    "openthaieval": extract_openthaieval,
    "ifeval": extract_ifeval,
    "ifeval_ifbench": extract_ifeval,
    "livecodebench": extract_livecodebench,
    "livecodebench_th": extract_livecodebench,
    "hotpotqa": extract_hotpotqa,
    "hotpotqa_agentic": extract_hotpotqa,
    "mt_bench": extract_mt_bench,
}


def extract(family: str, raw: str, meta: dict | None = None):
    """Dispatch to the right extractor. Returns str for most families,
    dict {"answer", "sources"} for hotpotqa."""
    fn = _EXTRACTORS.get(family)
    if not fn:
        return raw.strip() if raw else ""
    return fn(raw, meta)


# ---------- self-test ----------


def _selftest() -> None:
    cases = [
        ("aime24", "Let me work through this... so the answer is \\boxed{42}.", "42"),
        ("math500", "After simplification, we get \\boxed{1/2}.", "1/2"),
        ("math500", "Solving: ... คำตอบคือ 16", "16"),
        ("openthaieval", "ดูจากบทความ คำตอบคือ (2) เพราะ...", "(2)"),
        ("openthaieval", "Answer: 3", "(3)"),
        ("openthaieval", "ตอบ entailment เพราะ...", "entailment"),
        ("ifeval", "Sure! Here's the response: - hello\n- world", "- hello\n- world"),
        ("ifeval", "ครับ นี่คือคำตอบ:\n- ข้อ: หนึ่ง", "- ข้อ: หนึ่ง"),
        ("livecodebench", "Sure. ```python\ndef f(x): return x*2\n```\nThat solves it.", "def f(x): return x*2"),
        ("hotpotqa", "The answer is Paris. Source: https://en.wikipedia.org/wiki/Paris", {"answer": "Paris", "sources": ["https://en.wikipedia.org/wiki/Paris"]}),
    ]
    for fam, raw, expected in cases:
        got = extract(fam, raw)
        ok = got == expected
        print(f"  [{'PASS' if ok else 'FAIL'}] {fam}: got={got!r} expected={expected!r}")


if __name__ == "__main__":
    _selftest()
