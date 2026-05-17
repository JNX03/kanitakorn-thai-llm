"""Thai prosody verifier.

Used by Phase 0.5 (wired into verify_instruction for `klon_composition`,
`kap_composition`, `chant_composition`, `formal_register` task types) and by
Phase 2 teacher_loop_th's กลอน-teaching skill.

Public API:
    count_thai_syllables(text) -> int
    extract_tone_marks(text) -> list[tuple[int, str]]
    check_klon_4(text) -> ProsodyResult
    check_klon_8(text) -> ProsodyResult            # กลอนสุภาพ / กลอนแปด
    check_kap_yani_11(text) -> ProsodyResult       # กาพย์ยานี ๑๑
    check_register(text, target) -> ProsodyResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    from pythainlp.tokenize import subword_tokenize, syllable_tokenize
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pythainlp is required for thai_prosody_verifier") from exc


TONE_MARKS = {
    "่": "mai_ek",       # ่
    "้": "mai_tho",      # ้
    "๊": "mai_tri",      # ๊
    "๋": "mai_chattawa", # ๋
}

THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
ARABIC_DIGITS = "0123456789"

CASUAL_PARTICLES = ("ครับ", "ค่ะ", "นะคะ", "จ้ะ", "จ้า", "จ๊ะ", "นะจ๊ะ", "เนอะ", "เด้อ")
FORMAL_MARKERS = ("ขอบพระคุณ", "ขออนุญาต", "เรียน", "ตามที่", "ด้วยความเคารพ", "ในนามของ", "อนึ่ง")


@dataclass
class ProsodyResult:
    """Result of a prosody check.

    `ok` is the overall pass/fail.
    `failures` is a list of human-readable lines (Thai) describing each
    structural failure — used directly as the teacher's correction message in
    the teacher_loop_th pipeline.
    """

    ok: bool
    form: str
    failures: list[str] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)


def _strip_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def count_thai_syllables(text: str) -> int:
    """Best-effort Thai syllable count using pythainlp's subword tokenizer.

    Filters out pure punctuation/whitespace tokens and non-Thai-script
    segments so the count reflects spoken syllables rather than orthographic
    chunks. Latin words count as 1 syllable each (a coarse approximation, but
    matches how Thai prosodic critics treat foreign loanwords in modern กลอน).
    """
    tokens = subword_tokenize(text, engine="dict")
    count = 0
    for tok in tokens:
        s = tok.strip()
        if not s:
            continue
        # Skip pure punctuation / whitespace.
        if not re.search(r"[฀-๿A-Za-z]", s):
            continue
        # A Latin-only token counts as one syllable (approximation).
        if re.fullmatch(r"[A-Za-z]+", s):
            count += 1
            continue
        count += 1
    return count


def extract_tone_marks(text: str) -> list[tuple[int, str]]:
    """Return (index, tone-mark-name) for every tone mark in `text`."""
    return [(i, TONE_MARKS[ch]) for i, ch in enumerate(text) if ch in TONE_MARKS]


def _syllables_per_line(lines: list[str]) -> list[int]:
    return [count_thai_syllables(line) for line in lines]


def check_klon_4(text: str) -> ProsodyResult:
    """กลอน ๔ — each บท = 4 บาท × 4 พยางค์ (tolerance ±1 per line).

    Strict structural check: number of stanzas × 4 lines, each ≤ 5 syllables.
    """
    lines = _strip_lines(text)
    if not lines:
        return ProsodyResult(False, "klon_4", ["ข้อความว่าง"])
    if len(lines) % 4 != 0:
        return ProsodyResult(
            False,
            "klon_4",
            [f"กลอน ๔ ต้องมีจำนวนวรรคเป็นทวีคูณของ ๔ (พบ {len(lines)} วรรค)"],
        )
    syllables = _syllables_per_line(lines)
    failures: list[str] = []
    for i, n in enumerate(syllables, start=1):
        if not 3 <= n <= 5:
            failures.append(f"วรรคที่ {i} มี {n} พยางค์ (กลอน ๔ ต้องการ ๔ พยางค์, ยอม ±๑)")
    return ProsodyResult(
        ok=not failures,
        form="klon_4",
        failures=failures,
        measurements={"lines": len(lines), "syllables_per_line": syllables},
    )


def check_klon_8(text: str) -> ProsodyResult:
    """กลอนสุภาพ / กลอนแปด — 4 วรรคต่อบท, 7–9 พยางค์ต่อวรรค.

    Also checks the canonical rhyme pattern: คำสุดท้ายของวรรคที่ ๑ สัมผัสกับคำที่ ๓ หรือ ๕
    ของวรรคที่ ๒; คำสุดท้ายของวรรคที่ ๒ สัมผัสกับคำสุดท้ายของวรรคที่ ๓.
    Rhyme check uses a coarse final-vowel + tone hash since exact rhyme rules
    require dictionary lookups.
    """
    lines = _strip_lines(text)
    if not lines:
        return ProsodyResult(False, "klon_8", ["ข้อความว่าง"])
    if len(lines) % 4 != 0:
        return ProsodyResult(
            False,
            "klon_8",
            [f"กลอน ๘ ต้องมีจำนวนวรรคเป็นทวีคูณของ ๔ (พบ {len(lines)} วรรค)"],
        )
    syllables = _syllables_per_line(lines)
    failures: list[str] = []
    for i, n in enumerate(syllables, start=1):
        if not 7 <= n <= 9:
            failures.append(f"วรรคที่ {i} มี {n} พยางค์ (กลอน ๘ ต้องการ ๘ พยางค์, ยอม ±๑)")

    # Coarse rhyme check per stanza of 4 lines.
    for stanza_idx in range(0, len(lines), 4):
        stanza = lines[stanza_idx : stanza_idx + 4]
        last_words = [_last_thai_word(line) for line in stanza]
        # Line 1 last word ↔ line 2 internal (positions 3 or 5): we approximate
        # by checking ANY of line-2's middle tokens shares the final-vowel hash.
        if len(last_words) >= 2 and last_words[0] and last_words[1]:
            line2_tokens = _thai_tokens(stanza[1])
            if not any(_rhymes(last_words[0], tok) for tok in line2_tokens[1:-1]):
                failures.append(
                    f"บทที่ {stanza_idx // 4 + 1}: คำสุดท้ายวรรคที่ ๑ ('{last_words[0]}') ไม่สัมผัสกับวรรคที่ ๒"
                )
        if len(last_words) >= 3 and last_words[1] and last_words[2]:
            if not _rhymes(last_words[1], last_words[2]):
                failures.append(
                    f"บทที่ {stanza_idx // 4 + 1}: คำสุดท้ายวรรคที่ ๒ ('{last_words[1]}') ไม่สัมผัสกับคำสุดท้ายวรรคที่ ๓ ('{last_words[2]}')"
                )

    return ProsodyResult(
        ok=not failures,
        form="klon_8",
        failures=failures,
        measurements={"lines": len(lines), "syllables_per_line": syllables},
    )


def check_kap_yani_11(text: str) -> ProsodyResult:
    """กาพย์ยานี ๑๑ — 11 พยางค์ต่อบรรทัด, แบ่งเป็น 5+6.

    A บท has 2 lines totaling 11 syllables each (5 then 6). We accept 4+7 and
    5+6 patterns since modern กาพย์ยานี is flexible on the split.
    """
    lines = _strip_lines(text)
    if not lines:
        return ProsodyResult(False, "kap_yani_11", ["ข้อความว่าง"])
    if len(lines) % 2 != 0:
        return ProsodyResult(
            False,
            "kap_yani_11",
            [f"กาพย์ยานี ๑๑ ต้องมีจำนวนวรรคคู่ (พบ {len(lines)} วรรค)"],
        )
    syllables = _syllables_per_line(lines)
    failures: list[str] = []
    for i, n in enumerate(syllables, start=1):
        if not 10 <= n <= 12:
            failures.append(f"วรรคที่ {i} มี {n} พยางค์ (กาพย์ยานี ๑๑ ต้องการ ๑๑ พยางค์, ยอม ±๑)")
    return ProsodyResult(
        ok=not failures,
        form="kap_yani_11",
        failures=failures,
        measurements={"lines": len(lines), "syllables_per_line": syllables},
    )


def check_register(text: str, target: str = "formal") -> ProsodyResult:
    """Check the register (formal vs informal) of a Thai response.

    Formal: no casual particles; presence of at least one formal marker is a
    plus, not strictly required. Informal: at least one casual particle.
    """
    failures: list[str] = []
    casual_found = [p for p in CASUAL_PARTICLES if p in text]
    formal_found = [p for p in FORMAL_MARKERS if p in text]
    if target == "formal":
        if casual_found:
            failures.append(f"พบคำพูดทางการน้อย / ใช้คำลงท้ายไม่เป็นทางการ: {', '.join(casual_found)}")
    elif target == "informal":
        if not casual_found:
            failures.append("ไม่พบคำลงท้ายแบบไม่เป็นทางการ (เช่น ครับ/ค่ะ/นะคะ)")
    else:
        failures.append(f"ไม่รู้จัก target='{target}' (รองรับเฉพาะ 'formal' หรือ 'informal')")
    return ProsodyResult(
        ok=not failures,
        form=f"register_{target}",
        failures=failures,
        measurements={"casual_particles": casual_found, "formal_markers": formal_found},
    )


# ----- internal helpers -----


_THAI_WORD_RE = re.compile(r"[฀-๿]+")


def _thai_tokens(line: str) -> list[str]:
    return _THAI_WORD_RE.findall(line)


def _last_thai_word(line: str) -> str:
    toks = _thai_tokens(line)
    return toks[-1] if toks else ""


def _rhyme_key(word: str) -> str:
    """Strip leading consonants, keep final vowel + tone mark + final consonant.

    Coarse heuristic: take the last 2 Thai characters (or 1 if shorter). Two
    words rhyme if their rhyme keys match. This matches แม่ ก กา rhyme classes
    approximately and is sufficient for catching obviously-wrong rhymes; it
    will pass some non-rhymes (false negatives on rhyme failures), which is
    safer than rejecting correct rhymes the verifier doesn't know about.
    """
    if not word:
        return ""
    return word[-2:] if len(word) >= 2 else word[-1:]


def _rhymes(a: str, b: str) -> bool:
    ka, kb = _rhyme_key(a), _rhyme_key(b)
    return bool(ka) and ka == kb


# Optional self-test entry point.
def _selftest() -> None:
    good_klon_4 = "ฟ้าใสสดใส\nลมพัดเบาเบา\nหญ้าเขียวเขียว\nนกร้องเพลง"
    bad_klon_4 = "นี่คือบรรทัดที่ยาวเกินไปอย่างแน่นอนสำหรับกลอนสี่\nอีกบรรทัด\nบรรทัดสาม\nบรรทัดสี่"
    r1 = check_klon_4(good_klon_4)
    r2 = check_klon_4(bad_klon_4)
    print("good klon_4:", r1.ok, r1.failures, r1.measurements)
    print("bad klon_4:", r2.ok, r2.failures, r2.measurements)
    print("tone marks in 'น้ำใจ':", extract_tone_marks("น้ำใจ"))
    print(
        "register check (formal target, casual text):",
        check_register("ขอบคุณมากครับ", target="formal"),
    )


if __name__ == "__main__":
    _selftest()
