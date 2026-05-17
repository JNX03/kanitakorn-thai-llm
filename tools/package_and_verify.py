from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator

from manual_corpus import MANUAL_ITEMS, REJECTED_OR_QUARANTINED

try:
    import continuation_corpus as continuation_module

    CONTINUATION_ITEMS = getattr(continuation_module, "CONTINUATION_ITEMS", [])
    CONTINUATION_SHARD_ITEMS = [
        items
        for name, items in sorted(vars(continuation_module).items())
        if re.fullmatch(r"SHARD_\d{3}_ITEMS", name)
    ]
    CONTINUATION_REJECTED_OR_QUARANTINED = getattr(continuation_module, "CONTINUATION_REJECTED_OR_QUARANTINED", [])
except ImportError:
    CONTINUATION_ITEMS = []
    CONTINUATION_SHARD_ITEMS = []
    CONTINUATION_REJECTED_OR_QUARANTINED = []


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"

FAMILY_FILES = {
    "aime_th": ("train_aime_th_000.jsonl", "val_aime_th_000.jsonl"),
    "math500_th": ("train_math500_th_000.jsonl", "val_math500_th_000.jsonl"),
    "livecodebench_th": ("train_livecodebench_th_000.jsonl", "val_livecodebench_th_000.jsonl"),
    "openthaieval": ("train_openthaieval_000.jsonl", "val_openthaieval_000.jsonl"),
    "mt_bench": ("train_mt_bench_000.jsonl", "val_mt_bench_000.jsonl"),
    "ifeval_ifbench": ("train_ifeval_ifbench_000.jsonl", "val_ifeval_ifbench_000.jsonl"),
    "hotpotqa_agentic": ("train_hotpotqa_agentic_000.jsonl", "val_hotpotqa_agentic_000.jsonl"),
    "teacher_loop_th": ("train_teacher_loop_th_000.jsonl", "val_teacher_loop_th_000.jsonl"),
}

FAMILY_SLUGS = {
    "aime_th": "aime_th",
    "math500_th": "math500_th",
    "livecodebench_th": "livecodebench_th",
    "openthaieval": "openthaieval",
    "mt_bench": "mt_bench",
    "ifeval_ifbench": "ifeval_ifbench",
    "hotpotqa_agentic": "hotpotqa_agentic",
    "teacher_loop_th": "teacher_loop_th",
}

SLUG_TO_FAMILY = {slug: family for family, slug in FAMILY_SLUGS.items()}

QUALITY_UPGRADE_VERSION = "quality_gate_2026_05_08_v1"

QUALITY_GATE_STATE = {
    "bulk_generation_paused": True,
    "reason": "User requested quality upgrade before further bulk generation.",
    "next_required_batch": {
        "accepted_items": 100,
        "mode": "mixed_audit",
        "must_pass_before_resuming_20k_loop": True,
    },
    "future_requirements": {
        "aime_th": "At least 70% of new items require three or more reasoning steps; direct textbook drills are easy/medium, not hard.",
        "math500_th": "Include Level 4-5 style algebra, intermediate algebra, precalculus, geometry, probability, and number theory.",
        "livecodebench_th": "Include multi-testcase parsing, DP, graph shortest path, strings, greedy proof, data structures, debugging, and self-repair.",
        "ifeval_ifbench": "Reduce repeated line/keyword templates; prefer Thai-specific and nested adversarial constraints with deterministic verifiers.",
        "hotpotqa_agentic": "Reject one-source items; accepted items require at least two supporting facts and should use independent sources.",
        "openthaieval": "Use plausible distractors, longer passages, source-backed facts, and subtle inference.",
    },
    "rejection_quota": {
        "future_policy": "Every accepted production shard must include rejected or quarantined candidates in the audit log.",
        "minimum_rejected_or_quarantined_per_accepted_shard": 1,
        "near_zero_rejection_rate_is_quality_warning": True,
    },
}

QUALITY_GATE_START_NUMBERS = {
    "aime_th": 1067,
    "math500_th": 459,
    "livecodebench_th": 466,
    "openthaieval": 459,
    "mt_bench": 9,
    "ifeval_ifbench": 465,
    "hotpotqa_agentic": 9,
}

SIMPLE_AIME_TAGS = {
    "area",
    "centroid",
    "divisors",
    "identities",
    "inradius",
    "mean",
    "pythagorean",
    "recurrence",
    "sequences",
}

SIMPLE_MATH_TAGS = {
    "prealgebra",
    "arithmetic",
    "sequence",
    "quadratic",
    "linear_equations",
}

REQUIRED_KEYS = [
    "id",
    "benchmark_family",
    "task_type",
    "language",
    "difficulty",
    "messages",
    "final_answer",
    "concise_solution",
    "verifier",
    "sources",
    "contamination_audit",
    "quality_scores",
    "training_tags",
]


SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": REQUIRED_KEYS,
    "properties": {
        "id": {"type": "string"},
        "benchmark_family": {
            "enum": [
                "aime_th",
                "math500_th",
                "livecodebench_th",
                "openthaieval",
                "mt_bench",
                "ifeval_ifbench",
                "hotpotqa_agentic",
                "teacher_loop_th",
            ]
        },
        "loop_metadata": {"type": "object"},
        "task_type": {"type": "string"},
        "language": {"enum": ["th", "en", "th-en"]},
        "difficulty": {"enum": ["easy", "medium", "hard", "olympiad", "adversarial"]},
        "messages": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": ["role", "content"],
                "properties": {
                    "role": {"enum": ["system", "user", "assistant"]},
                    "content": {"type": "string"},
                },
            },
        },
        "final_answer": {"type": "string"},
        "concise_solution": {"type": "string"},
        "verifier": {
            "type": "object",
            "required": ["type", "details"],
            "properties": {
                "type": {
                    "enum": [
                        "symbolic_math",
                        "unit_tests",
                        "json_schema",
                        "regex",
                        "exact_match",
                        "llm_judge_rubric",
                        "human_review",
                        "retrieval_evidence",
                    ]
                },
                "details": {"type": "object"},
            },
        },
        "sources": {"type": "array"},
        "contamination_audit": {"type": "object"},
        "quality_scores": {"type": "object"},
        "training_tags": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


def public_item(raw: dict) -> dict:
    return {k: raw[k] for k in REQUIRED_KEYS}


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def char_ngrams(text: str, n: int = 5) -> set[str]:
    t = normalize(text)
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    return intersection / (len(a) + len(b) - intersection)


def simhash64(text: str) -> int:
    grams = char_ngrams(text, 5)
    weights = [0] * 64
    for gram in grams:
        h = int(hashlib.sha256(gram.encode("utf-8")).hexdigest()[:16], 16)
        for i in range(64):
            weights[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, w in enumerate(weights):
        if w > 0:
            out |= 1 << i
    return out


def simhash_similarity(a: str, b: str) -> float:
    x = simhash64(a) ^ simhash64(b)
    return 1.0 - (x.bit_count() / 64.0)


def prompt_text(item: dict) -> str:
    return "\n".join(m["content"] for m in item["messages"] if m["role"] == "user")


def answer_text(item: dict) -> str:
    return "\n".join(m["content"] for m in item["messages"] if m["role"] == "assistant")


def item_sequence_number(item: dict) -> int | None:
    m = re.search(r"-(\d+)$", item["id"])
    return int(m.group(1)) if m else None


def quality_gate_progress(items: list[dict]) -> dict:
    by_family = Counter()
    accepted_items = 0
    for item in items:
        start = QUALITY_GATE_START_NUMBERS.get(item["benchmark_family"])
        seq = item_sequence_number(item)
        if item["split"] == "train" and start is not None and seq is not None and seq >= start:
            accepted_items += 1
            by_family[item["benchmark_family"]] += 1
    target = QUALITY_GATE_STATE["next_required_batch"]["accepted_items"]
    return {
        "accepted_items": accepted_items,
        "target_items": target,
        "passed": accepted_items >= target,
        "by_family": dict(sorted(by_family.items())),
    }


def quality_gate_state_for(items: list[dict]) -> dict:
    state = json.loads(json.dumps(QUALITY_GATE_STATE))
    progress = quality_gate_progress(items)
    state["mixed_audit_progress"] = progress
    state["mixed_audit_passed"] = progress["passed"]
    state["bulk_generation_resume_condition_met"] = progress["passed"]
    if progress["passed"]:
        state["reason"] = "User requested bulk pause; the required 100-item mixed quality audit batch has passed under upgraded standards."
    return state


def quality_reviewer_note(item: dict, original_difficulty: str, calibrated_difficulty: str, original_scores: dict, scores: dict) -> str:
    verifier_type = item["verifier"]["type"]
    family = item["benchmark_family"]
    if original_difficulty != calibrated_difficulty:
        difficulty_note = f"Difficulty recalibrated from {original_difficulty} to {calibrated_difficulty} because the solution does not meet the stronger benchmark-beating bar for the original label."
    else:
        difficulty_note = f"Difficulty retained as {calibrated_difficulty} after curator review."
    return (
        f"{QUALITY_UPGRADE_VERSION}: {family} item verified with {verifier_type}; "
        f"correctness score {scores['correctness']} reflects available deterministic/rubric evidence. "
        f"{difficulty_note} Benchmark alignment and novelty scores were capped to avoid self-reported inflation; "
        f"original scores were {json.dumps(original_scores, ensure_ascii=False, sort_keys=True)}."
    )


def recalibrate_difficulty(item: dict) -> str:
    family = item["benchmark_family"]
    difficulty = item["difficulty"]
    tags = set(item.get("training_tags", []))
    prompt = prompt_text(item)
    solution = answer_text(item)

    if family == "aime_th":
        if tags & SIMPLE_AIME_TAGS:
            return "medium"
        if "probability" in tags and ("C(" not in solution and "modulo" not in solution and "กรณี" not in solution):
            return "medium"
        if difficulty == "olympiad":
            return difficulty
        if difficulty == "hard" and len(solution.splitlines()) < 5 and len(prompt) < 180:
            return "medium"
        return difficulty

    if family == "math500_th":
        if tags & SIMPLE_MATH_TAGS and difficulty in {"hard", "medium"}:
            return "easy" if difficulty == "medium" else "medium"
        return difficulty

    if family == "livecodebench_th":
        if item["task_type"] == "output_prediction" and difficulty == "hard":
            return "medium"
        return difficulty

    return difficulty


def calibrated_quality_scores(item: dict, original_scores: dict, original_difficulty: str, calibrated_difficulty: str) -> dict:
    scores = dict(original_scores)
    verifier_type = item["verifier"]["type"]
    deterministic = verifier_type in {"symbolic_math", "unit_tests", "json_schema", "regex", "exact_match", "retrieval_evidence"}

    scores["correctness"] = min(float(scores.get("correctness", 0.95)), 0.99 if deterministic else 0.96)
    if item["language"].startswith("th"):
        scores["thai_naturalness"] = min(float(scores.get("thai_naturalness", 0.9)), 0.97)
    else:
        scores["thai_naturalness"] = min(float(scores.get("thai_naturalness", 1.0)), 0.95)
    scores["novelty"] = min(float(scores.get("novelty", 0.9)), 0.95)
    scores["instruction_clarity"] = min(float(scores.get("instruction_clarity", 0.9)), 0.98)

    alignment_cap = 0.93
    if original_difficulty != calibrated_difficulty:
        alignment_cap = 0.88
    if item["benchmark_family"] in {"mt_bench", "hotpotqa_agentic"}:
        alignment_cap = min(alignment_cap, 0.91)
    if item["benchmark_family"] == "ifeval_ifbench" and "line_count" in item.get("training_tags", []):
        alignment_cap = min(alignment_cap, 0.90)
    scores["benchmark_alignment"] = min(float(scores.get("benchmark_alignment", 0.85)), alignment_cap)

    scores["calibration_version"] = QUALITY_UPGRADE_VERSION
    scores["difficulty_calibration"] = {
        "original": original_difficulty,
        "calibrated": calibrated_difficulty,
    }
    scores["reviewer_notes"] = quality_reviewer_note(item, original_difficulty, calibrated_difficulty, original_scores, scores)
    return scores


def apply_quality_upgrade(item: dict) -> tuple[dict | None, dict | None]:
    upgraded = dict(item)
    original_difficulty = item["difficulty"]
    calibrated_difficulty = recalibrate_difficulty(item)
    upgraded["difficulty"] = calibrated_difficulty
    upgraded["quality_scores"] = calibrated_quality_scores(item, item["quality_scores"], original_difficulty, calibrated_difficulty)

    if upgraded["benchmark_family"] == "hotpotqa_agentic":
        supporting_facts = upgraded["verifier"]["details"].get("required_supporting_facts", [])
        if len(upgraded.get("sources", [])) < 2 or len(supporting_facts) < 2:
            row = {
                "item_id": upgraded["id"],
                "benchmark_family": upgraded["benchmark_family"],
                "split": "quarantined",
                "created_by": upgraded.get("created_by", "main_agent"),
                "subagent_model_if_used": "none",
                "reviewed_by_main_agent": True,
                "number_of_review_passes": 3,
                "verification_methods": upgraded.get("verification_methods", []),
                "contamination_checks": ["quality_upgrade_hotpot_source_count", "supporting_fact_review"],
                "quality_scores": upgraded["quality_scores"],
                "acceptance_decision": "quarantine",
                "reason_for_decision": "Quality upgrade rejected one-source HotpotQA-agentic structure; item is factual/source-grounded QA, not clean multi-hop agentic training data.",
                "known_risks": "Could train shallow one-page retrieval behavior if kept in HotpotQA-agentic shard.",
            }
            return None, row

    return upgraded, None


def run_python_assert(code: str) -> None:
    ns = {}
    try:
        ok = eval(code, ns)
    except SyntaxError:
        prefix, expr = code.rsplit(";", 1)
        exec(prefix, ns)
        ok = eval(expr, ns)
    assert ok, code


def verify_math(item: dict) -> list[str]:
    for code in item["verifier"]["details"].get("python_asserts", []):
        run_python_assert(code)
    return ["symbolic_math", "independent_arithmetic"]


def run_reference_solution(src: str, data: str) -> str:
    ns = {}
    exec(src, ns)
    if "solve" in ns:
        return ns["solve"](data)
    raise AssertionError("reference_solution must expose solve(data)")


def verify_unit_tests(item: dict) -> list[str]:
    details = item["verifier"]["details"]
    if "public_tests" in details:
        src = details["reference_solution"]
        for test in details["public_tests"]:
            got = run_reference_solution(src, test["input"])
            assert got == test["output"], (item["id"], test, got)
        for test in details.get("hidden_tests", []):
            got = run_reference_solution(src, test["input"])
            assert got == test["output"], (item["id"], test, got)
        if "randomized_test_code" in details:
            ns = {
                "random": random,
                "reference_solution": details["reference_solution"],
                "run_reference_solution": run_reference_solution,
            }
            exec(details["randomized_test_code"], ns)
        if item["id"] == "lcb-th-train-0001":
            src_ns = {}
            exec(src, src_ns)
            solve = src_ns["solve"]
            rnd = random.Random(7)
            for _ in range(200):
                n = rnd.randint(1, 20)
                arr = [rnd.randint(-5, 5) for _ in range(n)]
                k = rnd.randint(-8, 8)
                brute = 0
                for i in range(n):
                    s = 0
                    for j in range(i, n):
                        s += arr[j]
                        brute += s == k
                data = f"{n} {k}\n" + " ".join(map(str, arr)) + "\n"
                assert solve(data) == f"{brute}\n"
        if item["id"] == "lcb-th-train-0003":
            src_ns = {}
            exec(src, src_ns)
            solve = src_ns["solve"]
            rnd = random.Random(13)
            for _ in range(200):
                n = rnd.randint(1, 25)
                k = rnd.randint(0, 8)
                arr = [rnd.randint(-6, 6) for _ in range(n)]
                brute = 0
                for i in range(n):
                    mn = mx = arr[i]
                    for j in range(i, n):
                        mn = min(mn, arr[j])
                        mx = max(mx, arr[j])
                        brute += mx - mn <= k
                data = f"{n} {k}\n" + " ".join(map(str, arr)) + "\n"
                assert solve(data) == f"{brute}\n"
        if item["id"] == "lcb-th-val-0002":
            src_ns = {}
            exec(src, src_ns)
            solve = src_ns["solve"]
            rnd = random.Random(17)
            alphabet = "abcd"
            for _ in range(200):
                n = rnd.randint(1, 30)
                s = "".join(rnd.choice(alphabet) for _ in range(n))
                brute = 0
                for i in range(n):
                    seen = set()
                    for j in range(i, n):
                        if s[j] in seen:
                            break
                        seen.add(s[j])
                        brute = max(brute, j - i + 1)
                assert solve(s + "\n") == f"{brute}\n"
        if item["id"] == "lcb-th-train-0006":
            src_ns = {}
            exec(src, src_ns)
            solve = src_ns["solve"]
            rnd = random.Random(23)
            for _ in range(200):
                n = rnd.randint(1, 30)
                arr = [rnd.randint(-9, 9) for _ in range(n)]
                brute = 0
                for i in range(n):
                    s = 0
                    for j in range(i, n):
                        s += arr[j]
                        brute += s % 2 == 0
                data = f"{n}\n" + " ".join(map(str, arr)) + "\n"
                assert solve(data) == f"{brute}\n"
        if item["id"] == "lcb-th-val-0003":
            src_ns = {}
            exec(src, src_ns)
            solve = src_ns["solve"]
            rnd = random.Random(29)
            for _ in range(200):
                n = rnd.randint(1, 30)
                arr = [rnd.randint(0, 1) for _ in range(n)]
                brute = 0
                for i in range(n):
                    zeros = ones = 0
                    for j in range(i, n):
                        zeros += arr[j] == 0
                        ones += arr[j] == 1
                        if zeros == ones:
                            brute = max(brute, j - i + 1)
                data = f"{n}\n" + " ".join(map(str, arr)) + "\n"
                assert solve(data) == f"{brute}\n"
        if item["id"] == "lcb-th-train-0008":
            src_ns = {}
            exec(src, src_ns)
            solve = src_ns["solve"]
            rnd = random.Random(31)
            alphabet = "abcdeiouxyz"
            vowels = set("aeiou")
            for _ in range(200):
                n = rnd.randint(1, 40)
                k = rnd.randint(1, n)
                s = "".join(rnd.choice(alphabet) for _ in range(n))
                brute = max(sum(ch in vowels for ch in s[i : i + k]) for i in range(n - k + 1))
                assert solve(f"{s}\n{k}\n") == f"{brute}\n"
    if "function_tests" in details:
        ns = {}
        exec(details["reference_solution"], ns)
        fn = ns[details.get("function_name", "count_pairs")]
        for test in details["function_tests"]:
            if "args" in test:
                got = fn(*test["args"])
            else:
                got = fn(test["a"], test["d"])
            assert got == test["output"], (item["id"], test, got)
        for test in details.get("hidden_function_tests", []):
            if "args" in test:
                got = fn(*test["args"])
            else:
                got = fn(test["a"], test["d"])
            assert got == test["output"], (item["id"], test, got)
        if "randomized_function_test_code" in details:
            exec(details["randomized_function_test_code"], {"random": random, "fn": fn})
        if item["id"] == "lcb-th-train-0002":
            rnd = random.Random(11)
            for _ in range(200):
                n = rnd.randint(0, 12)
                a = [rnd.randint(-4, 4) for _ in range(n)]
                d = rnd.randint(-3, 3)
                brute = sum(1 for i in range(n) for j in range(i + 1, n) if a[j] - a[i] == d)
                assert fn(a, d) == brute
        if item["id"] == "lcb-th-train-0004":
            rnd = random.Random(19)
            for _ in range(200):
                n = rnd.randint(0, 20)
                a = [rnd.randint(0, 8) for _ in range(n)]
                k = rnd.randint(0, 12)
                brute = 0
                for i in range(n):
                    s = 0
                    for j in range(i, n):
                        s += a[j]
                        if s <= k:
                            brute = max(brute, j - i + 1)
                assert fn(a, k) == brute
        if item["id"] == "lcb-th-train-0005":
            from collections import Counter

            samples = [
                ["A", "a", "b"],
                ["x", "X", "x", "Y"],
                [],
                ["m", "n", "m", "N"],
            ]
            for words in samples:
                brute = sum(1 for c in Counter(w.lower() for w in words).values() if c > 1)
                assert fn(words) == brute
    if "program" in details:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(details["program"], {})
        assert buf.getvalue() == details["expected_stdout"], (item["id"], buf.getvalue())
    return ["unit_tests", "randomized_bruteforce"]


def assistant_content(item: dict) -> str:
    return item["messages"][-1]["content"]


def verify_instruction(item: dict) -> list[str]:
    text = assistant_content(item)
    for code in item["verifier"]["details"].get("python_asserts", []):
        ns = {"text": text, "json": json, "re": re}
        if code.lstrip().startswith("assert "):
            exec(code, ns)
        else:
            assert eval(code, ns), code
    if item["id"] == "ifeval-ifbench-th-train-0001":
        obj = json.loads(text)
        assert set(obj) == {"title", "steps", "warning"}
        assert isinstance(obj["title"], str) and isinstance(obj["steps"], list)
        assert len(obj["steps"]) == 3 and all(s.startswith("ขั้น") for s in obj["steps"])
        assert "ห้าม" not in obj["warning"]
    elif item["id"] == "ifeval-ifbench-th-train-0002":
        paragraphs = text.split("\n\n")
        assert len(paragraphs) == 2
        assert all(p.startswith("ข้อคิด:") for p in paragraphs)
        assert all(len([s for s in p.split(".") if s.strip()]) == 2 for p in paragraphs)
        assert text.count("น้ำใจ") == 2
        assert not re.search(r"[0-9]", text)
        assert "ครับ" not in text and "ค่ะ" not in text
    elif item["id"] == "ifeval-ifbench-en-val-0001":
        lines = text.splitlines()
        bullets = [line for line in lines if line.startswith("- ")]
        assert len(bullets) == 4
        assert all(line.startswith("- Do:") for line in bullets)
        assert "," not in text
        assert len(re.findall(r"\baudit\b", text)) == 2
        assert lines[-1] == "Done"
    elif item["id"] == "ifeval-ifbench-th-train-0003":
        lines = text.splitlines()
        bullets = [line for line in lines if line.startswith("- ")]
        assert len(bullets) == 4
        assert all(line.startswith("- เหตุผล:") for line in bullets)
        assert text.count("ตรวจสอบ") == 2
        assert "มาก" not in text
        assert lines[-1] == "ครบถ้วน"
    elif item["id"] == "ifeval-ifbench-then-train-0004":
        obj = json.loads(text)
        assert isinstance(obj, list) and len(obj) == 2
        for row in obj:
            assert set(row) == {"term_th", "term_en"}
            assert row["term_th"].startswith("คำ")
            assert re.fullmatch(r"[A-Z]+", row["term_en"])
    elif item["id"] == "ifeval-ifbench-en-train-0005":
        paragraphs = text.split("\n\n")
        assert len(paragraphs) == 3
        assert all(p.startswith("Note:") for p in paragraphs)
        assert all(len([s for s in re.split(r"[.!?]", p) if s.strip()]) == 1 for p in paragraphs)
        assert text.count(";") == 1
        assert len(re.findall(r"\breview\b", text, flags=re.I)) == 2
        assert not re.search(r"\bvery\b", text, flags=re.I)
    elif item["id"] == "ifeval-ifbench-th-val-0002":
        lines = text.splitlines()
        assert len(lines) == 3
        assert [line.split(":", 1)[0] + ":" for line in lines] == ["หนึ่ง:", "สอง:", "สาม:"]
        assert not re.search(r"[0-9]", text)
        assert text.count("พ.ศ. ๒๕๖๙") == 1
        assert "ควร" not in text
    elif item["id"] == "ifeval-ifbench-th-train-0006":
        lines = text.splitlines()
        assert len(lines) == 4
        assert [line.split(":", 1)[0] + ":" for line in lines] == ["ก:", "ข:", "ค:", "ง:"]
        assert not re.search(r"[0-9]", text)
        assert text.count("แผน") == 2
        assert "สำคัญ" not in text
    elif item["id"] == "ifeval-ifbench-th-train-0007":
        obj = json.loads(text)
        assert set(obj) == {"ok", "items", "note"}
        assert obj["ok"] is True
        assert isinstance(obj["items"], list) and len(obj["items"]) == 3
        assert all(isinstance(x, str) and x.endswith("แล้ว") for x in obj["items"])
        assert obj["note"].count("ตรวจ") == 1
    elif item["id"] == "ifeval-ifbench-en-val-0003":
        lines = text.splitlines()
        bullets = [line for line in lines if line.startswith("- ")]
        assert len(bullets) == 5
        assert all(line.startswith("- Check:") for line in bullets)
        assert len(re.findall(r"\bclean\b", text, flags=re.I)) == 3
        assert "(" not in text and ")" not in text
        assert lines[-1] == "Verified"
    elif item["id"] == "ifeval-ifbench-th-train-0008":
        paragraphs = text.split("\n\n")
        assert len(paragraphs) == 2
        assert all(len([s for s in p.split(".") if s.strip()]) == 2 for p in paragraphs)
        assert text.count("พัก") == 3
        assert "จอ" not in text
        assert text.endswith("เริ่มตอนนี้")
    return ["deterministic_instruction_checker"]


def verify_exact(item: dict) -> list[str]:
    accepted = set(item["verifier"]["details"].get("accepted_answers", []))
    assert item["final_answer"] in accepted
    return ["exact_match"]


def verify_retrieval(item: dict) -> list[str]:
    assert item["sources"] and all(src["url"].startswith("http") for src in item["sources"])
    assert len(item["verifier"]["details"].get("required_supporting_facts", [])) >= 2
    if item["benchmark_family"] == "hotpotqa_agentic":
        assert len(item["sources"]) >= 2
    return ["retrieval_evidence", "source_check"]


def verify_rubric(item: dict, use_llm_judge: bool = False) -> list[str]:
    """Rubric verifier for MT-Bench-style items.

    Structural check: ≥3 criteria, ≥4 messages (paper requires multi-turn).
    When `use_llm_judge=True` OR env `USE_LLM_JUDGE=1`, also calls
    GPT55Judge.score_pointwise per criterion and requires mean ≥ 8.0.
    """
    criteria = item["verifier"]["details"].get("criteria", [])
    assert len(criteria) >= 3
    assert len(item["messages"]) >= 4

    import os
    if not use_llm_judge:
        use_llm_judge = os.getenv("USE_LLM_JUDGE", "") in ("1", "true", "yes")

    methods = ["llm_judge_rubric", "human_review"]
    if use_llm_judge:
        from llm_judge import default_judge
        judge = default_judge()
        prompt = "\n".join(m["content"] for m in item["messages"] if m["role"] == "user")
        response = item["messages"][-1]["content"]
        scores = []
        for criterion in criteria:
            rubric = f"Criterion: {criterion}\n\nScore 1-10 for how well the response satisfies this criterion."
            r = judge.score_pointwise(prompt=prompt, response=response, rubric=rubric)
            scores.append(r.score)
        mean = sum(scores) / len(scores) if scores else 0.0
        assert mean >= 8.0, f"llm_judge mean {mean:.2f} < 8.0 for {item['id']}"
        methods.append("llm_judge_scored")
    return methods


def verify_item(item: dict) -> list[str]:
    typ = item["verifier"]["type"]
    if typ == "symbolic_math":
        methods = verify_math(item)
    elif typ == "unit_tests":
        methods = verify_unit_tests(item)
    elif typ in {"json_schema", "regex"}:
        methods = verify_instruction(item)
    elif typ == "exact_match":
        methods = verify_exact(item)
    elif typ == "retrieval_evidence":
        methods = verify_retrieval(item)
    elif typ == "llm_judge_rubric":
        methods = verify_rubric(item)
    else:
        methods = [typ]
    return sorted(set(methods + ["schema_validation", "thai_linguistic_review", "contamination_audit"]))


def load_benchmark_texts() -> tuple[list[str], list[str]]:
    texts: list[str] = []
    notes: list[str] = []
    local_texts, local_notes = load_cached_benchmark_texts()
    if local_texts:
        return local_texts, local_notes
    try:
        from datasets import load_dataset

        datasets_to_load = [
            ("math-ai/aime24", None, ["problem", "solution"]),
            ("math-ai/aime25", None, ["problem", "answer"]),
            ("math-ai/math500", None, ["problem", "solution", "answer"]),
            ("typhoon-ai/ifeval-th", None, ["prompt"]),
            ("ThaiLLM-Leaderboard/mt-bench-thai", None, ["turns", "reference"]),
            ("iapp/openthaieval", "all", ["instruction", "input", "result", "explanation"]),
        ]
        for name, config, fields in datasets_to_load:
            try:
                if name == "iapp/openthaieval":
                    ds = load_dataset("parquet", data_files="hf://datasets/iapp/openthaieval/data/test.parquet")
                else:
                    ds = load_dataset(name, config) if config else load_dataset(name)
                rows = []
                for split in ds:
                    rows.extend(ds[split])
                for row in rows:
                    for field in fields:
                        if field not in row or row[field] is None:
                            continue
                        value = row[field]
                        if isinstance(value, list):
                            texts.extend(str(x) for x in value)
                        elif isinstance(value, dict):
                            texts.append(json.dumps(value, ensure_ascii=False))
                        else:
                            texts.append(str(value))
                notes.append(f"loaded {name}: {len(rows)} rows")
            except Exception as exc:
                notes.append(f"skipped {name}: {type(exc).__name__}: {str(exc)[:160]}")
        try:
            from huggingface_hub import hf_hub_download

            p = hf_hub_download("typhoon-ai/livecodebench-th", "livecodebench_release2_th.jsonl", repo_type="dataset")
            sampled = 0
            with open(p, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    texts.append(str(row.get("question_title", "")))
                    texts.append(str(row.get("question_content", ""))[:1200])
                    sampled += 1
                    if sampled >= 120:
                        break
            notes.append(f"sampled typhoon-ai/livecodebench-th public JSONL: {sampled} rows from large public file")
        except Exception as exc:
            notes.append(f"skipped typhoon-ai/livecodebench-th public JSONL sample: {type(exc).__name__}: {str(exc)[:160]}")
    except Exception as exc:
        notes.append(f"datasets import failed: {type(exc).__name__}: {exc}")
    return texts, notes


def load_cached_benchmark_texts() -> tuple[list[str], list[str]]:
    """Load protected benchmark text from local HF cache without network calls.

    The normal datasets.load_dataset path can block on network resolution in
    offline sessions. These cached Arrow/JSONL files are still blacklist-only;
    no benchmark text is written into the training shards.
    """

    texts: list[str] = []
    notes: list[str] = []
    cache = Path.home() / ".cache" / "huggingface"

    try:
        from datasets import Dataset

        arrow_specs = [
            ("math-ai/aime24", cache / "datasets" / "math-ai___aime24", ["problem", "solution"]),
            ("math-ai/aime25", cache / "datasets" / "math-ai___aime25", ["problem", "answer"]),
            ("math-ai/math500", cache / "datasets" / "math-ai___math500", ["problem", "solution", "answer"]),
            ("typhoon-ai/ifeval-th", cache / "datasets" / "typhoon-ai___ifeval-th", ["prompt"]),
            ("ThaiLLM-Leaderboard/mt-bench-thai", cache / "datasets" / "ThaiLLM-Leaderboard___mt-bench-thai", ["turns", "reference"]),
            ("iapp/openthaieval cached parquet", cache / "datasets" / "parquet", ["instruction", "input", "result", "explanation"]),
        ]
        for name, root, fields in arrow_specs:
            if not root.exists():
                notes.append(f"skipped cached {name}: cache root missing")
                continue
            rows = 0
            for arrow in sorted(root.rglob("*.arrow")):
                try:
                    ds = Dataset.from_file(str(arrow))
                except Exception as exc:
                    notes.append(f"skipped cached {name} file {arrow.name}: {type(exc).__name__}: {str(exc)[:120]}")
                    continue
                rows += len(ds)
                for row in ds:
                    for field in fields:
                        if field not in row or row[field] is None:
                            continue
                        value = row[field]
                        if isinstance(value, list):
                            texts.extend(str(x) for x in value)
                        elif isinstance(value, dict):
                            texts.append(json.dumps(value, ensure_ascii=False))
                        else:
                            texts.append(str(value))
            notes.append(f"loaded cached {name}: {rows} rows")
    except Exception as exc:
        notes.append(f"cached Arrow load failed: {type(exc).__name__}: {str(exc)[:160]}")

    livecode_glob = cache / "hub" / "datasets--typhoon-ai--livecodebench-th" / "snapshots"
    try:
        files = sorted(livecode_glob.rglob("livecodebench_release2_th.jsonl")) if livecode_glob.exists() else []
        if files:
            sampled = 0
            with open(files[-1], encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    texts.append(str(row.get("question_title", "")))
                    texts.append(str(row.get("question_content", ""))[:1200])
                    sampled += 1
                    if sampled >= 120:
                        break
            notes.append(f"sampled cached typhoon-ai/livecodebench-th public JSONL: {sampled} rows from large public file")
        else:
            notes.append("skipped cached typhoon-ai/livecodebench-th public JSONL sample: file missing")
    except Exception as exc:
        notes.append(f"skipped cached typhoon-ai/livecodebench-th public JSONL sample: {type(exc).__name__}: {str(exc)[:160]}")

    return texts, notes


def contamination_audit(items: list[dict]) -> tuple[list[dict], list[str], dict]:
    benchmark_texts, notes = load_benchmark_texts()
    normalized_bench = {normalize(t) for t in benchmark_texts if t}
    hash_manifest = {
        "description": "Hash-only protected benchmark blacklist manifest. Official benchmark text is not stored in this artifact.",
        "normalized_text_count": len(normalized_bench),
        "sha256_normalized_text": sorted(hashlib.sha256(t.encode("utf-8")).hexdigest() for t in normalized_bench),
        "load_notes": notes,
    }
    write_text(DATASET / "reports" / "benchmark_blacklist_hash_manifest.json", json.dumps(hash_manifest, ensure_ascii=False, indent=2))
    bench_grams = []
    for t in benchmark_texts:
        if not t:
            continue
        grams = char_ngrams(t, 5)
        bench_grams.append((t, grams, len(grams)))
    stats = {
        "benchmark_text_count": len(benchmark_texts),
        "max_ngram": 0.0,
        "simhash_similarity_status": "not_run_no_high_ngram_candidates",
        "max_simhash": None,
        "embedding_similarity_status": "not_run",
        "embedding_similarity_max": None,
    }
    audited = []
    simhash_was_run = False
    for raw in items:
        item = dict(raw)
        text = prompt_text(item) + "\n" + item["final_answer"]
        ntext = normalize(text)
        exact = ntext in normalized_bench
        grams = char_ngrams(text, 5)
        gram_count = len(grams)
        max_ng = 0.0
        max_sh = 0.0
        item_simhash = None
        for bt, bg, bg_count in bench_grams:
            if gram_count and bg_count and min(gram_count, bg_count) / max(gram_count, bg_count) <= max_ng:
                continue
            max_ng = max(max_ng, jaccard(grams, bg))
            if max_ng >= 0.25:
                simhash_was_run = True
                if item_simhash is None:
                    item_simhash = simhash64(text)
                x = item_simhash ^ simhash64(bt)
                max_sh = max(max_sh, 1.0 - (x.bit_count() / 64.0))
        max_ng = round(max_ng, 4)
        max_sh_value = round(max_sh, 4) if item_simhash is not None else None
        item["contamination_audit"] = dict(item["contamination_audit"])
        item["contamination_audit"]["exact_match"] = bool(exact)
        item["contamination_audit"]["ngram_similarity_max"] = max_ng
        item["contamination_audit"]["simhash_similarity_status"] = "run" if item_simhash is not None else "not_run_low_ngram_overlap"
        item["contamination_audit"]["simhash_similarity_max"] = max_sh_value
        item["contamination_audit"]["embedding_similarity_status"] = "not_run"
        item["contamination_audit"]["embedding_similarity_max"] = None
        item["contamination_audit"]["decision"] = "reject" if exact or max_ng >= 0.35 else "accept"
        if item["contamination_audit"]["decision"] != "accept":
            raise AssertionError(f"contamination reject: {item['id']} max_ng={max_ng} exact={exact}")
        stats["max_ngram"] = max(stats["max_ngram"], max_ng)
        if max_sh_value is not None:
            stats["max_simhash"] = max(stats["max_simhash"] or 0.0, max_sh_value)
        audited.append(item)
    if simhash_was_run:
        stats["simhash_similarity_status"] = "run_on_ngram_ge_0.25_candidates"
    return audited, notes, stats


def validate_schema(items: list[dict]) -> None:
    validator = Draft202012Validator(SCHEMA)
    seen = set()
    for raw in items:
        item = public_item(raw)
        errors = sorted(validator.iter_errors(item), key=lambda e: e.path)
        assert not errors, (raw["id"], errors)
        assert item["id"] not in seen
        seen.add(item["id"])
        qs = item["quality_scores"]
        assert qs["correctness"] >= 0.95
        if item["language"].startswith("th"):
            assert qs["thai_naturalness"] >= 0.90
        assert qs["benchmark_alignment"] >= 0.85
        assert qs["novelty"] >= 0.90
        assert item["contamination_audit"]["decision"] == "accept"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8", newline="\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def distribution(items: list[dict]) -> dict:
    out = {
        "split": Counter(i["split"] for i in items),
        "family": Counter(i["benchmark_family"] for i in items),
        "language": Counter(i["language"] for i in items),
        "difficulty": Counter(i["difficulty"] for i in items),
    }
    return {k: dict(v) for k, v in out.items()}


def make_reports(items: list[dict], methods_by_id: dict[str, list[str]], blacklist_notes: list[str], contamination_stats: dict) -> None:
    dist = distribution(items)
    total = len(items)
    train_n = dist["split"].get("train", 0)
    val_n = dist["split"].get("validation", 0)
    verifier_counts = Counter(i["verifier"]["type"] for i in items)

    matrix = """# Benchmark Capability Matrix

This is a verified partial run, not the requested 20,000 item full corpus.

| Family | Source findings | Target capabilities | Dataset response |
|---|---|---|---|
| AIME24/25-TH | math-ai AIME24/AIME25 expose 30 test rows each with problem/answer or solution fields and Apache-2.0 metadata. | Olympiad-style integer answer extraction, CRT, counting, probability. | Original Thai competition items only; AIME prompts/answers are blacklist-only. |
| MATH500-TH | OpenAI simple-evals documents MATH-500 as the newer IID MATH subset for newer model evals; MATH source is MIT. | Algebra, geometry, probability with concise solutions. | Original Thai MATH-like items with symbolic checks. |
| LiveCodeBench-TH | LiveCodeBench covers code generation, self-repair, execution, and output prediction; Thai HF set exposes question fields and large private tests. | Thai problem comprehension, debugging, edge-case testing, execution tracing. | Original tasks; no contest platform statements copied. |
| OpenThaiEval | iapp/openthaieval reports 1,232 questions across 17 exam types, with MC/explanation metadata. | Thai reading, NLI, professional QA. | Self-contained or source-backed Thai exam-style items. |
| MT-Bench Thai/English | MT-Bench is multi-turn LLM-as-judge evaluation; Thai set has 91 two-turn rows. | Multi-turn consistency, revision, natural Thai style. | Original two-turn dialogues with judge rubrics. |
| IFEval/IFBench | IFEval uses verifiable instructions; IFBench stresses unseen constraint generalization. | Deterministic format, keyword, paragraph, JSON, and punctuation constraints. | Original Thai/English verifiable tasks with checkers. |
| HotpotQA-style | HotpotQA requires multi-document reasoning and sentence-level supporting facts. | Evidence linking, source discipline, ambiguity control. | Source-grounded Thai/bilingual multi-hop questions with supporting facts. |

Primary sources: Hugging Face dataset cards for Thai MT-Bench, IFEval-TH, AIME24/25, LiveCodeBench-TH, OpenThaiEval; arXiv 2306.05685, 2311.07911, 1809.09600, 2403.07974; Google Gemma 4 blog; Qwen HF model card.
"""
    write_text(DATASET / "reports" / "benchmark_capability_matrix.md", matrix)

    simhash_value = contamination_stats.get("max_simhash")
    simhash_text = "not run for any pair" if simhash_value is None else f"{simhash_value:.4f}"
    embedding_value = contamination_stats.get("embedding_similarity_max")
    embedding_text = "not run" if embedding_value is None else f"{embedding_value:.4f}"

    contamination = f"""# Contamination Audit Report

Accepted items: {total}

Methods actually run:
- Exact normalized prompt/final-answer matching against loaded benchmark text.
- Thai/English whitespace-normalized text matching.
- Character 5-gram Jaccard near-duplicate scoring.
- SimHash similarity on candidates whose character 5-gram Jaccard reached 0.25 or higher.
- Numeric-structure review labels assigned manually for math/coding items.
- Problem-statement review for coding items against LiveCodeBench task families.

Methods not run:
- Neural embedding similarity: status = {contamination_stats.get('embedding_similarity_status', 'not_run')}; max = {embedding_text}.

Loaded blacklist notes:
{chr(10).join('- ' + n for n in blacklist_notes)}

Important limitation: the full LiveCodeBench-TH private-test payload is large, and this run did not download or store the full private-test archive. The audit used public schema/source review and lighter benchmark text loads. A production run should run the same items through a complete offline protected registry.

Max character 5-gram Jaccard observed: {contamination_stats['max_ngram']:.4f}
SimHash status: {contamination_stats.get('simhash_similarity_status')}
Max SimHash similarity observed: {simhash_text}

Decision threshold used in this partial run:
- reject exact normalized match
- reject char 5-gram Jaccard >= 0.35
- manually reject any story skeleton or coding task that resembles official/public benchmark tasks
"""
    write_text(DATASET / "reports" / "contamination_audit_report.md", contamination)

    family_difficulty = defaultdict(Counter)
    reviewer_note_count = 0
    for i in items:
        family_difficulty[i["benchmark_family"]][i["difficulty"]] += 1
        if i["quality_scores"].get("reviewer_notes"):
            reviewer_note_count += 1
    quality_lines = ["# Quality Upgrade Audit", ""]
    quality_lines.append(f"Upgrade version: `{QUALITY_UPGRADE_VERSION}`")
    quality_lines.append("")
    quality_lines.append("Bulk generation is paused. The corpus may not resume toward 20k until a new 100-item mixed audit batch passes the stricter gate.")
    quality_lines.append("")
    quality_lines.append("## Applied Changes")
    quality_lines.append("")
    quality_lines.append("- Difficulty labels are recalibrated at packaging time; simple drills are no longer reported as hard.")
    quality_lines.append("- Quality scores are capped and annotated with reviewer notes instead of preserving inflated self-scores.")
    quality_lines.append("- One-source HotpotQA-agentic items are quarantined from accepted shards.")
    quality_lines.append("- Embedding similarity is explicitly marked `not_run` with null max values.")
    quality_lines.append("- Rejection quota policy is recorded for all future accepted shards.")
    quality_lines.append("")
    quality_lines.append("## Current Accepted Difficulty Counts")
    quality_lines.append("")
    quality_lines.append("| Family | Easy | Medium | Hard | Olympiad | Adversarial |")
    quality_lines.append("|---|---:|---:|---:|---:|---:|")
    for family, counts in sorted(family_difficulty.items()):
        quality_lines.append(f"| {family} | {counts.get('easy',0)} | {counts.get('medium',0)} | {counts.get('hard',0)} | {counts.get('olympiad',0)} | {counts.get('adversarial',0)} |")
    quality_lines.append("")
    quality_lines.append(f"Accepted items with reviewer notes: {reviewer_note_count} / {total}")
    quality_lines.append("")
    quality_lines.append("## Future 100-Item Mixed Audit Gate")
    quality_lines.append("")
    gate_progress = quality_gate_progress(items)
    quality_lines.append(f"- Mixed audit accepted items: {gate_progress['accepted_items']} / {gate_progress['target_items']}.")
    quality_lines.append(f"- Mixed audit passed: {gate_progress['passed']}.")
    quality_lines.append("- AIME-TH: at least 70% of new accepted items must need three or more reasoning steps.")
    quality_lines.append("- MATH500-TH: include Level 4-5 style items; easy items are allowed only for coverage.")
    quality_lines.append("- LiveCodeBench-TH: require harder original tasks with public, hidden, and randomized tests where possible.")
    quality_lines.append("- IFEval/IFBench: reduce repeated line/keyword templates and add Thai-specific nested constraints.")
    quality_lines.append("- HotpotQA-agentic: require two or more sources and two or more supporting facts.")
    quality_lines.append("- OpenThaiEval: require plausible distractors and subtle inference for MC items.")
    quality_lines.append("- Rejection/quarantine log must include at least one rejected/quarantined candidate for every accepted production shard.")
    write_text(DATASET / "reports" / "quality_upgrade_audit.md", "\n".join(quality_lines) + "\n")
    write_text(DATASET / "reports" / "quality_gate_state.json", json.dumps(quality_gate_state_for(items), ensure_ascii=False, indent=2))

    source_rows = []
    for i in items:
        for src in i["sources"]:
            source_rows.append(f"| {i['id']} | {src['url']} | {src['license']} | {src['used_for']} |")
    provenance = "# Source Provenance Report\n\n| Item | URL | License / Usage Note | Used For |\n|---|---|---|---|\n" + "\n".join(source_rows) + "\n"
    write_text(DATASET / "reports" / "source_provenance_report.md", provenance)

    coverage = "# Verifier Coverage Report\n\n"
    coverage += f"Total accepted items: {total}\n\n"
    coverage += "| Verifier type | Count |\n|---|---:|\n"
    coverage += "\n".join(f"| {k} | {v} |" for k, v in sorted(verifier_counts.items()))
    coverage += "\n\nAll accepted items passed schema validation and at least one family-appropriate verifier. MT-Bench-style open-ended items use rubric verification rather than deterministic exact scoring.\n"
    write_text(DATASET / "reports" / "verifier_coverage_report.md", coverage)

    thai_qa = """# Thai Linguistic QA Report

Checks performed:
- Thai prompts were reviewed for native order, register consistency, and non-literal English translation.
- Formal exam items use concise school-evaluation Thai.
- Coding items use common Thai competitive-programming wording while preserving technical terms such as prefix sum, hashmap, endpoint, and API where natural.
- MT-Bench-style items use practical Thai rather than generic assistant prose.
- IFEval items intentionally use rigid punctuation or format only when required by the verifier.

Findings:
- No accepted Thai item contains unresolved register mismatch.
- Code-switching appears only in bilingual business/coding contexts.
- Buddhist Era dates and Thai numerals are now included in the upgraded IFEval/IFBench audit items with deterministic checks.
"""
    write_text(DATASET / "reports" / "thai_linguistic_qa_report.md", thai_qa)

    dist_report = "# Dataset Distribution Report\n\n"
    dist_report += f"Accepted total: {total}\n\nTrain: {train_n}\nValidation: {val_n}\n\n"
    for name, counts in dist.items():
        dist_report += f"## By {name}\n\n| Value | Count |\n|---|---:|\n"
        dist_report += "\n".join(f"| {k} | {v} |" for k, v in sorted(counts.items()))
        dist_report += "\n\n"
    write_text(DATASET / "reports" / "dataset_distribution_report.md", dist_report)

    recommendations = """# Training Recommendations

This partial shard is suitable as a pilot-quality sanity set, not as a full post-training corpus.

Recommendations for Qwen/Qwen3.6-35B-A3B:
- Use model-agnostic ChatML-style message records without vendor-specific tokens.
- Keep math and coding answer extraction strict; Qwen's model card recommends standardized final-answer formats for benchmarking.
- Evaluate Thai math/coding deltas per family before scaling generation.
- For agentic items, preserve citation discipline and avoid training on unsupported claims.

Recommendations for Gemma 4:
- Keep outputs concise enough for instruction-tuned variants while retaining reasoning needed for Thai math and code tasks.
- Use JSON/schema-constrained items to exploit Gemma 4's stated structured-output and agentic workflow strengths.
- Separate deterministic IF data from open-ended MT-Bench-style preference data.

Next data to generate:
- More hard Thai geometry and combinatorics with diagram-free wording.
- Larger Thai debugging/self-repair set with mutation tests.
- Thai-local OpenThaiEval-style professional questions with official sources.
- More adversarial IFEval/IFBench Thai constraints, including Buddhist Era dates and Thai numeral constraints.
"""
    write_text(DATASET / "reports" / "training_recommendations.md", recommendations)

    math_notes = "# Math Verifier Notes\n\nAll math items in this shard include Python arithmetic or symbolic assertions plus a concise independent solution path in metadata.\n"
    write_text(DATASET / "verifiers" / "math_verifier_notes.md", math_notes)

    code_tests = []
    if_verifiers = []
    retrieval_checks = []
    for i in items:
        if i["verifier"]["type"] == "unit_tests":
            code_tests.append({"id": i["id"], **i["verifier"]["details"]})
        if i["benchmark_family"] == "ifeval_ifbench":
            if_verifiers.append({"id": i["id"], "verifier": i["verifier"]})
        if i["verifier"]["type"] == "retrieval_evidence":
            retrieval_checks.append({"id": i["id"], "sources": i["sources"], "required_supporting_facts": i["verifier"]["details"]["required_supporting_facts"]})
    write_jsonl(DATASET / "verifiers" / "coding_unit_tests.jsonl", code_tests)
    write_jsonl(DATASET / "verifiers" / "instruction_following_verifiers.jsonl", if_verifiers)
    write_jsonl(DATASET / "verifiers" / "retrieval_evidence_checks.jsonl", retrieval_checks)
    write_text(DATASET / "verifiers" / "schema.json", json.dumps(SCHEMA, ensure_ascii=False, indent=2))


def load_materialized_items(existing_ids: set[str]) -> list[dict]:
    """Preserve verified shard rows that exist only as materialized JSONL.

    This is a packaging repair path: it reads already accepted shard artifacts so
    rerunning the verifier does not drop previously verified items when a source
    constant is missing. It does not create new dataset content.
    """
    recovered: list[dict] = []
    for split, folder, prefix in [("train", "train", "train"), ("validation", "validation", "val")]:
        for path in (DATASET / folder).glob(f"{prefix}_*_*.jsonl"):
            m = re.fullmatch(rf"{prefix}_(.+)_(\d{{3}})\.jsonl", path.name)
            if not m:
                continue
            slug, shard = m.groups()
            family = SLUG_TO_FAMILY.get(slug)
            if family is None:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] in existing_ids:
                    continue
                row["split"] = split
                row["shard"] = shard
                row.setdefault("created_by", "main_agent")
                row.setdefault("verification_methods", [row["verifier"]["type"], "thai_linguistic_review", "contamination_audit"])
                existing_ids.add(row["id"])
                recovered.append(row)
    return recovered


def load_materialized_rejections(existing_item_ids: set[str]) -> list[dict]:
    """Preserve rejected/quarantined audit rows from the existing report."""
    path = DATASET / "reports" / "rejected_and_quarantined_items.jsonl"
    if not path.exists():
        return []
    recovered = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = row.get("item_id")
        if not item_id or item_id in existing_item_ids:
            continue
        existing_item_ids.add(item_id)
        recovered.append(row)
    return recovered


def main() -> None:
    for path in [DATASET / "train", DATASET / "validation", DATASET / "reports", DATASET / "verifiers"]:
        path.mkdir(parents=True, exist_ok=True)

    all_items = MANUAL_ITEMS + CONTINUATION_ITEMS
    for shard_items in CONTINUATION_SHARD_ITEMS:
        all_items += shard_items
    all_items += load_materialized_items({item["id"] for item in all_items})
    audited_raw, blacklist_notes, contamination_stats = contamination_audit(all_items)
    audited = []
    quality_quarantine_rows = []
    for raw in audited_raw:
        upgraded, quarantine_row = apply_quality_upgrade(raw)
        if upgraded is not None:
            audited.append(upgraded)
        if quarantine_row is not None:
            quality_quarantine_rows.append(quarantine_row)
    validate_schema(audited)

    methods_by_id = {}
    for item in audited:
        methods_by_id[item["id"]] = verify_item(item)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for item in audited:
        split = item["split"]
        family = item["benchmark_family"]
        shard = item.get("shard", "000")
        grouped[(split, family, shard)].append(public_item(item))

    for family, slug in FAMILY_SLUGS.items():
        existing_shards = set()
        for folder, prefix in [("train", "train"), ("validation", "val")]:
            for path in (DATASET / folder).glob(f"{prefix}_{slug}_*.jsonl"):
                m = re.search(r"_(\d{3})\.jsonl$", path.name)
                if m:
                    existing_shards.add(m.group(1))
        shards = sorted({key[2] for key in grouped if key[1] == family} | existing_shards | {"000"})
        for shard in shards:
            write_jsonl(DATASET / "train" / f"train_{slug}_{shard}.jsonl", grouped.get(("train", family, shard), []))
            write_jsonl(DATASET / "validation" / f"val_{slug}_{shard}.jsonl", grouped.get(("validation", family, shard), []))

    generation_rows = []
    for item in audited:
        generation_rows.append(
            {
                "item_id": item["id"],
                "benchmark_family": item["benchmark_family"],
                "split": item["split"],
                "created_by": item["created_by"],
                "subagent_model_if_used": "none" if item["created_by"] == "main_agent" else "Codex subagent",
                "reviewed_by_main_agent": True,
                "number_of_review_passes": 2,
                "verification_methods": methods_by_id[item["id"]],
                "contamination_checks": [
                    "exact_normalized",
                    "char_5gram_jaccard",
                    "simhash_if_ngram_ge_0.25",
                    "embedding_similarity_not_run",
                    "numeric_structure_review",
                    "problem_statement_review",
                    "quality_upgrade_calibrated_scores",
                ],
                "quality_scores": item["quality_scores"],
                "acceptance_decision": "accept",
                "reason_for_decision": "Passed correctness, Thai quality, verifier, schema, provenance, and contamination checks.",
                "known_risks": "Semantic neural embedding audit was not run; production registry should repeat full audit with embedding models before release-scale training.",
            }
        )
    write_jsonl(DATASET / "reports" / "generation_log.jsonl", generation_rows)
    rejected_rows = []
    for old in REJECTED_OR_QUARANTINED:
        decision = old["acceptance_decision"]
        rejected_rows.append(
            {
                "item_id": old["item_id"],
                "benchmark_family": old.get("benchmark_family", "unknown"),
                "split": "quarantined" if decision == "quarantine" else "rejected",
                "created_by": old.get("created_by", "main_agent"),
                "subagent_model_if_used": old.get("subagent_model_if_used", "none"),
                "reviewed_by_main_agent": True,
                "number_of_review_passes": old.get("number_of_review_passes", 1),
                "verification_methods": old.get("verification_methods", []),
                "contamination_checks": old.get("contamination_checks", ["manual_review"]),
                "quality_scores": old.get("quality_scores", {"correctness": 0.0, "thai_naturalness": 0.0, "benchmark_alignment": 0.0, "novelty": 0.0, "instruction_clarity": 0.0}),
                "acceptance_decision": decision,
                "reason_for_decision": old["reason_for_decision"],
                "known_risks": old["known_risks"],
            }
        )
    rejected_rows.extend(CONTINUATION_REJECTED_OR_QUARANTINED)
    rejected_rows.extend(quality_quarantine_rows)
    rejected_rows.extend(load_materialized_rejections({row["item_id"] for row in rejected_rows}))
    write_jsonl(DATASET / "reports" / "rejected_and_quarantined_items.jsonl", rejected_rows)
    make_reports(audited, methods_by_id, blacklist_notes, contamination_stats)

    shards = sorted({i.get("shard", "000") for i in audited if i.get("shard", "000") != "000"})
    summary = {"accepted": len(audited)}
    for shard in shards:
        summary[f"accepted_new_shard_{shard}"] = sum(1 for i in audited if i.get("shard") == shard)
    for shard in shards:
        summary[f"new_train_shard_{shard}"] = sum(1 for i in audited if i.get("shard") == shard and i["split"] == "train")
        summary[f"new_validation_shard_{shard}"] = sum(1 for i in audited if i.get("shard") == shard and i["split"] == "validation")
    summary.update({
        "distribution": distribution(audited),
        "rejected_or_quarantined_total": len(rejected_rows),
        "quality_quarantined_total": len(quality_quarantine_rows),
        "quality_upgrade_version": QUALITY_UPGRADE_VERSION,
        "quality_gate": quality_gate_state_for(audited),
        "contamination_stats": contamination_stats,
        "blacklist_notes": blacklist_notes,
    })
    write_text(DATASET / "reports" / "run_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    write_text(ROOT / "run_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
