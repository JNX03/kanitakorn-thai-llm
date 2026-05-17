"""Phase 3.1 — training-time few-shot wrapper.

Deterministic hash-bucketed picker. Same `seed` + same `family` + same
`difficulty` → same in-context examples across runs, so SFT data is bit-for-bit
reproducible.

CLI:
    python tools/few_shot_collator.py --out dataset/sft_ready [--limit N]

Output: dataset/sft_ready/{family}_{split}.jsonl, one JSON-line per record
with a `text` field formatted in Qwen / Gemma chat-template style. Records
that already have multi-turn `messages` (mt_bench, teacher_loop_th) are
serialized using the chat template directly; single-turn records get N few-
shot examples prepended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"

FAMILY_FEW_SHOT_K = {
    "aime_th": 3,
    "math500_th": 3,
    "livecodebench_th": 2,
    "openthaieval": 3,
    "mt_bench": 1,
    "ifeval_ifbench": 2,
    "hotpotqa_agentic": 1,
    "teacher_loop_th": 0,  # already a multi-turn lesson
}

CHAT_TEMPLATE_SYSTEM = (
    "You are a helpful assistant fluent in Thai and English. Follow all "
    "constraints stated in the user's request precisely."
)


def stable_seed(family: str, difficulty: str, record_id: str) -> int:
    h = hashlib.sha256(f"{family}|{difficulty}|{record_id}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def load_split(split: str) -> dict[str, list[dict]]:
    """Return {family: [records...]} from dataset/{train|validation}/."""
    folder = DATASET / split
    out: dict[str, list[dict]] = defaultdict(list)
    if not folder.exists():
        return out
    for path in sorted(folder.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec["benchmark_family"]].append(rec)
    return out


def pick_few_shot(pool: list[dict], k: int, seed: int, exclude_id: str) -> list[dict]:
    """Pick k examples deterministically, excluding the target record."""
    candidates = [r for r in pool if r["id"] != exclude_id]
    if not candidates or k == 0:
        return []
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:k]


def render_chat_template(messages: list[dict], system: str = CHAT_TEMPLATE_SYSTEM) -> str:
    """Render Qwen-/Gemma-style chat template.

    Qwen format:
        <|im_start|>system\n{system}<|im_end|>
        <|im_start|>user\n{user}<|im_end|>
        <|im_start|>assistant\n{assistant}<|im_end|>
    """
    parts = [f"<|im_start|>system\n{system}<|im_end|>"]
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    return "\n".join(parts)


def render_example_block(example: dict) -> str:
    """Render a few-shot example as plain Q/A pairs (not chat-template-wrapped)."""
    user = "\n".join(m["content"] for m in example["messages"] if m["role"] == "user")
    asst = "\n".join(m["content"] for m in example["messages"] if m["role"] == "assistant")
    return f"# Example\n## Question\n{user}\n## Answer\n{asst}"


def build_sft_text(record: dict, family_pool: list[dict], k: int) -> str:
    examples = pick_few_shot(
        family_pool,
        k,
        seed=stable_seed(record["benchmark_family"], record["difficulty"], record["id"]),
        exclude_id=record["id"],
    )
    # Prepend few-shot examples to the user's message, render whole convo with chat template.
    if examples:
        prefix = "\n\n".join(render_example_block(e) for e in examples)
        msgs = [dict(m) for m in record["messages"]]
        # Find first user message; prepend the few-shot block.
        for m in msgs:
            if m["role"] == "user":
                m["content"] = prefix + "\n\n# Your task\n" + m["content"]
                break
    else:
        msgs = record["messages"]
    return render_chat_template(msgs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DATASET / "sft_ready"))
    parser.add_argument("--limit", type=int, default=None, help="optional cap per (family, split)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = defaultdict(dict)
    for split in ("train", "validation"):
        by_family = load_split(split)
        for family, records in by_family.items():
            k = FAMILY_FEW_SHOT_K.get(family, 2)
            # Few-shot pool is always drawn from train.
            train_pool = load_split("train").get(family, []) if split != "train" else records
            limit = args.limit if args.limit else len(records)
            out_path = out_dir / f"{family}_{split}.jsonl"
            with out_path.open("w", encoding="utf-8") as fh:
                for rec in records[:limit]:
                    text = build_sft_text(rec, train_pool, k)
                    fh.write(
                        json.dumps(
                            {
                                "id": rec["id"],
                                "family": family,
                                "split": split,
                                "difficulty": rec.get("difficulty"),
                                "language": rec.get("language"),
                                "text": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            summary[family][split] = limit
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote sft_ready files under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
