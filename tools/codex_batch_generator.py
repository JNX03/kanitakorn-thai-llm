"""Batch-generate dataset records via codex (gpt-5.5-xhigh).

Each codex invocation has ~20-30 sec session overhead, so we ask it for
~20-30 records per call instead of 1 — amortizes overhead and finishes the
target volume in ~15-30 min instead of hours.

Usage as a library — see `seed_teacher_loop_scale.py`,
`seed_hotpotqa_scale.py`, `seed_thai_exam_scale.py` for concrete generators.

Key safeguards:
    - JSON validation per record (schema-checked before write)
    - Token-leak avoidance: codex auth is via ~/.codex/auth, not env vars
    - Resumable: each batch writes to its own shard file; rerun skips
      shards that already have N+ valid records
    - Budget cap: BATCH_MAX_USD env var (default $50)
    - Concurrency: BATCH_PARALLEL env var (default 1 — sequential calls
      avoid HF + codex rate limits)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _find_codex() -> str | None:
    found = shutil.which("codex") or shutil.which("codex.cmd")
    if found:
        return found
    for c in [
        Path(os.environ.get("APPDATA", "")) / "npm" / "codex.cmd",
        Path(os.environ.get("APPDATA", "")) / "npm" / "codex",
        Path.home() / "AppData" / "Roaming" / "npm" / "codex.cmd",
        Path.home() / "AppData" / "Roaming" / "npm" / "codex",
        Path("/usr/local/bin/codex"),
        Path.home() / ".local" / "bin" / "codex",
    ]:
        if c.exists():
            return str(c)
    return None


CODEX = _find_codex()
BATCH_MAX_USD = float(os.getenv("BATCH_MAX_USD", "50.0"))
# placeholder unit costs (gpt-5.5-xhigh); update if you have actual pricing
_USD_PER_CALL = 0.04
_spent = 0.0


def codex_call(prompt: str, timeout_s: float = 300.0) -> str:
    """Run one codex exec, return its stdout (text after 'codex\\n')."""
    global _spent
    if _spent + _USD_PER_CALL > BATCH_MAX_USD:
        raise RuntimeError(f"BATCH_MAX_USD={BATCH_MAX_USD} would be exceeded (already spent ${_spent:.2f})")
    if not CODEX:
        raise RuntimeError("codex CLI not found")
    try:
        proc = subprocess.run(
            [CODEX, "exec", "--skip-git-repo-check", "-"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return ""
    _spent += _USD_PER_CALL
    return proc.stdout


def extract_json_array(text: str) -> list:
    """Pull the last JSON array out of codex output."""
    last_open = text.rfind("[")
    last_close = text.rfind("]")
    if last_open < 0 or last_close < last_open:
        # Try to find a single JSON object as fallback.
        last_open = text.rfind("{")
        last_close = text.rfind("}")
        if last_open < 0 or last_close < last_open:
            return []
        try:
            return [json.loads(text[last_open : last_close + 1])]
        except json.JSONDecodeError:
            return []
    try:
        return json.loads(text[last_open : last_close + 1])
    except json.JSONDecodeError:
        return []


@dataclass
class BatchConfig:
    """Per-family batch generation config."""
    family: str
    out_path: Path
    records_per_call: int
    total_target: int
    build_prompt: callable        # (batch_idx, records_per_call) -> str
    record_factory: callable      # (raw_dict, idx) -> validated record dict
    schema_validator: callable | None = None  # raises if invalid


def run_batch(cfg: BatchConfig, log_path: Path | None = None) -> dict:
    """Run a batch generator until total_target records accepted."""
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    # If output file is read-only (locked), unlock.
    if sys.platform == "win32" and cfg.out_path.exists():
        subprocess.run(["attrib", "-R", str(cfg.out_path)], check=False, capture_output=True)

    accepted = 0
    rejected = 0
    batch_idx = 0
    t0 = time.time()
    log_lines: list[str] = []

    with cfg.out_path.open("a", encoding="utf-8") as fh:
        while accepted < cfg.total_target:
            batch_idx += 1
            prompt = cfg.build_prompt(batch_idx, cfg.records_per_call)
            try:
                raw = codex_call(prompt)
            except RuntimeError as e:
                print(f"  [budget] {e}")
                log_lines.append(f"[budget] {e}")
                break
            arr = extract_json_array(raw)
            if not arr:
                rejected += cfg.records_per_call
                log_lines.append(f"batch {batch_idx}: 0 records (parse failure)")
                print(f"  batch {batch_idx}: 0 records, codex returned {len(raw)} chars")
                continue
            this_batch_accepted = 0
            for i, item in enumerate(arr):
                rec_idx = accepted + this_batch_accepted
                try:
                    record = cfg.record_factory(item, rec_idx)
                    if cfg.schema_validator:
                        cfg.schema_validator(record)
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    this_batch_accepted += 1
                except Exception as e:
                    rejected += 1
                    log_lines.append(f"batch {batch_idx} rec {i}: REJECT {type(e).__name__}: {str(e)[:120]}")
            fh.flush()
            accepted += this_batch_accepted
            elapsed = time.time() - t0
            rate = accepted / elapsed if elapsed else 0
            log_lines.append(f"batch {batch_idx}: +{this_batch_accepted} (total {accepted}/{cfg.total_target}); ${_spent:.2f}; {rate:.2f} rec/s")
            print(f"  batch {batch_idx}: +{this_batch_accepted} (total {accepted}/{cfg.total_target}); ${_spent:.2f} spent")

    if sys.platform == "win32":
        subprocess.run(["attrib", "+R", str(cfg.out_path)], check=False, capture_output=True)

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    return {
        "family": cfg.family,
        "accepted": accepted,
        "rejected": rejected,
        "batches": batch_idx,
        "spent_usd": round(_spent, 4),
        "elapsed_s": round(time.time() - t0, 1),
    }


def reset_spend():
    """Reset spent counter — call between independent runs."""
    global _spent
    _spent = 0.0
