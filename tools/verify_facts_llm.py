"""LLM-as-judge fact verification for hand-authored records (W2 v2).

Replaces the keyword-matching verify_facts.py with a real LLM judgment:
fetches each source URL, gives the page text + claimed facts to the model,
and asks "do the sources actually support these facts?" — returns JSON
{verdict: OK|WARN|FAIL, supported: [...], unsupported: [...], rationale}.

Backend: `codex exec` (gpt-5.5 xhigh). The codex CLI handles auth via its own
login state, so no OPENAI_API_KEY env var is required. Each record is ~1
codex call, ~$0.02 — at 30 records that's <$1 total.

Run on a single record file:
    python tools/verify_facts_llm.py --check dataset/train/train_hotpotqa_agentic_seed.jsonl

Output:
    dataset/reports/fact_verification_llm.md — markdown table per record
    dataset/reports/fact_verification_llm.jsonl — raw verdicts for the audit log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "dataset" / "reports"


def _find_codex() -> str | None:
    """Locate the codex CLI across Windows/POSIX install locations."""
    # 1. Standard PATH lookup.
    found = shutil.which("codex") or shutil.which("codex.cmd") or shutil.which("codex.ps1")
    if found:
        return found
    # 2. Common npm install paths.
    candidates = [
        Path(os.environ.get("APPDATA", "")) / "npm" / "codex.cmd",
        Path(os.environ.get("APPDATA", "")) / "npm" / "codex",
        Path.home() / "AppData" / "Roaming" / "npm" / "codex.cmd",
        Path.home() / "AppData" / "Roaming" / "npm" / "codex",
        Path("/usr/local/bin/codex"),
        Path.home() / ".local" / "bin" / "codex",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


_CODEX_PATH = _find_codex()


def fetch_page_text(url: str, timeout_s: float = 15.0, max_chars: int = 8000) -> str:
    """Fetch a URL and return a plain-text excerpt (HTML tags stripped)."""
    try:
        encoded = urllib.parse.quote(url, safe=":/?=&#%+")
        req = urllib.request.Request(encoded, headers={"User-Agent": "Mozilla/5.0 fact-verifier"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.getcode() != 200:
                return f"[FETCH ERROR: HTTP {resp.getcode()}]"
            data = resp.read()
    except Exception as e:
        return f"[FETCH ERROR: {type(e).__name__}: {e}]"
    html = data.decode("utf-8", errors="ignore")
    # Crude HTML → text: strip scripts/styles, then tags, collapse whitespace.
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


_PROMPT_TEMPLATE = """\
You are a strict fact-verification auditor. The dataset author claims the following facts are supported by these source URLs. Decide whether the source pages actually support EACH claim. Return STRICT JSON only — no preamble, no markdown fences.

## Record id
{record_id}

## Claimed answer
{answer}

## Source pages (excerpts; HTML stripped, max 8000 chars each)

{pages_block}

## Required supporting facts (each must be supported by at least one source above)

{facts_block}

Return JSON in this exact shape:
{{
  "verdict": "OK" | "WARN" | "FAIL",
  "supported_facts": [<list of facts that ARE supported, by index 0..N-1>],
  "unsupported_facts": [<list of indices where the source does NOT actually support the claim>],
  "rationale": "<one short sentence>"
}}

Use FAIL only when at least one fact is actively contradicted or wholly missing from all sources.
Use WARN when sources are tangentially related but don't clearly state the fact.
Use OK only when every fact is clearly supported by the fetched text.
"""


def codex_judge(prompt: str, timeout_s: float = 180.0) -> dict:
    """Run codex exec and parse the JSON verdict."""
    if not _CODEX_PATH:
        return {"verdict": "FAIL", "rationale": "codex CLI not found in PATH or common install dirs", "supported_facts": [], "unsupported_facts": []}
    try:
        result = subprocess.run(
            [_CODEX_PATH, "exec", "--skip-git-repo-check", "-"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(ROOT),
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return {"verdict": "FAIL", "rationale": "codex timeout", "supported_facts": [], "unsupported_facts": []}
    except FileNotFoundError as e:
        return {"verdict": "FAIL", "rationale": f"codex exec failed: {e}", "supported_facts": [], "unsupported_facts": []}
    except OSError as e:
        return {"verdict": "FAIL", "rationale": f"OSError calling codex: {e}", "supported_facts": [], "unsupported_facts": []}
    stdout = result.stdout
    # codex output includes a header + "codex\n<answer>\n--" block. Pull the last {...}.
    last_open = stdout.rfind("{")
    last_close = stdout.rfind("}")
    if last_open < 0 or last_close < last_open:
        return {"verdict": "FAIL", "rationale": f"no JSON in codex output: {stdout[:200]}", "supported_facts": [], "unsupported_facts": []}
    candidate = stdout[last_open : last_close + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        return {"verdict": "FAIL", "rationale": f"unparseable JSON: {e}; raw: {candidate[:200]}", "supported_facts": [], "unsupported_facts": []}


def verify_record_llm(rec: dict) -> dict:
    sources = rec.get("sources", [])
    urls = [s.get("url", "") for s in sources if s.get("url", "").startswith("http")]
    facts = rec.get("verifier", {}).get("details", {}).get("required_supporting_facts", [])

    pages = []
    for url in urls:
        text = fetch_page_text(url)
        pages.append(f"### Source: {url}\n{text}\n")
    pages_block = "\n".join(pages) if pages else "[no sources provided]"

    facts_block = "\n".join(f"{i}. {fact}" for i, fact in enumerate(facts)) if facts else "[no facts listed]"

    prompt = _PROMPT_TEMPLATE.format(
        record_id=rec.get("id", "<no-id>"),
        answer=rec.get("final_answer", ""),
        pages_block=pages_block,
        facts_block=facts_block,
    )
    verdict = codex_judge(prompt)
    verdict["id"] = rec.get("id", "<no-id>")
    verdict["claimed_answer"] = rec.get("final_answer", "")
    verdict["n_facts"] = len(facts)
    return verdict


def write_reports(results: list[dict], out_md: Path, out_jsonl: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {"OK": 0, "WARN": 0, "FAIL": 0}
    for r in results:
        counts[r.get("verdict", "FAIL")] = counts.get(r.get("verdict", "FAIL"), 0) + 1

    lines = [
        "# Fact verification report (LLM-as-judge)",
        "",
        f"Judge: codex (gpt-5.5 xhigh). Backend reads the full fetched HTML excerpt and decides whether sources actually support the claims.",
        "",
        f"**OK: {counts.get('OK', 0)} · WARN: {counts.get('WARN', 0)} · FAIL: {counts.get('FAIL', 0)} · TOTAL: {len(results)}**",
        "",
        "| id | verdict | claimed answer | rationale | unsupported facts |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        rid = r.get("id", "")
        verdict = r.get("verdict", "?")
        ans = (r.get("claimed_answer") or "")[:60]
        rat = (r.get("rationale") or "")[:150]
        unsup = ", ".join(str(i) for i in r.get("unsupported_facts", []))
        lines.append(f"| {rid} | {verdict} | {ans} | {rat} | {unsup or '—'} |")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", required=True)
    parser.add_argument("--out-md", default=str(REPORTS / "fact_verification_llm.md"))
    parser.add_argument("--out-jsonl", default=str(REPORTS / "fact_verification_llm.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    p = Path(args.check)
    records: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("benchmark_family") != "hotpotqa_agentic":
            continue
        records.append(rec)
        if args.limit and len(records) >= args.limit:
            break

    results: list[dict] = []
    for i, rec in enumerate(records, start=1):
        print(f"[{i}/{len(records)}] judging {rec.get('id')} ...", flush=True)
        r = verify_record_llm(rec)
        print(f"  -> {r.get('verdict', '?')}  {r.get('rationale', '')[:120]}", flush=True)
        results.append(r)

    write_reports(results, Path(args.out_md), Path(args.out_jsonl))
    print(f"wrote {args.out_md}")
    failed = sum(1 for r in results if r.get("verdict") == "FAIL")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
