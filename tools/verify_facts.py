"""Fact-verification gate for hand-authored records (W2).

Before any hand-authored record is added to dataset/, it must pass this check:

    1. Every URL in `sources` must return HTTP 200 (verifiable).
    2. Every claim in `verifier.details.required_supporting_facts` must be
       cross-referenced against at least one of those sources (substring match
       on the page summary, or — when available — an LLM judge that reads the
       fetched page).

Run on a single record file:
    python tools/verify_facts.py --check dataset/train/train_hotpotqa_agentic_seed.jsonl

Output:
    dataset/reports/fact_verification.md — one row per record with status:
      OK         every URL resolves, supporting facts match the fetched text
      WARN       URLs resolve but facts can't be matched literally — needs LLM-judge
      FAIL       at least one URL is broken or returns non-relevant content

CRITICAL: this script does NOT auto-fix. It flags. Fixes must be made by the
author who knows the intended claim.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "dataset" / "reports"


def fetch_page(url: str, timeout_s: float = 15.0) -> tuple[int, str]:
    """Fetch a URL, return (http_status, body_text). Body is lowercased.

    Handles Thai (and other non-ASCII) characters in the path via urlquote.
    """
    try:
        # urllib needs the path component percent-encoded for non-ASCII chars.
        encoded = urllib.parse.quote(url, safe=":/?=&#%+")
        req = urllib.request.Request(encoded, headers={"User-Agent": "Mozilla/5.0 fact-verifier"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = resp.getcode()
            data = resp.read()
        if not isinstance(data, bytes):
            data = bytes(data)
        return status, data.decode("utf-8", errors="ignore").lower()
    except Exception as e:
        return 0, f"fetch error: {type(e).__name__}: {e}"


def strip_thai_punct(s: str) -> str:
    return re.sub(r"[\s​()\[\]{}.,:;!?'\"`’“”]+", " ", s.lower()).strip()


def fact_keywords(fact: str) -> list[str]:
    """Extract candidate keyword spans from a Thai/English fact string.

    Conservative: pulls noun-ish runs of 4+ chars to avoid trivial matches.
    """
    text = strip_thai_punct(fact)
    # Match Thai runs and Latin word runs of 4+ chars.
    candidates = re.findall(r"[฀-๿A-Za-z][฀-๿A-Za-z0-9_]{3,}", text)
    return [c for c in candidates if len(c) >= 4][:5]


def verify_record(rec: dict) -> dict:
    """Check sources and supporting facts. Returns a result dict."""
    sources = rec.get("sources", [])
    urls = [s.get("url", "") for s in sources if s.get("url", "").startswith("http")]
    facts = rec.get("verifier", {}).get("details", {}).get("required_supporting_facts", [])

    status_by_url: dict[str, int] = {}
    bodies: dict[str, str] = {}
    for url in urls:
        status, body = fetch_page(url)
        status_by_url[url] = status
        if status == 200:
            bodies[url] = body

    broken = [u for u, s in status_by_url.items() if s != 200]
    fact_matches: list[dict] = []
    for fact in facts:
        kws = fact_keywords(fact)
        if not kws:
            fact_matches.append({"fact": fact, "matched_url": None, "matched_keywords": []})
            continue
        # A fact "matches" a source if at least 2 of its keywords appear in the body.
        best_url = None
        best_hits: list[str] = []
        for url, body in bodies.items():
            hits = [kw for kw in kws if kw in body]
            if len(hits) > len(best_hits):
                best_hits = hits
                best_url = url
        fact_matches.append(
            {"fact": fact, "matched_url": best_url, "matched_keywords": best_hits}
        )

    weak_facts = [fm for fm in fact_matches if len(fm["matched_keywords"]) < 2]
    if broken:
        status = "FAIL"
    elif weak_facts:
        status = "WARN"
    else:
        status = "OK"

    return {
        "id": rec.get("id", "<no-id>"),
        "status": status,
        "broken_urls": broken,
        "fact_matches": fact_matches,
        "weak_facts": [fm["fact"] for fm in weak_facts],
    }


def write_report(results: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Fact verification report\n"]
    by_status = {"OK": 0, "WARN": 0, "FAIL": 0}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    lines.append(f"OK: {by_status.get('OK', 0)}  ·  WARN: {by_status.get('WARN', 0)}  ·  FAIL: {by_status.get('FAIL', 0)}  ·  TOTAL: {len(results)}")
    lines.append("")
    lines.append("| id | status | broken URLs | weak facts |")
    lines.append("|---|---|---|---|")
    for r in results:
        weak = "; ".join(r["weak_facts"])[:200] or "—"
        broken = ", ".join(r["broken_urls"])[:200] or "—"
        lines.append(f"| {r['id']} | {r['status']} | {broken} | {weak} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", required=True, help="JSONL file of records to verify")
    parser.add_argument("--out", default=str(REPORTS / "fact_verification.md"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    p = Path(args.check)
    results: list[dict] = []
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if args.limit and n >= args.limit:
            break
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("benchmark_family") != "hotpotqa_agentic":
            continue
        r = verify_record(rec)
        results.append(r)
        n += 1
        print(f"  [{r['status']}] {r['id']}  broken={len(r['broken_urls'])}  weak_facts={len(r['weak_facts'])}")

    write_report(results, Path(args.out))
    print(f"wrote {args.out}")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
