"""Web-search verifier for facts the LLM-judge alone cannot ground.

The user explicitly asked: "allow it to be search every data and make sure
everything is correct." Standard LLM-judge calls grade by pattern; this
verifier additionally fetches external evidence to check claims, especially
for HotpotQA-style multi-hop answers and MATH problems where an answer key
can be cross-checked against a published source.

Backends (auto-detected, in order):
    1. Tavily Search API           (if TAVILY_API_KEY set)
    2. Brave Search API            (if BRAVE_API_KEY set)
    3. DuckDuckGo HTML scrape      (no key — fallback, may rate-limit)

CLI:
    python tools/web_search_verifier.py --query "AIME 2024 problem 1 answer"
    python tools/web_search_verifier.py --check-record dataset/train/train_aime_th_001.jsonl

Each accepted record's `final_answer` and `concise_solution` are extracted; the
verifier runs a search like `<problem statement> answer` and reports whether
the top results agree with the claimed answer. False positives ARE possible —
this is a soft signal, not a hard gate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import urllib.parse
    import urllib.request
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("urllib stdlib required") from exc


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


def _tavily_search(query: str, k: int = 5) -> list[SearchHit]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    import urllib.request as req
    body = json.dumps({"api_key": api_key, "query": query, "max_results": k, "search_depth": "basic"}).encode("utf-8")
    r = req.Request(
        "https://api.tavily.com/search",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with req.urlopen(r, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [SearchHit(h.get("title", ""), h.get("url", ""), h.get("content", "")) for h in data.get("results", [])]


def _brave_search(query: str, k: int = 5) -> list[SearchHit]:
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return []
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": query, "count": k})
    r = urllib.request.Request(url, headers={"X-Subscription-Token": api_key, "Accept": "application/json"})
    with urllib.request.urlopen(r, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = []
    for h in data.get("web", {}).get("results", [])[:k]:
        out.append(SearchHit(h.get("title", ""), h.get("url", ""), h.get("description", "")))
    return out


def _ddg_search(query: str, k: int = 5) -> list[SearchHit]:
    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    r = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return []
    # Crude scrape: <a class="result__a" href="URL">TITLE</a> ... <a class="result__snippet">SNIPPET</a>
    hits: list[SearchHit] = []
    for m in re.finditer(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        flags=re.DOTALL,
    ):
        url_, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        hits.append(SearchHit(title=title, url=url_, snippet=""))
        if len(hits) >= k:
            break
    return hits


def web_search(query: str, k: int = 5) -> list[SearchHit]:
    for backend in (_tavily_search, _brave_search, _ddg_search):
        try:
            hits = backend(query, k)
        except Exception:
            hits = []
        if hits:
            return hits
    return []


def verify_answer_against_web(question: str, claimed_answer: str, max_results: int = 5) -> dict:
    """Soft check: search for the question, see if any result contains the claimed answer.

    Returns:
        {
            "verified": bool,
            "evidence": list[SearchHit],
            "method": "web_search",
            "claimed_answer": str,
        }

    `verified=True` means at least one search snippet/title literally contains
    the claimed answer string. Numbers and short answers usually appear
    verbatim in summaries. Long-form answers may not — for those, escalate to
    the LLM-judge after the web pull.
    """
    hits = web_search(f"{question} answer", k=max_results)
    normalized_answer = re.sub(r"\s+", " ", claimed_answer.strip().lower())
    matches: list[SearchHit] = []
    for hit in hits:
        haystack = (hit.title + " " + hit.snippet).lower()
        if normalized_answer and normalized_answer in haystack:
            matches.append(hit)
    return {
        "verified": bool(matches),
        "evidence": [{"title": h.title, "url": h.url, "snippet": h.snippet[:200]} for h in matches] or [
            {"title": h.title, "url": h.url, "snippet": h.snippet[:200]} for h in hits[:max_results]
        ],
        "method": "web_search",
        "claimed_answer": claimed_answer,
    }


def check_record_file(path: Path, limit: int) -> dict:
    """Run the search verifier on the first `limit` records in a shard."""
    stats = {"checked": 0, "verified": 0, "unverified_with_evidence": 0, "no_evidence": 0}
    audit_lines: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if stats["checked"] >= limit:
            break
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["benchmark_family"] not in {"aime_th", "math500_th", "openthaieval", "hotpotqa_agentic"}:
            continue
        question = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
        answer = rec["final_answer"]
        result = verify_answer_against_web(question, answer, max_results=5)
        stats["checked"] += 1
        if result["verified"]:
            stats["verified"] += 1
        elif result["evidence"]:
            stats["unverified_with_evidence"] += 1
        else:
            stats["no_evidence"] += 1
        audit_lines.append({"id": rec["id"], **result})
    return {"stats": stats, "audit": audit_lines}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="ad-hoc search query")
    parser.add_argument("--check-record", help="JSONL path to spot-check (first N records)")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-results", type=int, default=5)
    args = parser.parse_args()

    if args.query:
        hits = web_search(args.query, k=args.max_results)
        for h in hits:
            print(f"- [{h.title}]({h.url})\n  {h.snippet[:200]}")
        if not hits:
            print("no results (or all backends unavailable)")
        return 0

    if args.check_record:
        out = check_record_file(Path(args.check_record), args.limit)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print("specify --query <text> or --check-record <jsonl-path>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
