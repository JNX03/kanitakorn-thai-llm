"""Re-judge saved MT-Bench responses using codex CLI (unlimited)."""
import json, subprocess, shutil, os, time, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CODEX = (shutil.which("codex") or shutil.which("codex.cmd")
         or os.path.expanduser("~/AppData/Roaming/npm/codex.cmd"))

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 10}},
    "required": ["score"]
}
SCHEMA_PATH = "C:/Users/Jnx03/Desktop/kanitakornv2/mt_bench_schema.json"
Path(SCHEMA_PATH).write_text(json.dumps(JUDGE_SCHEMA), encoding="utf-8")

def judge(prompt, response):
    p = f"""Score this AI response to a Thai prompt (1-10): helpfulness, accuracy, depth, Thai fluency.

[Prompt]
{prompt[:1200]}

[Response]
{response[:2000]}

Output JSON: {{"score": <1-10 int>}}"""
    args = [CODEX, "exec", "--skip-git-repo-check", "--ephemeral",
            "-s", "read-only", "--output-schema", SCHEMA_PATH,
            "--color", "never", "-c", "model_reasoning_effort=\"low\"", p]
    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=120)
        out = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return None
    i = out.find('{"score"')
    if i < 0: return None
    depth = 0; end = i
    for j in range(i, len(out)):
        if out[j] == '{': depth += 1
        elif out[j] == '}':
            depth -= 1
            if depth == 0: end = j+1; break
    try:
        return float(json.loads(out[i:end]).get("score", 0))
    except: return None

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Jnx03/Desktop/kanitakornv2/reports/qwen3_mt_bench_th_remote.json"
    d = json.load(open(src, encoding="utf-8"))
    items = d.get("items", [])
    print(f"Re-judging {len(items)} responses with codex")

    def worker(idx_item):
        idx, it = idx_item
        s = judge(it.get("prompt",""), it.get("response",""))
        return idx, s

    new_scores = [None] * len(items)
    completed = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(worker, (i, it)) for i, it in enumerate(items)]
        for f in as_completed(futs):
            idx, s = f.result()
            new_scores[idx] = s
            completed += 1
            if completed % 10 == 0: print(f"  {completed}/{len(items)}")

    valid = [s for s in new_scores if s is not None]
    avg_old = sum(it.get("score", 0) or 0 for it in items) / max(len(items), 1)
    avg_new = sum(valid) / max(len(valid), 1)
    print(f"\nOld (gemini): avg={avg_old:.2f}/10 → {avg_old*10:.1f}%")
    print(f"New (codex):  avg={avg_new:.2f}/10 → {avg_new*10:.1f}%  ({len(valid)} valid)")

    out = Path(src).with_name(Path(src).stem + "_codexjudged.json")
    for it, s in zip(items, new_scores):
        it["score_codex"] = s
    d["avg_score_codex"] = avg_new
    out.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out}")

if __name__ == "__main__":
    main()
