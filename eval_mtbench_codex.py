"""MT-Bench eval using codex CLI as judge (no API cost, unlimited).

Pipeline:
1. Sample N model responses per prompt (vLLM, temperature 0.7)
2. For each response, call codex CLI with a JSON-schema-constrained judge prompt
3. Take max of N scores per prompt (Best-of-N)
4. Report avg

Codex CLI:
    codex exec --skip-git-repo-check --ephemeral -s read-only \
        --output-schema schema.json --color never -c model_reasoning_effort=low <prompt>
"""
import argparse, json, os, re, subprocess, shutil, tempfile, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Codex CLI binary
CODEX = (shutil.which("codex") or shutil.which("codex.cmd")
         or os.path.expanduser("~/AppData/Roaming/npm/codex.cmd")
         or "/usr/local/bin/codex")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 10},
        "reasoning": {"type": "string"},
    },
    "required": ["score"]
}

def codex_judge(prompt: str, response: str, schema_path: str) -> float:
    """Use codex CLI to score 1-10."""
    judge_prompt = f"""Evaluate this AI response to a Thai prompt on a 1-10 scale.
Consider: helpfulness, accuracy, depth, Thai language fluency.

[Prompt (ภาษาไทย)]
{prompt[:1200]}

[AI Response]
{response[:2000]}

Output JSON: {{"score": <int 1-10>, "reasoning": "<1 sentence>"}}"""

    args = [CODEX, "exec", "--skip-git-repo-check", "--ephemeral",
            "-s", "read-only", "--output-schema", schema_path,
            "--color", "never", "-c", "model_reasoning_effort=\"low\"",
            judge_prompt]
    try:
        p = subprocess.run(args, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=120)
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return None
    # Extract JSON
    i = out.find('{"score"')
    if i < 0:
        i = out.find('{"reasoning"')  # alt order
    if i < 0: return None
    depth = 0; end = i
    for j in range(i, len(out)):
        if out[j] == '{': depth += 1
        elif out[j] == '}':
            depth -= 1
            if depth == 0: end = j+1; break
    try:
        data = json.loads(out[i:end])
        return float(data.get("score", 0))
    except: return None

def get_visible(t):
    if "</think>" in t:
        v = t.split("</think>",1)[1].strip()
        return v if len(v) >= 100 else t
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--judge-concurrency", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Schema file
    schema_path = "/tmp/judge_schema.json"
    Path(schema_path).write_text(json.dumps(JUDGE_SCHEMA), encoding="utf-8")

    print(f"[load] {args.base}")
    llm_kwargs = {"model": args.base, "tensor_parallel_size": 1,
                  "max_model_len": args.max_model_len, "dtype": "bfloat16",
                  "gpu_memory_utilization": 0.92}
    if args.adapter:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_loras"] = 1
        llm_kwargs["max_lora_rank"] = 16
    llm = LLM(**llm_kwargs)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    # MT-Bench
    for split in ["test", "train"]:
        try:
            ds = load_dataset("ThaiLLM-Leaderboard/mt-bench-thai", split=split); break
        except: continue
    items = []
    for r in ds:
        q = (r["turns"][0] if isinstance(r.get("turns"), list) and r["turns"] else
             r.get("question_1") or r.get("question") or r.get("instruction") or "")
        if q: items.append(q)
    items = items[:args.limit]
    print(f"[items] {len(items)} prompts × {args.n} samples = {len(items)*args.n} generations")

    prompts = []
    for q in items:
        msgs = [{"role":"user","content": q}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    print("[generating]")
    sp_kwargs = {"n": args.n, "temperature": args.temperature, "top_p": 0.95,
                 "max_tokens": args.max_tokens}
    if args.adapter:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("v5", 1, args.adapter)
        outputs = llm.generate(prompts, SamplingParams(**sp_kwargs), lora_request=lora_req)
    else:
        outputs = llm.generate(prompts, SamplingParams(**sp_kwargs))

    # Build judging tasks
    print(f"[judging] {len(items)*args.n} calls via codex (concurrency {args.judge_concurrency})")
    tasks = []
    for prompt_idx, (q, out) in enumerate(zip(items, outputs)):
        for cand_idx, o in enumerate(out.outputs):
            tasks.append((prompt_idx, cand_idx, q, get_visible(o.text)))

    scores_map = {}  # (prompt_idx, cand_idx) -> score
    def worker(task):
        pi, ci, q, c = task
        s = codex_judge(q, c, schema_path)
        return (pi, ci, s)

    completed = 0
    with ThreadPoolExecutor(max_workers=args.judge_concurrency) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for f in as_completed(futs):
            pi, ci, s = f.result()
            scores_map[(pi, ci)] = s
            completed += 1
            if completed % 10 == 0:
                print(f"  [judge] {completed}/{len(tasks)}")

    # Aggregate per prompt
    records = []
    all_max = []
    all_greedy = []
    for prompt_idx, (q, out) in enumerate(zip(items, outputs)):
        candidates = [get_visible(o.text) for o in out.outputs]
        scores = [scores_map.get((prompt_idx, ci)) or 0 for ci in range(len(candidates))]
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        max_score = scores[best_idx]
        greedy_score = scores[0]
        all_max.append(max_score)
        all_greedy.append(greedy_score)
        records.append({"prompt": q[:200], "best": candidates[best_idx][:500],
                        "best_score": max_score, "greedy_score": greedy_score,
                        "all_scores": scores})

    avg_max = sum(all_max) / max(len(all_max), 1)
    avg_greedy = sum(all_greedy) / max(len(all_greedy), 1)
    print(f"\n[mt_bench codex-judge] n={args.n} t={args.temperature}")
    print(f"  best-of-{args.n}: avg={avg_max:.2f}/10 → {avg_max*10:.1f}%")
    print(f"  first-sample:    avg={avg_greedy:.2f}/10 → {avg_greedy*10:.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "benchmark":"mt_bench_th_codex_bon", "n":args.n, "temperature":args.temperature,
        "accuracy_bon": avg_max/10, "avg_max": avg_max,
        "accuracy_greedy": avg_greedy/10, "avg_greedy": avg_greedy,
        "items": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {args.out}")

if __name__ == "__main__":
    main()
