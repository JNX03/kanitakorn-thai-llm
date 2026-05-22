"""Synthesize IFEval-style Thai instruction following with verifiable constraints.

Each record has: a Thai instruction + a deterministic constraint + a model-friendly
example response that satisfies the constraint.

Format aligned with eval_ifeval_mtbench.py constraint checker.
"""
import argparse, json, subprocess, threading, time, shutil, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
CODEX = (shutil.which("codex") or shutil.which("codex.cmd")
         or os.path.expanduser("~/AppData/Roaming/npm/codex.cmd"))

PROMPT = """สร้าง 8 ข้อ Thai instruction-following ที่มี constraint ตรวจสอบได้

แต่ละข้อต้องมี:
- คำสั่งภาษาไทย (เช่น "เขียนสรุปข่าวเทคโนโลยี")
- constraint ตรวจสอบได้: ขั้นต่ำ/มากสุดจำนวนคำ, ห้ามใช้เครื่องหมายจุลภาค, ต้องมี bullet points X จุด, ห้ามใช้คำที่กำหนด, ตอบเป็นตัวอักษรพิมพ์ใหญ่ทั้งหมด ฯลฯ
- ตัวอย่างคำตอบที่ตรงตาม constraint

constraint แต่ละข้อต้องตรงตาม IFEval format:
- length_constraints:number_words {relation: at least|at most, num_words: N}
- punctuation:no_comma {}
- detectable_format:number_bullet_lists {num_bullets: N}
- keywords:existence {keywords: [...]}
- keywords:forbidden_words {forbidden_words: [...]}
- change_case:english_capital {}

ส่งคืน JSON:
{
  "items": [
    {"instruction": "...", "constraint_id": "length_constraints:number_words",
     "kwargs": {"relation": "at least", "num_words": 300},
     "example_response": "..."},
    ...
  ]
}"""

SCHEMA = json.dumps({
    "type":"object","additionalProperties":False,
    "properties":{"items":{"type":"array","items":{
        "type":"object","additionalProperties":False,
        "properties":{
            "instruction":{"type":"string"},
            "constraint_id":{"type":"string"},
            "kwargs_json":{"type":"string","description":"JSON string of constraint kwargs"},
            "example_response":{"type":"string"}
        },"required":["instruction","constraint_id","kwargs_json","example_response"]
    }}},"required":["items"]
})

LOCK = threading.Lock()
STATS = {"got":0}

def call_codex(prompt: str) -> list:
    if not CODEX: return []
    schema_path = ROOT / "synth_ifeval_schema.json"
    schema_path.write_text(SCHEMA, encoding="utf-8")
    args = [CODEX,"exec","--skip-git-repo-check","--ephemeral","-s","read-only",
            "--output-schema",str(schema_path),"--color","never",
            "-c","model_reasoning_effort=\"low\"", prompt]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=300)
        text = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired: return []
    i = text.find('{"items"')
    if i < 0: return []
    depth = 0; end = i
    for j in range(i, len(text)):
        if text[j] == '{': depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0: end = j+1; break
    try:
        return json.loads(text[i:end]).get("items", [])
    except: return []

def worker(fh):
    items = call_codex(PROMPT)
    valid = []
    for it in items:
        ins = (it.get("instruction") or "").strip()
        cid = (it.get("constraint_id") or "").strip()
        resp = (it.get("example_response") or "").strip()
        kjs = (it.get("kwargs_json") or "{}").strip()
        if not ins or not cid or not resp: continue
        if not any('฀' <= c <= '๿' for c in ins[:500]): continue
        try:
            kwargs = json.loads(kjs)
        except: kwargs = {}
        valid.append({
            "messages": [
                {"role":"user","content": ins},
                {"role":"assistant","content": resp},
            ],
            "constraint": {"id": cid, "kwargs": kwargs}
        })
    with LOCK:
        for it in valid:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
            fh.flush()
        STATS["got"] += len(valid)
        print(f"  +{len(valid)} (total: {STATS['got']})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = Path(args.out).open("w", encoding="utf-8")
    n_calls = (args.target // 6) + 5
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, fh) for _ in range(n_calls)]
        for f in as_completed(futs):
            f.result()
            if STATS["got"] >= args.target: break
    fh.close()
    print(f"\n[done] {STATS['got']} records in {time.time()-t0:.0f}s -> {args.out}")

if __name__ == "__main__":
    main()
