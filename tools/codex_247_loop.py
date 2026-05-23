"""24/7 autonomous codex loop: synth → analyze failures → targeted synth → repeat.

Workflow each cycle (every ~15 min):
  1. Generate a batch of Thai exam records (rotating topics)
  2. If a fresh eval report exists, analyze worst-category failures
  3. Generate MORE records in that worst category
  4. Append to running corpus
  5. Sleep 60s, repeat

Run with:
    nohup python3 tools/codex_247_loop.py > /tmp/codex_loop.log 2>&1 &
"""
import argparse, json, subprocess, threading, time, shutil, os, random, glob
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
CODEX = (shutil.which("codex") or shutil.which("codex.cmd")
         or os.path.expanduser("~/AppData/Roaming/npm/codex.cmd"))

STATE_FILE = ROOT / ".codex_loop_state.json"
OUT_DIR = ROOT / "dataset/train"
ANALYSIS_DIR = ROOT / "dataset/analysis"
LOG_DIR = ROOT / "logs/codex_loop"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TOPICS = {
    "literature": "วรรณคดีไทย — สุนทรภู่ พระอภัยมณี รามเกียรติ์ ขุนช้างขุนแผน อิเหนา ลิลิตพระลอ",
    "prosody": "ฉันทลักษณ์ไทย — กลอนสุภาพ ฉันท์ โคลง กาพย์ยานี ๑๑ สูตรเอกโท การส่งสัมผัส",
    "grammar": "ไวยากรณ์ไทย — คำลักษณนาม คำราชาศัพท์ คำซ้อน คำประสม การใช้บุพบท",
    "history_sukhothai": "ประวัติศาสตร์ไทยสมัยสุโขทัย — พ่อขุนรามคำแหง ลายสือไทย เศรษฐกิจ การปกครอง",
    "history_ayutthaya": "ประวัติศาสตร์ไทยสมัยอยุธยา — รัชกาล สงครามพม่า การค้าต่างประเทศ",
    "history_ratta": "ประวัติศาสตร์ไทยรัตนโกสินทร์ — รัชกาลที่ 1-10 ตำแหน่ง นโยบาย",
    "buddhism": "พุทธศาสนา — อริยสัจ 4 ปฏิจจสมุปบาท พุทธประวัติ พระไตรปิฎก",
    "investment_basics": "การลงทุนพื้นฐาน — หุ้น พันธบัตร กองทุน NPV IRR ความเสี่ยง",
    "investment_advanced": "การลงทุนขั้นสูง — Beta ตราสารอนุพันธ์ ภาษีการลงทุน Single License",
    "physics": "ฟิสิกส์ ม.6 — กลศาสตร์ ไฟฟ้า แสง คลื่น พลังงาน",
    "chemistry": "เคมี ม.6 — อะตอม พันธะเคมี สารละลาย สมดุล กรดเบส",
    "biology": "ชีววิทยา ม.6 — เซลล์ พันธุกรรม วิวัฒนาการ ระบบนิเวศ",
    "math_advanced": "คณิตศาสตร์ ม.6 — แคลคูลัส ความน่าจะเป็น เมทริกซ์ พีชคณิตเชิงเส้น",
    "geography": "ภูมิศาสตร์ไทย — ภาคต่างๆ จังหวัด สภาพภูมิอากาศ ทรัพยากร",
    "english": "ภาษาอังกฤษระดับสูง — grammar reading comprehension vocabulary",
    "social_studies": "สังคมศึกษา — รัฐศาสตร์ เศรษฐศาสตร์ กฎหมาย",
}

PROMPT_TEMPLATE = """สร้างข้อสอบไทย MCQ 6 ข้อ ในหัวข้อ: {topic}
ระดับยาก (O-NET/A-Level/Investment License)

ข้อกำหนดเข้มงวด:
1. คำถามภาษาไทยล้วน
2. ตัวเลือก 5 ตัวเลือก (a-e)
3. chain_of_thought — เหตุผลภาษาไทยล้วนอย่างน้อย 150 คำ ทำตามขั้นตอน: อ่านโจทย์, วิเคราะห์ตัวเลือก, ตัดทิ้งตัวเลือกผิด, สรุปคำตอบ
4. คำตอบ "1"-"5"

ส่งคืน JSON: {{ "items": [{{"question":"...","choices":[...],"chain_of_thought":"...","answer":"3"}}] }}"""

SCHEMA = json.dumps({
    "type":"object","additionalProperties":False,
    "properties":{"items":{"type":"array","items":{
        "type":"object","additionalProperties":False,
        "properties":{
            "question":{"type":"string"},"choices":{"type":"array","items":{"type":"string"},"minItems":5,"maxItems":5},
            "chain_of_thought":{"type":"string"},
            "answer":{"type":"string","enum":["1","2","3","4","5"]}
        },"required":["question","choices","chain_of_thought","answer"]
    }}},"required":["items"]
})


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"cycle": 0, "total_generated": 0, "category_counts": {}, "last_analysis": None}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def call_codex(prompt: str, timeout=300):
    schema_path = ROOT / "loop_schema.json"
    schema_path.write_text(SCHEMA, encoding="utf-8")
    args = [CODEX, "exec", "--skip-git-repo-check", "--ephemeral",
            "-s", "read-only", "--output-schema", str(schema_path),
            "--color", "never", "-c", "model_reasoning_effort=\"medium\"", prompt]
    try:
        p = subprocess.run(args, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=timeout)
        text = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return []
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
    except:
        return []


def validate(it):
    q = (it.get("question") or "").strip()
    choices = it.get("choices") or []
    cot = (it.get("chain_of_thought") or "").strip()
    ans = str(it.get("answer","")).strip()
    if not q or len(choices) != 5 or not cot or ans not in {"1","2","3","4","5"}: return None
    thai_ratio = sum(1 for c in cot if '฀' <= c <= '๿') / max(len(cot), 1)
    if thai_ratio < 0.5: return None
    letter = chr(ord('a') + int(ans) - 1)
    user = q + "\n\nตัวเลือก:\n" + "\n".join(f"({chr(97+i)}) {c}" for i, c in enumerate(choices))
    user += "\n\nคิดและให้เหตุผลเป็นภาษาไทย แล้วระบุคำตอบในรูปแบบ คำตอบคือ (X)"
    assistant = cot + f"\n\nคำตอบคือ ({letter})"
    return {"messages":[
        {"role":"user","content": user},
        {"role":"assistant","content": assistant},
    ]}


def analyze_latest_eval():
    """Find most recent eval report and identify weakest category."""
    candidates = sorted(glob.glob(str(ROOT / "reports/v*_thaiexam_full.json")), reverse=True)
    if not candidates:
        candidates = sorted(glob.glob(str(ROOT / "reports/v*_thaiexam.json")), reverse=True)
    if not candidates: return None
    path = candidates[0]
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except:
        return None
    items = d.get("items", [])
    if not items: return None
    cat_correct = {}; cat_total = {}
    for i in items:
        samples = i.get("samples", [])
        s = samples[0][:500] if samples else ""
        cat = "other"
        for key, kw_list in [
            ("literature", ["ฉัน","พระอภัย","สุนทร","รามเกียรติ","โคลง","กลอน"]),
            ("history", ["ประวัติ","สุโขทัย","อยุธยา","รัชกาล","สมัย"]),
            ("investment", ["ลงทุน","หุ้น","การเงิน","เศรษฐ","NPV","พันธบัตร"]),
            ("grammar", ["ไวยากรณ์","คำกริยา","ลักษณนาม","ราชาศัพท์","คำซ้อน","คำประสม"]),
            ("physics", ["ฟิสิกส์","คลื่น","แม่เหล็ก","โมเมนต์","พลังงาน"]),
            ("chemistry", ["เคมี","โมล","อะตอม","พันธะ","สารละลาย"]),
            ("biology", ["ชีว","เซลล์","พันธุกรรม"]),
            ("buddhism", ["พุทธ","อริยสัจ","นิพพาน","พระไตรปิฎก"]),
        ]:
            if any(k in s for k in kw_list):
                cat = key; break
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if i.get("correct"): cat_correct[cat] = cat_correct.get(cat, 0) + 1
    weakest = []
    for cat, total in cat_total.items():
        if total < 3: continue
        acc = cat_correct.get(cat, 0) / total
        weakest.append((cat, acc, total))
    weakest.sort(key=lambda x: x[1])
    return {"report": path, "weakest": weakest[:5], "all": [(c, cat_correct.get(c,0)/t, t) for c, t in cat_total.items()]}


def cycle(state):
    state["cycle"] += 1
    cycle_n = state["cycle"]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"cycle_{cycle_n:04d}_{timestamp}.log"

    # Analyze recent eval (if any)
    analysis = analyze_latest_eval()
    if analysis:
        analysis_path = ANALYSIS_DIR / f"analysis_cycle_{cycle_n:04d}.json"
        analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
        state["last_analysis"] = str(analysis_path)
        weak_cats = [w[0] for w in analysis["weakest"]]
        with log_file.open("a", encoding="utf-8") as lf:
            lf.write(f"[{timestamp}] cycle={cycle_n} weakest={analysis['weakest']}\n")
    else:
        weak_cats = []

    # Pick topics: 70% weak + 30% random rotation
    chosen = []
    for wc in weak_cats[:3]:
        for k in TOPICS:
            if wc in k or wc == k.split("_")[0]:
                chosen.append(k)
                break
    while len(chosen) < 6:
        chosen.append(random.choice(list(TOPICS.keys())))
    chosen = chosen[:6]

    # Generate
    out_file = OUT_DIR / f"train_codex_loop_{cycle_n:04d}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    lock = threading.Lock()
    with out_file.open("w", encoding="utf-8") as fh:
        def task(topic_key):
            items = call_codex(PROMPT_TEMPLATE.format(topic=TOPICS[topic_key]))
            valid = [validate(it) for it in items]
            valid = [v for v in valid if v]
            with lock:
                for v in valid:
                    fh.write(json.dumps(v, ensure_ascii=False) + "\n")
                state["category_counts"][topic_key] = state["category_counts"].get(topic_key, 0) + len(valid)
                return len(valid)

        n_calls = max(8, len(chosen) * 2)
        topics_for_calls = [chosen[i % len(chosen)] for i in range(n_calls)]
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(task, t) for t in topics_for_calls]
            for f in as_completed(futs):
                try: written += f.result()
                except: pass

    state["total_generated"] += written
    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"[{timestamp}] cycle {cycle_n}: wrote {written} -> {out_file} (total {state['total_generated']})\n")
    save_state(state)
    print(f"[{timestamp}] cycle {cycle_n}: +{written} (total {state['total_generated']}) -> {out_file.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cycles", type=int, default=10000)
    ap.add_argument("--sleep-secs", type=int, default=60)
    args = ap.parse_args()

    state = load_state()
    print(f"[loop] starting from cycle {state['cycle']}, total_generated={state['total_generated']}")
    while state["cycle"] < args.max_cycles:
        try:
            cycle(state)
        except KeyboardInterrupt:
            print("interrupted"); break
        except Exception as e:
            print(f"[loop] error in cycle: {e}")
        time.sleep(args.sleep_secs)


if __name__ == "__main__":
    main()
