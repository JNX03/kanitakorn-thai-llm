"""Rebuild SFT records using the target base model's chat template.

The existing dataset/sft_ready/*_train.jsonl files were baked with Qwen's
<|im_start|>...<|im_end|> format. But R1-Distill-Qwen-14B and other bases
use different chat templates (DeepSeek: <｜User｜>...<｜Assistant｜>).

This script:
  1. Parses the existing Qwen-formatted text to extract user/assistant messages
  2. Applies the target base's chat template
  3. Writes new SFT-ready files alongside the originals

Usage:
    python3 tools/rebuild_for_base.py \\
        --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
        --in-glob "dataset/sft_ready/*_train.jsonl" \\
        --out-dir dataset/sft_ready_r1d14b/
"""
import argparse, json, re, glob
from pathlib import Path

from transformers import AutoTokenizer

def parse_qwen_text_to_messages(text: str):
    """Extract messages list from Qwen-format <|im_start|>role\\ncontent<|im_end|>."""
    msgs = []
    # Pattern: <|im_start|>role\n CONTENT <|im_end|>
    pattern = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)
    for role, content in pattern.findall(text):
        msgs.append({"role": role, "content": content.strip()})
    return msgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--in-glob", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include-distill", action="store_true", help="Also merge in dataset/train/train_distill_r1_*.jsonl as additional rows")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if not getattr(tok, "chat_template", None):
        raise SystemExit(f"Base {args.base} has no chat_template")
    print(f"[base] {args.base}")
    print(f"[template starts] {tok.chat_template[:200]!r}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n_total = 0; n_kept = 0
    for fp in sorted(glob.glob(args.in_glob)):
        fp = Path(fp)
        out_fp = out_dir / fp.name
        out_fh = out_fp.open("w", encoding="utf-8")
        cnt = 0
        for line in fp.open(encoding="utf-8"):
            if not line.strip(): continue
            n_total += 1
            r = json.loads(line)
            text = r.get("text", "")
            msgs = parse_qwen_text_to_messages(text)
            if not msgs or len(msgs) < 2: continue
            try:
                new_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception as e:
                continue
            if not new_text: continue
            n_kept += 1; cnt += 1
            r["text"] = new_text
            r["messages"] = msgs  # also keep messages for flexibility
            out_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        out_fh.close()
        print(f"  {fp.name} -> {cnt} records")

    # Optional: merge distillation files
    if args.include_distill:
        distill_pat = Path("dataset/train") / "train_distill_r1_*.jsonl"
        for fp in sorted(glob.glob(str(distill_pat))):
            fp = Path(fp)
            family = "aime_th" if "aime" in fp.name else "math500_th"
            out_fp = out_dir / f"{family}_train.jsonl"
            cnt = 0
            with out_fp.open("a", encoding="utf-8") as out_fh:
                for line in fp.open(encoding="utf-8"):
                    if not line.strip(): continue
                    r = json.loads(line)
                    msgs = r.get("messages", [])
                    if len(msgs) < 2: continue
                    try:
                        new_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                    except: continue
                    out_fh.write(json.dumps({
                        "id": r.get("id"),
                        "family": family,
                        "split": "train",
                        "difficulty": "olympiad" if family == "aime_th" else "competition",
                        "language": r.get("language", "th"),
                        "text": new_text,
                        "messages": msgs,
                        "source": "distill_r1",
                    }, ensure_ascii=False) + "\n")
                    cnt += 1; n_kept += 1
            print(f"  [distill] {fp.name} -> +{cnt} merged into {out_fp.name}")

    # Write a new manifest pointing at out_dir
    src_manifest = Path("dataset/sft_ready/manifest.json")
    if src_manifest.exists():
        m = json.loads(src_manifest.read_text(encoding="utf-8"))
        # Update file paths
        for fam, paths in m["sft_files"].items():
            for split in ("train", "validation"):
                if split in paths:
                    name = Path(paths[split]).name
                    paths[split] = f"{out_dir.name}/{name}" if out_dir.parent.name == "dataset" else str(out_dir / name)
        new_manifest = out_dir / "manifest.json"
        new_manifest.write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"[manifest] new manifest at {new_manifest}")

    print(f"\n[done] kept {n_kept}/{n_total} records ({100*n_kept/max(n_total,1):.1f}%)")

if __name__ == "__main__":
    main()
