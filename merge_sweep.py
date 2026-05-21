"""Task-arithmetic merge sweep for Stage C.

Given N LoRA adapter directories (all trained on same base), produces all
weighted-sum merges across an alpha grid, then optionally evaluates each
on top-3 benchmarks. Outputs a markdown ranking.

Usage:
    python3 merge_sweep.py \\
        --adapters base=runs/stage1 thai=runs/stage_thai math=runs/stage_math \\
        --grid 0.5,0.3,0.2 0.6,0.2,0.2 0.7,0.15,0.15 0.4,0.4,0.2 \\
        --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
        --eval-after \\
        --eval-base-cmd "python3 eval_self_consistency.py --base {base} --adapter {adapter} --benchmark {bench} --limit 100 --out {out}"
"""
import argparse, json, subprocess, os
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file

def parse_adapters(specs):
    """name=path → dict"""
    out = {}
    for s in specs:
        if "=" not in s: raise SystemExit(f"bad adapter spec: {s}")
        n, p = s.split("=", 1)
        out[n] = Path(p)
    return out

def parse_grid(grid_strs):
    """'0.5,0.3,0.2' → (0.5, 0.3, 0.2)"""
    return [tuple(float(x) for x in g.split(",")) for g in grid_strs]

def merge_adapters(adapters_paths, weights, out_dir):
    """Element-wise weighted sum."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sds = {n: load_file(p / "adapter_model.safetensors") for n, p in adapters_paths.items()}
    keys = list(next(iter(sds.values())).keys())
    # Verify all have same keys
    for n, sd in sds.items():
        if set(sd.keys()) != set(keys):
            raise SystemExit(f"key mismatch in {n}")
    names = list(adapters_paths.keys())
    merged = {}
    for k in keys:
        ts = [sds[n][k].float() for n in names]
        s = sum(w * t for w, t in zip(weights, ts))
        merged[k] = s.to(ts[0].dtype)
    save_file(merged, out_dir / "adapter_model.safetensors")
    # Copy config files from first adapter
    first = list(adapters_paths.values())[0]
    for fname in ("adapter_config.json","tokenizer.json","tokenizer_config.json",
                  "chat_template.jinja","training_args.bin","README.md","special_tokens_map.json"):
        src = first / fname
        if src.exists(): shutil.copy(src, out_dir / fname)
    return out_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True, help="name=path entries")
    ap.add_argument("--grid", nargs="+", required=True, help="Each a comma-separated weight tuple")
    ap.add_argument("--out-root", default="runs/merge_sweep")
    ap.add_argument("--base", default=None, help="HF base for eval")
    ap.add_argument("--eval-after", action="store_true")
    ap.add_argument("--benchmarks", nargs="+", default=["thaiexam", "math500", "aime24"])
    ap.add_argument("--limit-per-bench", type=int, default=100)
    args = ap.parse_args()

    adapters = parse_adapters(args.adapters)
    grid = parse_grid(args.grid)
    names = list(adapters.keys())
    if any(len(g) != len(names) for g in grid):
        raise SystemExit(f"grid tuples must have len {len(names)} = #adapters")
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for weights in grid:
        tag = "_".join(f"{n}{w:.2f}" for n, w in zip(names, weights))
        out_dir = out_root / tag
        print(f"[merge] {tag} -> {out_dir}")
        merge_adapters(adapters, weights, out_dir)

        if args.eval_after and args.base:
            scores = {}
            for bench in args.benchmarks:
                rj = out_dir / f"eval_{bench}.json"
                cmd = ["python3", "eval_self_consistency.py",
                       "--base", args.base, "--adapter", str(out_dir),
                       "--benchmark", bench, "--n", "1",
                       "--limit", str(args.limit_per_bench),
                       "--out", str(rj)]
                rc = subprocess.run(cmd).returncode
                if rc == 0 and rj.exists():
                    scores[bench] = json.loads(rj.read_text())["accuracy"]
                else:
                    scores[bench] = None
            results.append({"tag": tag, "weights": weights, "scores": scores})
            print(f"   scores: {scores}")
        else:
            results.append({"tag": tag, "weights": weights, "scores": None})

    # Markdown report
    report = ["# Merge Sweep Results\n"]
    hdr = "| Tag | " + " | ".join(args.benchmarks) + " |"
    report.append(hdr)
    report.append("|" + "---|" * (len(args.benchmarks) + 1))
    for r in results:
        row = f"| {r['tag']} |"
        for b in args.benchmarks:
            v = (r["scores"] or {}).get(b)
            row += f" {v:.3f} |" if v is not None else " — |"
        report.append(row)
    rp = out_root / "report.md"
    rp.write_text("\n".join(report), encoding="utf-8")
    print(f"\n=== report saved to {rp}")
    print("\n".join(report))

if __name__ == "__main__":
    main()
