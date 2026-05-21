"""Day-0 base model bake-off. Train each candidate base on a 500-record subset
for 1 epoch (~30 min each on 2x A100), then run a quick eval suite of 100-record
samples per benchmark family. Pick winner by aggregate Δ-to-target.

Usage:
    python3 ab_base_compare.py --bases Qwen/Qwen3-8B-Base ThaiLLM/ThaiLLM-8B
    python3 ab_base_compare.py --bases-from-list bake_off_bases.txt

Output: reports/base_bakeoff_<timestamp>.md with per-base score across all
benchmark families and final ranking.
"""
import argparse, json, os, subprocess, time, sys
from pathlib import Path

CANDIDATES = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",   # MATH 93.9 / AIME 69.7 prebaked
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",    # MATH 92.8 / AIME 55.5 prebaked, VRAM-safe
    "Qwen/Qwen3-8B-Base",
    "ThaiLLM/ThaiLLM-8B",
    "Qwen/Qwen2.5-Math-7B-Instruct",
]

# Targets — used for ranking
TARGETS = {
    "aime24_th": 15, "aime24": 25,
    "math500_th": 56, "math500": 82,
    "livecodebench_th": 35, "livecodebench": 60,
    "openthaieval": 80, "hotpotqa": 46,
    "ifeval": 57, "mt_bench": 85,
    "thaiexam": 75, "ifeval_th": 82,
}

def train_one(base: str, out: Path, subset_n: int = 500, epochs: int = 1):
    """Quick 1-epoch SFT on a small subset."""
    cmd = [
        "python3", "train_2xa100_interruptible.py",
    ]
    env = os.environ.copy()
    env.update({
        "SFT_BASE": base,
        "SFT_OUT": str(out),
        "SFT_EPOCHS": str(epochs),
        "SFT_LORA_R": "16",
        "SFT_MIX_EN": "0",  # pure Thai for bake-off, faster
        "AB_SUBSET_N": str(subset_n),
    })
    print(f"[ab] training {base} -> {out}")
    t0 = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    elapsed = time.time() - t0
    print(f"[ab] {base} done in {elapsed/60:.1f} min, rc={rc}")
    return rc == 0

def eval_one(base: str, adapter: Path, benchmarks: list, limit_per_bench: int = 100):
    """Run quick eval suite."""
    results = {}
    for bench in benchmarks:
        out_json = adapter / f"eval_{bench}.json"
        cmd = [
            "python3", "eval_self_consistency.py",
            "--base", base, "--adapter", str(adapter / "final"),
            "--benchmark", bench, "--n", "1",
            "--limit", str(limit_per_bench),
            "--out", str(out_json),
        ]
        rc = subprocess.run(cmd).returncode
        if rc == 0 and out_json.exists():
            results[bench] = json.loads(out_json.read_text(encoding="utf-8"))["accuracy"]
        else:
            results[bench] = None
    return results

def rank(all_results: dict) -> list:
    """Return list of (base, total_norm_score) sorted desc.
    norm_score per benchmark = score / target. Sum across benchmarks."""
    rankings = []
    for base, results in all_results.items():
        total = 0.0
        n = 0
        for bench, acc in results.items():
            if acc is None: continue
            target = TARGETS.get(bench, 1.0)
            # If target is in 0-100 scale, divide; if 0-1 scale, multiply
            target_norm = target / 100 if target > 1 else target
            total += acc / target_norm
            n += 1
        avg = total / max(n, 1)
        rankings.append((base, avg, n))
    rankings.sort(key=lambda x: -x[1])
    return rankings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=CANDIDATES)
    ap.add_argument("--benchmarks", nargs="+", default=["thaiexam", "math500", "aime24"])
    ap.add_argument("--subset-n", type=int, default=500)
    ap.add_argument("--limit-per-bench", type=int, default=100)
    ap.add_argument("--out-dir", default="runs/bakeoff")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_all = {}
    timing = {}

    for base in args.bases:
        slug = base.replace("/", "_")
        adapter_dir = out_dir / slug
        t0 = time.time()
        ok = train_one(base, adapter_dir, args.subset_n, epochs=1)
        if not ok:
            print(f"[ab] {base} TRAINING FAILED, skipping")
            results_all[base] = {b: None for b in args.benchmarks}
            continue
        results = eval_one(base, adapter_dir, args.benchmarks, args.limit_per_bench)
        results_all[base] = results
        timing[base] = time.time() - t0

    # Rank
    ranking = rank(results_all)
    report_path = out_dir / f"bakeoff_report_{int(time.time())}.md"
    lines = ["# Base Model Bake-off Results", ""]
    lines.append("## Per-base scores")
    lines.append("")
    hdr = "| Base | " + " | ".join(args.benchmarks) + " | Avg norm |"
    sep = "|" + "---|" * (len(args.benchmarks) + 2)
    lines.append(hdr)
    lines.append(sep)
    for base, score, n in ranking:
        row = f"| {base} |"
        for b in args.benchmarks:
            v = results_all[base].get(b)
            row += f" {v:.3f} |" if v is not None else " — |"
        row += f" **{score:.3f}** |"
        lines.append(row)
    lines.append("")
    lines.append("## Ranking (descending norm score)")
    for i, (base, score, n) in enumerate(ranking, 1):
        lines.append(f"{i}. **{base}** — {score:.3f} (over {n} benchmarks)")
    lines.append("")
    lines.append(f"## Winner: `{ranking[0][0]}`")
    lines.append("")
    lines.append(f"Set `SFT_BASE={ranking[0][0]}` for full Stage 1.")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n=== Report written to {report_path}")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
