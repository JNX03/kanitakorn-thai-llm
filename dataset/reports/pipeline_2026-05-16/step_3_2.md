# phase 3.2: build_train_manifest
## cmd
C:\Python313\python.exe tools/build_train_manifest.py
## stdout
{
  "strategy": "sqrt",
  "train_counts": {
    "aime_th": 1161,
    "hotpotqa_agentic": 25,
    "ifeval_ifbench": 619,
    "livecodebench_th": 619,
    "math500_th": 620,
    "mt_bench": 79,
    "openthaieval": 618
  },
  "weights": {
    "aime_th": 0.231037,
    "hotpotqa_agentic": 0.033903,
    "ifeval_ifbench": 0.168698,
    "livecodebench_th": 0.168698,
    "math500_th": 0.168835,
    "mt_bench": 0.060267,
    "openthaieval": 0.168562
  },
  "sft_files": {
    "aime_th": {
      "train": "sft_ready/aime_th_train.jsonl",
      "validation": "sft_ready/aime_th_validation.jsonl"
    },
    "hotpotqa_agentic": {
      "train": "sft_ready/hotpotqa_agentic_train.jsonl",
      "validation": "sft_ready/hotpotqa_agentic_validation.jsonl"
    },
    "ifeval_ifbench": {
      "train": "sft_ready/ifeval_ifbench_train.jsonl",
      "validation": "sft_ready/ifeval_ifbench_validation.jsonl"
    },
    "livecodebench_th": {
      "train": "sft_ready/livecodebench_th_train.jsonl",
      "validation": "sft_ready/livecodebench_th_validation.jsonl"
    },
    "math500_th": {
      "train": "sft_ready/math500_th_train.jsonl",
      "validation": "sft_ready/math500_th_validation.jsonl"
    },
    "mt_bench": {
      "train": "sft_ready/mt_bench_train.jsonl",
      "validation": "sft_ready/mt_bench_validation.jsonl"
    },
    "openthaieval": {
      "train": "sft_ready/openthaieval_train.jsonl",
      "validation": "sft_ready/openthaieval_validation.jsonl"
    }
  }
}
weight sum = 1.000000

## stderr

## returncode
0
