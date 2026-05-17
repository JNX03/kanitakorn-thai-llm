"""Run a model over benchmark inputs to produce predictions for benchmark_eval.

Three model backends:

    --backend hf-local <hf-id>     transformers + accelerate (Qwen, Gemma, etc.)
    --backend openai <model>       OpenAI Chat Completions
    --backend anthropic <model>    Anthropic Messages

Pipeline:
    1) tools/benchmark_eval.py --inputs-only inputs.jsonl
    2) tools/run_inference.py --backend hf-local Qwen/Qwen2.5-7B-Instruct \\
            --inputs inputs.jsonl --out predictions.jsonl
    3) tools/benchmark_eval.py --score-from predictions.jsonl --model Qwen2.5-7B

Each output line carries the same fields as the input line plus a
`prediction` field with the model's response. Family-specific formatting is
applied via the per-family prompt builder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Backend:
    name: str
    model: str

    def generate(self, prompt: str, system: str = "") -> str:  # pragma: no cover
        raise NotImplementedError


class HFLocalBackend(Backend):
    def __init__(self, model: str, max_new_tokens: int = 1024) -> None:
        super().__init__("hf-local", model)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            import torch  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "hf-local backend requires `transformers` and `torch` — install or use --backend openai"
            ) from e
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model_obj = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="auto")
        self.torch = torch
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model_obj.device)
        with self.torch.no_grad():
            out = self.model_obj.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tok.eos_token_id,
            )
        full = self.tok.decode(out[0], skip_special_tokens=True)
        # Strip the prompt prefix.
        if full.startswith(text):
            return full[len(text):].strip()
        return full.strip()


class OpenAIBackend(Backend):
    def __init__(self, model: str) -> None:
        super().__init__("openai", model)
        from openai import OpenAI  # type: ignore
        self.client = OpenAI()

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=messages,
        )
        return (resp.choices[0].message.content or "").strip()


class AnthropicBackend(Backend):
    def __init__(self, model: str) -> None:
        super().__init__("anthropic", model)
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError("anthropic SDK required for --backend anthropic") from e
        self.client = Anthropic()

    def generate(self, prompt: str, system: str = "") -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        out = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return out.strip()


FAMILY_SYSTEM_PROMPTS = {
    "aime24": "You are a math expert. Solve this AIME-style problem. Show concise steps. End with `\\boxed{ANSWER}`.",
    "aime25": "You are a math expert. Solve this AIME-style problem. Show concise steps. End with `\\boxed{ANSWER}`.",
    "math500": "You are a math expert. Solve this MATH500 problem. Show concise steps. End with `\\boxed{ANSWER}`.",
    "ifeval": "Follow the user's instruction precisely. Output ONLY the response — no preamble, no explanation.",
    "mt_bench": "You are a helpful assistant. Answer in Thai if the question is in Thai, in English otherwise.",
    "openthaieval": "You are a Thai academic-exam expert. Read the question carefully and answer with the single letter (1), (2), (3), or (4).",
    "livecodebench": "You are a competitive programmer. Read the problem and output a complete Python solution. Output ONLY code — wrap in ```python ... ```.",
    "hotpotqa": "You are a fact-finder. Answer concisely with the single named entity / number / date. Cite the supporting passages used.",
}


def make_backend(name: str, model: str) -> Backend:
    if name == "hf-local":
        return HFLocalBackend(model)
    if name == "openai":
        return OpenAIBackend(model)
    if name == "anthropic":
        return AnthropicBackend(model)
    raise SystemExit(f"unknown backend: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["hf-local", "openai", "anthropic"])
    parser.add_argument("--model", required=True, help="HF model id, or API model name")
    parser.add_argument("--inputs", required=True, help="JSONL from benchmark_eval --inputs-only")
    parser.add_argument("--out", required=True, help="JSONL for benchmark_eval --score-from")
    parser.add_argument("--limit", type=int, default=None, help="optional cap on number of inputs")
    args = parser.parse_args()

    if args.backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set")
        return 2
    if args.backend == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set")
        return 2

    backend = make_backend(args.backend, args.model)
    inputs_path = Path(args.inputs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_failed = 0
    t_start = time.time()
    with out_path.open("w", encoding="utf-8") as fout:
        for line in inputs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if args.limit and n_done >= args.limit:
                break
            rec = json.loads(line)
            family = rec.get("family", "")
            system = FAMILY_SYSTEM_PROMPTS.get(family, "")
            prompt = rec.get("prompt", "")
            try:
                rec["prediction"] = backend.generate(prompt, system=system)
            except Exception as e:
                rec["prediction"] = ""
                rec["error"] = f"{type(e).__name__}: {e}"
                n_failed += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 25 == 0:
                rate = n_done / (time.time() - t_start)
                print(f"  {n_done} done, {n_failed} failed, {rate:.1f}/sec")

    print(f"inference complete: {n_done} predictions written to {out_path} ({n_failed} errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
