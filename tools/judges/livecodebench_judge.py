"""LiveCodeBench-TH judge.

Primary scorer: sandboxed unit-test execution. The LLM judge handles only the
"explain the runtime error" subtasks from the LCB v6 paper (arxiv 2507.02833).

Sandbox: subprocess with timeout + memory cap. Not a full security boundary;
for adversarial code, run inside Docker or vercel:vercel-sandbox.

Used by:
    * tools/benchmark_eval.py — runs against typhoon-ai/livecodebench-th
"""

from __future__ import annotations

import json
import subprocess
try:
    import resource  # POSIX-only; on Windows we fall back to timeout-only sandboxing.
except ImportError:  # pragma: no cover — Windows
    resource = None  # type: ignore
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_judge import GPT55Judge, default_judge  # noqa: E402
from answer_formatter import extract_livecodebench  # noqa: E402


_EXPLAIN_RUBRIC = (
    "The candidate is explaining why a piece of code raises a specific runtime "
    "error. Compare the candidate explanation to the gold explanation. Return "
    "JSON: {\"score\": 1 if the candidate identifies the same root cause else 0, "
    "\"rationale\": <short>}."
)


@dataclass
class LiveCodeBenchScore:
    pass_at_1: float
    n: int
    n_pass: int
    n_error: int


class LiveCodeBenchJudge:
    def __init__(
        self,
        judge: GPT55Judge | None = None,
        time_limit_s: float = 5.0,
        memory_limit_mb: int = 512,
    ) -> None:
        self.judge = judge or default_judge()
        self.time_limit_s = time_limit_s
        self.memory_limit_mb = memory_limit_mb

    def run_code(self, source: str, stdin_input: str) -> tuple[bool, str, str]:
        """Run python source with stdin; return (timed_out, stdout, stderr)."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
            fh.write(source)
            tmp_path = fh.name
        try:
            preexec = None
            if sys.platform != "win32" and resource is not None:
                mem_bytes = self.memory_limit_mb * 1024 * 1024

                def _limit() -> None:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                    resource.setrlimit(resource.RLIMIT_CPU, (int(self.time_limit_s) + 1, int(self.time_limit_s) + 1))

                preexec = _limit

            try:
                proc = subprocess.run(
                    [sys.executable, tmp_path],
                    input=stdin_input,
                    capture_output=True,
                    text=True,
                    timeout=self.time_limit_s,
                    preexec_fn=preexec,
                )
                return False, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired:
                return True, "", "timeout"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def score_code_generation(self, source: str, public_tests: list[dict], hidden_tests: list[dict]) -> bool:
        # Unwrap markdown code fences before executing.
        source = extract_livecodebench(source)
        for batch in (public_tests, hidden_tests):
            for test in batch:
                timed_out, out, err = self.run_code(source, test["input"])
                if timed_out:
                    return False
                if out != test["output"]:
                    return False
        return True

    def score_runtime_explain(self, candidate_explanation: str, gold_explanation: str, code_snippet: str) -> bool:
        r = self.judge.score_pointwise(
            prompt=f"Code:\n{code_snippet}\n\nGold explanation: {gold_explanation}",
            response=candidate_explanation,
            rubric=_EXPLAIN_RUBRIC,
        )
        return r.score >= 1

    def aggregate(self, items: list[dict]) -> LiveCodeBenchScore:
        """Items: [{kind: "code"|"explain", ...task-specific fields}]."""
        passed = errored = 0
        for it in items:
            try:
                if it["kind"] == "code":
                    ok = self.score_code_generation(it["source"], it.get("public_tests", []), it.get("hidden_tests", []))
                elif it["kind"] == "explain":
                    ok = self.score_runtime_explain(it["explanation"], it["gold_explanation"], it.get("code", ""))
                else:
                    ok = False
            except Exception:
                ok = False
                errored += 1
            if ok:
                passed += 1
        n = len(items)
        return LiveCodeBenchScore(
            pass_at_1=passed / n if n else 0.0,
            n=n,
            n_pass=passed,
            n_error=errored,
        )
