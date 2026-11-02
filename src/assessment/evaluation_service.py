"""Evaluate submitted code against a question's test cases.

Two evaluation modes (be clear about what each means):
  sample : only the visible example tests. A quick check, NOT proof of correctness.
  judge  : visible + hidden tests. Closest thing to a real online judge here, but it
           runs on a local, limited sandbox (see src/execution/sandbox.py).
"""

import re
from dataclasses import dataclass

from src.execution.sandbox import run_python
from src.utils.config import get_settings
from src.utils.helpers import normalize_output

EVAL_MODES = ("sample", "judge")
MAX_CODE_CHARS = 20_000
PER_TEST_TIMEOUT_S = 2.0
MAX_SHOWN_CHARS = 500

_LINE_RE = re.compile(r'File ".*?main\.py", line (\d+)')


@dataclass(frozen=True)
class CaseOutcome:
    index: int
    hidden: bool
    status: str  # passed | wrong_answer | runtime_error | timeout | output_limit | internal_error
    duration_ms: float
    # Details are filled in for VISIBLE tests only; hidden tests never expose them.
    input_text: str | None = None
    expected: str | None = None
    actual: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    status: str  # evaluated | syntax_error | invalid | disabled
    mode: str
    passed: int = 0
    total: int = 0
    outcomes: tuple = ()
    message: str = ""
    avg_duration_ms: float | None = None
    memory_limit_enforced: bool | None = None

    @property
    def is_correct(self) -> bool:
        return self.status == "evaluated" and self.total > 0 and self.passed == self.total

    @property
    def counts_as_attempt(self) -> bool:
        """Syntax errors count as a failed attempt; invalid/disabled runs do not."""
        return self.status in ("evaluated", "syntax_error")


def _clip(text: str, limit: int = MAX_SHOWN_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "... (truncated)"


def summarize_error(stderr: str) -> str:
    """One-line error summary (last traceback line plus the line number)."""
    text = (stderr or "").strip()
    if not text:
        return ""
    last = text.splitlines()[-1].strip()
    lines = _LINE_RE.findall(text)
    where = f" (line {lines[-1]})" if lines else ""
    return _clip(last + where)


def select_tests(test_cases: list[dict], mode: str) -> list[dict]:
    if mode not in EVAL_MODES:
        raise ValueError(f"mode must be one of {EVAL_MODES}")
    return [t for t in test_cases if mode == "judge" or not t.get("hidden", False)]


def evaluate_code(
    code: str,
    test_cases: list[dict],
    *,
    mode: str,
    enable_execution: bool | None = None,
    timeout: float = PER_TEST_TIMEOUT_S,
) -> EvaluationResult:
    selected = select_tests(test_cases, mode)
    total = len(selected)

    if code is None or not code.strip():
        return EvaluationResult("invalid", mode, total=total, message="Write some code before submitting.")
    if len(code) > MAX_CODE_CHARS:
        return EvaluationResult(
            "invalid", mode, total=total,
            message=f"Code is too long (maximum {MAX_CODE_CHARS} characters).",
        )

    if enable_execution is None:
        enable_execution = get_settings().enable_code_execution
    if not enable_execution:
        return EvaluationResult(
            "disabled", mode, total=total,
            message="Code execution is disabled in this deployment.",
        )

    try:
        compile(code, "<submission>", "exec")  # checks syntax only; does not run anything
    except SyntaxError as exc:
        where = f" on line {exc.lineno}" if exc.lineno else ""
        return EvaluationResult(
            "syntax_error", mode, total=total,
            message=f"{type(exc).__name__}{where}: {exc.msg}",
        )
    except (ValueError, RecursionError, MemoryError) as exc:
        return EvaluationResult(
            "syntax_error", mode, total=total,
            message=f"Your code could not be compiled ({type(exc).__name__}).",
        )

    outcomes: list[CaseOutcome] = []
    memory_enforced: bool | None = None
    for i, case in enumerate(selected, start=1):
        hidden = bool(case.get("hidden", False))
        result = run_python(code, case["input"], timeout=timeout)
        memory_enforced = result.memory_limit_enforced

        actual = normalize_output(result.stdout) if result.status == "ok" else None
        if result.status == "ok":
            status = "passed" if actual == normalize_output(case["output"]) else "wrong_answer"
        else:
            status = result.status

        if hidden:
            outcomes.append(CaseOutcome(i, True, status, result.duration_ms))
        else:
            outcomes.append(
                CaseOutcome(
                    i, False, status, result.duration_ms,
                    input_text=_clip(case["input"].rstrip("\n")),
                    expected=_clip(normalize_output(case["output"])),
                    actual=_clip(actual) if actual is not None else None,
                    error=summarize_error(result.stderr) if status == "runtime_error" else None,
                )
            )

    passed = sum(1 for o in outcomes if o.status == "passed")
    avg = sum(o.duration_ms for o in outcomes) / len(outcomes) if outcomes else None
    return EvaluationResult(
        "evaluated", mode, passed=passed, total=total, outcomes=tuple(outcomes),
        avg_duration_ms=avg, memory_limit_enforced=memory_enforced,
    )