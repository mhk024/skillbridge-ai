"""Check that every question's reference solution passes its own test cases.

    python -m scripts.validate_questions

Developer tool for TRUSTED reference solutions only (not the student code runner).
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from src.assessment.question_service import load_question_bank
from src.utils.config import get_settings
from src.utils.helpers import normalize_output

TIMEOUT_SECONDS = 5


def run_solution(code: str, stdin_text: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(code, encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-I", str(path)],
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=TIMEOUT_SECONDS,
        )


def validate_one(question: dict) -> list[str]:
    """Return a list of problems (empty list means the question is consistent)."""
    lines = question.get("reference_solution")
    if not lines:
        return ["no reference_solution provided"]
    code = "\n".join(lines) + "\n"
    problems = []
    for i, test in enumerate(question["test_cases"], start=1):
        try:
            proc = run_solution(code, test["input"])
        except subprocess.TimeoutExpired:
            problems.append(f"test {i}: timed out after {TIMEOUT_SECONDS}s")
            continue
        if proc.returncode != 0:
            problems.append(f"test {i}: crashed: {proc.stderr.strip()[-200:]}")
            continue
        got = normalize_output(proc.stdout)
        want = normalize_output(test["output"])
        if got != want:
            problems.append(f"test {i}: expected {want!r} but solution printed {got!r}")
    return problems


def validate_all(questions: list[dict] | None = None) -> dict[str, list[str]]:
    if questions is None:
        questions = load_question_bank(get_settings().questions_dir)
    return {q["id"]: validate_one(q) for q in questions}


def main() -> int:
    results = validate_all()
    failed = 0
    for qid, problems in results.items():
        if problems:
            failed += 1
            print(f"FAIL {qid}")
            for p in problems:
                print(f"     - {p}")
        else:
            print(f"ok   {qid}")
    print(f"\n{len(results) - failed}/{len(results)} questions consistent.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())