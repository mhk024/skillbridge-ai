import sys

import pytest

from src.execution.sandbox import run_python


def test_runs_code_and_captures_output():
    r = run_python("print('hello')")
    assert r.status == "ok"
    assert r.stdout.strip() == "hello"
    assert r.exit_code == 0


def test_stdin_is_passed_to_program():
    r = run_python("print(int(input()) * 2)", "21\n")
    assert r.stdout.strip() == "42"


def test_unicode_round_trip():
    r = run_python("print(input().upper())", "héllo ✓\n")
    assert r.stdout.strip() == "HÉLLO ✓"


def test_runtime_error_is_reported():
    r = run_python("print(1 / 0)")
    assert r.status == "runtime_error"
    assert "ZeroDivisionError" in r.stderr
    assert r.exit_code != 0


def test_infinite_loop_times_out():
    r = run_python("while True:\n    pass", timeout=1.0)
    assert r.status == "timeout"
    assert r.duration_ms < 8000


def test_output_flood_is_capped():
    r = run_python("while True:\n    print('x' * 1000)", timeout=5.0, max_output_bytes=10_000)
    assert r.status == "output_limit"
    assert len(r.stdout) <= 10_000


def test_runs_without_site_packages():
    r = run_python("import sys\nprint(any('site-packages' in p for p in sys.path))")
    assert r.stdout.strip() == "False"


def test_secrets_in_environment_are_not_visible(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "super-secret-value")
    r = run_python("import os\nprint(os.environ.get('LLM_API_KEY'))")
    assert r.stdout.strip() == "None"


def test_memory_limit_enforced_when_available():
    probe = run_python("print(1)")
    if not probe.memory_limit_enforced:
        pytest.skip("Memory limit could not be enforced on this system")
    r = run_python(
        "x = bytearray(2 * 1024 * 1024 * 1024)\nprint('allocated')", memory_mb=128, timeout=5.0
    )
    assert r.status != "ok"
    assert "allocated" not in r.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Job Object limit is Windows-specific")
def test_child_processes_blocked_on_windows():
    probe = run_python("print(1)")
    if not probe.memory_limit_enforced:
        pytest.skip("Job Object could not be attached on this system")
    r = run_python(
        "import subprocess\nsubprocess.run(['cmd', '/c', 'echo hi'])\nprint('spawned')"
    )
    assert "spawned" not in r.stdout