"""Run untrusted Python code in a separate, resource-limited process.

SECURITY BOUNDARY (read this):
  Enforced : wall-clock timeout, output size cap, stripped environment, temp working
             folder, isolated interpreter (no site-packages). Windows: memory cap and
             "no child processes" via a Job Object. Linux: rlimit memory/CPU/file size.
  NOT enforced: file-system reads, network access. This is NOT a secure sandbox for
             untrusted internet users. Use only locally, or disable with
             ENABLE_CODE_EXECUTION=false.
"""

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_S = 2.0
DEFAULT_MEMORY_MB = 256
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024

_IS_WINDOWS = sys.platform == "win32"


@dataclass(frozen=True)
class ExecutionResult:
    status: str  # ok | timeout | runtime_error | output_limit | internal_error
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: float = 0.0  # wall clock, includes interpreter start-up
    memory_limit_enforced: bool = False


# ----------------------------------------------------------------- Windows Job Object
if _IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _BASIC_LIMITS(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _EXTENDED_LIMITS(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BASIC_LIMITS),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _JobObjectExtendedLimitInformation = 9
    _LIMIT_ACTIVE_PROCESS = 0x00000008
    _LIMIT_PROCESS_MEMORY = 0x00000100
    _LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
    ]
    _k32.SetInformationJobObject.restype = wintypes.BOOL
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.AssignProcessToJobObject.restype = wintypes.BOOL
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CloseHandle.restype = wintypes.BOOL

    def _attach_job(pid: int, memory_mb: int):
        """Put the process in a Job Object: memory cap, one process only, kill on close.

        Returns the job handle, or None if any step failed (limits then NOT enforced).
        """
        try:
            job = _k32.CreateJobObjectW(None, None)
            if not job:
                return None
            info = _EXTENDED_LIMITS()
            info.BasicLimitInformation.LimitFlags = (
                _LIMIT_PROCESS_MEMORY | _LIMIT_ACTIVE_PROCESS | _LIMIT_KILL_ON_JOB_CLOSE
            )
            info.BasicLimitInformation.ActiveProcessLimit = 1
            info.ProcessMemoryLimit = memory_mb * 1024 * 1024
            if not _k32.SetInformationJobObject(
                job, _JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
            ):
                _k32.CloseHandle(job)
                return None
            process = _k32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
            if not process:
                _k32.CloseHandle(job)
                return None
            assigned = _k32.AssignProcessToJobObject(job, process)
            _k32.CloseHandle(process)
            if not assigned:
                _k32.CloseHandle(job)
                return None
            return job
        except Exception:
            return None

    def _close_job(job) -> None:
        if job:
            _k32.CloseHandle(job)  # KILL_ON_JOB_CLOSE ends anything left running


# ------------------------------------------------------------------------- helpers
def _python_executable() -> str:
    # Inside a venv on Windows, sys.executable is a launcher that starts the real
    # interpreter as a child process. Use the base interpreter so the Job Object limits
    # apply to the process that actually runs the code (it also has no venv packages).
    return getattr(sys, "_base_executable", None) or sys.executable


def _clean_env() -> dict:
    env = {}
    if _IS_WINDOWS:
        root = os.environ.get("SYSTEMROOT")
        if root:
            env["SYSTEMROOT"] = root  # required by Windows Python for some modules
    return env  # deliberately nothing else: no API keys, no PATH


def _posix_limiter(memory_mb: int, timeout: float):
    def apply() -> None:
        import resource

        mem = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        cpu = max(1, int(timeout) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))

    return apply


def _kill(proc: subprocess.Popen) -> None:
    try:
        if _IS_WINDOWS:
            proc.kill()
        else:
            os.killpg(proc.pid, getattr(signal, "SIGKILL"))
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass


class _Reader(threading.Thread):
    """Reads a pipe into memory, stopping (and killing the process) past the limit."""

    def __init__(self, stream, limit: int, on_overflow):
        super().__init__(daemon=True)
        self._stream = stream
        self._limit = limit
        self._on_overflow = on_overflow
        self.data = bytearray()
        self.overflowed = False

    def run(self) -> None:
        try:
            while True:
                chunk = self._stream.read1(65536)
                if not chunk:
                    break
                room = self._limit - len(self.data)
                if len(chunk) > room:
                    self.data.extend(chunk[: max(room, 0)])
                    self.overflowed = True
                    self._on_overflow()
                    break
                self.data.extend(chunk)
        except (OSError, ValueError):
            pass


def _feed_stdin(proc: subprocess.Popen, payload: bytes) -> None:
    try:
        proc.stdin.write(payload)
    except (OSError, ValueError):
        pass
    finally:
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass


# ----------------------------------------------------------------------- main API
def run_python(
    code: str,
    stdin_text: str = "",
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> ExecutionResult:
    """Run `code` with `stdin_text` as input in a separate limited process."""
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="skillbridge_run_") as tmp:
        script = Path(tmp) / "main.py"
        script.write_text(code, encoding="utf-8")
        cmd = [_python_executable(), "-I", "-S", "-X", "utf8", str(script)]

        kwargs: dict = {}
        if _IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
            kwargs["preexec_fn"] = _posix_limiter(memory_mb, timeout)

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmp,
                env=_clean_env(),
                **kwargs,
            )
        except OSError as exc:
            return ExecutionResult(
                status="internal_error",
                stderr=f"Could not start the code runner ({type(exc).__name__}).",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        job = _attach_job(proc.pid, memory_mb) if _IS_WINDOWS else None
        memory_enforced = (job is not None) if _IS_WINDOWS else True

        kill = lambda: _kill(proc)  # noqa: E731
        out_reader = _Reader(proc.stdout, max_output_bytes, kill)
        err_reader = _Reader(proc.stderr, max_output_bytes, kill)
        writer = threading.Thread(
            target=_feed_stdin,
            args=(proc, (stdin_text or "").encode("utf-8", errors="replace")),
            daemon=True,
        )
        for t in (out_reader, err_reader, writer):
            t.start()

        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill()
        proc.wait()  # reap the process
        for t in (out_reader, err_reader, writer):
            t.join(timeout=2)
        if _IS_WINDOWS:
            _close_job(job)

        duration_ms = (time.perf_counter() - start) * 1000

    stdout = bytes(out_reader.data).decode("utf-8", errors="replace")
    stderr = bytes(err_reader.data).decode("utf-8", errors="replace")

    if out_reader.overflowed or err_reader.overflowed:
        status = "output_limit"
    elif timed_out:
        status = "timeout"
    elif proc.returncode != 0:
        status = "runtime_error"
    else:
        status = "ok"

    return ExecutionResult(
        status=status,
        stdout=stdout,
        stderr=stderr,
        exit_code=proc.returncode,
        duration_ms=duration_ms,
        memory_limit_enforced=memory_enforced,
    )