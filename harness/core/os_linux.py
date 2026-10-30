"""Linux process control (process group + parent-death signal). Not yet exercised."""
from __future__ import annotations

import ctypes
import os
import signal
import subprocess

PR_SET_PDEATHSIG = 1


class ProcessGroup:
    """Runners get PR_SET_PDEATHSIG via preexec_fn, so they die with the core."""

    def add(self, pid: int) -> None:
        pass


def _pdeathsig():
    try:
        ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except OSError:
        pass


def runner_spawn_kwargs() -> dict:
    return {"preexec_fn": _pdeathsig}


def detached_spawn_kwargs() -> dict:
    return {"start_new_session": True, "close_fds": True}


def detached_spawn_kwargs_fallback() -> dict:
    return detached_spawn_kwargs()


def suspend(pid: int) -> None:
    os.kill(pid, signal.SIGSTOP)


def resume(pid: int) -> None:
    os.kill(pid, signal.SIGCONT)


def trim_working_set(pid: int) -> None:
    # Stopped processes are swapped out by the kernel under pressure; nothing explicit here.
    pass


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def pagefile_info() -> dict:
    try:
        with open("/proc/swaps") as f:
            return {"paging_files": [l.split()[0] for l in f.read().splitlines()[1:] if l.strip()]}
    except OSError:
        return {"paging_files": []}


def venv_python(venv: str, windowless: bool = False) -> str:
    return os.path.join(venv, "bin", "python")


def run_hidden(args, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, **kw)
