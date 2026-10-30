"""Windows-specific process control via ctypes (no pywin32)."""
from __future__ import annotations

import ctypes
import subprocess
import winreg
from ctypes import wintypes

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll")
psapi = ctypes.WinDLL("psapi")

PROCESS_ALL_ACCESS = 0x1F0FFF
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259
JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

k32.OpenProcess.restype = wintypes.HANDLE
k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k32.CloseHandle.argtypes = [wintypes.HANDLE]
k32.CreateJobObjectW.restype = wintypes.HANDLE
k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
k32.SetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD]
ntdll.NtSuspendProcess.argtypes = [wintypes.HANDLE]
ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
psapi.EmptyWorkingSet.argtypes = [wintypes.HANDLE]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in
                ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                 "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _open(pid: int, access=PROCESS_ALL_ACCESS):
    h = k32.OpenProcess(access, False, pid)
    if not h:
        raise OSError(ctypes.get_last_error(), f"OpenProcess({pid}) failed")
    return h


class ProcessGroup:
    """Job Object with kill-on-close: every runner dies with the core."""

    def __init__(self):
        self.handle = k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObject failed")
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(self.handle, JobObjectExtendedLimitInformation,
                                           ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

    def add(self, pid: int) -> None:
        h = _open(pid)
        try:
            if not k32.AssignProcessToJobObject(self.handle, h):
                raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        finally:
            k32.CloseHandle(h)


def runner_spawn_kwargs() -> dict:
    """Extra kwargs for asyncio.create_subprocess_exec: no console window, ever."""
    return {"creationflags": CREATE_NO_WINDOW}


# NOT DETACHED_PROCESS / pythonw: uv's venv launchers start the base interpreter as a child, and uv's
# pythonw launcher starts a *console* python.exe, which then gets a new, visible console window.
# CREATE_NO_WINDOW gives the launcher a hidden console that every console child inherits instead.
def detached_spawn_kwargs() -> dict:
    """For the shim launching the core: windowless (hidden console), own process group, outlives the agent."""
    return {"creationflags": CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB,
            "close_fds": True}


def detached_spawn_kwargs_fallback() -> dict:
    """Used when the caller's job forbids breakaway."""
    return {"creationflags": CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW, "close_fds": True}


def suspend(pid: int) -> None:
    h = _open(pid)
    try:
        if ntdll.NtSuspendProcess(h) != 0:
            raise OSError("NtSuspendProcess failed")
    finally:
        k32.CloseHandle(h)


def resume(pid: int) -> None:
    h = _open(pid)
    try:
        if ntdll.NtResumeProcess(h) != 0:
            raise OSError("NtResumeProcess failed")
    finally:
        k32.CloseHandle(h)


def trim_working_set(pid: int) -> None:
    """Page the process out: empty its working set (pages go to standby/pagefile)."""
    h = _open(pid)
    try:
        if not psapi.EmptyWorkingSet(h):
            k32.SetProcessWorkingSetSizeEx(h, ctypes.c_size_t(-1), ctypes.c_size_t(-1), 0)
    finally:
        k32.CloseHandle(h)


def pid_alive(pid: int) -> bool:
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    try:
        code = wintypes.DWORD()
        return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def pagefile_info() -> dict:
    """Configured paging files (registry) - used for the start-up pagefile warning."""
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management")
        val, _ = winreg.QueryValueEx(key, "PagingFiles")
        entries = [v for v in (val if isinstance(val, list) else [val]) if v]
    except OSError:
        entries = []
    return {"paging_files": entries}


def venv_python(venv: str, windowless: bool = False) -> str:
    import os
    return os.path.join(venv, "Scripts", "pythonw.exe" if windowless else "python.exe")


def run_hidden(args, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, creationflags=CREATE_NO_WINDOW, **kw)
