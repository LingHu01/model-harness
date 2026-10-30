"""Minimal NVML bindings over ctypes (stdlib only).

Used by the core (free VRAM at decision time) and by runners (external-allocator
sampler during requests). Every call degrades to None when NVML is unavailable.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading

_lib = None
_lock = threading.Lock()
_init_ok = False


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _ProcInfo(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


def _load():
    global _lib, _init_ok
    with _lock:
        if _lib is not None:
            return _init_ok
        try:
            if sys.platform == "win32":
                candidates = [os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvml.dll"),
                              os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                           "NVIDIA Corporation", "NVSMI", "nvml.dll")]
            else:
                candidates = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
            for c in candidates:
                try:
                    _lib = ctypes.CDLL(c)
                    break
                except OSError:
                    continue
            if _lib is None:
                _lib = False
                return False
            _init_ok = _lib.nvmlInit_v2() == 0
        except Exception:
            _lib, _init_ok = False, False
        return _init_ok


def device_index(device: str) -> int:
    """'cuda:1' -> 1, 'cuda' -> 0."""
    if ":" in device:
        return int(device.split(":", 1)[1])
    return 0


def _handle(index: int):
    h = ctypes.c_void_p()
    if _lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(h)) != 0:
        return None
    return h


def memory(index: int = 0):
    """(total_gb, free_gb, used_gb) or None."""
    if not _load():
        return None
    h = _handle(index)
    if h is None:
        return None
    m = _Memory()
    if _lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m)) != 0:
        return None
    g = 1024 ** 3
    return m.total / g, m.free / g, m.used / g


def device_name(index: int = 0):
    if not _load():
        return None
    h = _handle(index)
    if h is None:
        return None
    buf = ctypes.create_string_buffer(96)
    if _lib.nvmlDeviceGetName(h, buf, ctypes.c_uint(96)) != 0:
        return None
    return buf.value.decode(errors="replace")


def driver_version():
    if not _load():
        return None
    buf = ctypes.create_string_buffer(96)
    if _lib.nvmlSystemGetDriverVersion(buf, ctypes.c_uint(96)) != 0:
        return None
    return buf.value.decode(errors="replace")


def compute_processes(index: int = 0):
    """[(pid, used_gb or None)]. Often empty or without sizes under Windows WDDM."""
    if not _load():
        return []
    h = _handle(index)
    if h is None:
        return []
    count = ctypes.c_uint(64)
    arr = (_ProcInfo * 64)()
    fn = getattr(_lib, "nvmlDeviceGetComputeRunningProcesses_v3", None) or \
        getattr(_lib, "nvmlDeviceGetComputeRunningProcesses_v2", None)
    if fn is None or fn(h, ctypes.byref(count), arr) != 0:
        return []
    out = []
    for i in range(count.value):
        used = arr[i].usedGpuMemory
        out.append((arr[i].pid, None if used >= (1 << 63) else used / 1024 ** 3))
    return out
