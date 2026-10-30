"""Process memory readings without third-party deps (runners don't have psutil)."""
from __future__ import annotations

import ctypes
import sys


def peak_private_gb() -> float | None:
    """Peak host RAM of this process: peak working set (Windows) / peak RSS (Linux).

    Not PeakPagefileUsage (private bytes): under WDDM, CUDA device allocations are committed against
    the process, so private bytes count VRAM as RAM (measured: +4.3 GB for a 4 GB CUDA tensor).
    """
    try:
        if sys.platform == "win32":
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32")
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return pmc.PeakWorkingSetSize / 1024 ** 3
            return None
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2  # KiB -> GiB
    except Exception:
        return None
