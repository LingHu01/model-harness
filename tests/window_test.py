"""Windowless check (Windows): launch the core the way the shim does, load runners + a warm shell,
and diff the set of visible top-level windows. Any new window = failure.

    .venv\\Scripts\\python.exe tests\\window_test.py
"""
import ctypes
import json
import os
import subprocess
import sys
import time
import urllib.request
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HARNESS_CONFIG", str(ROOT / "tests" / "selftest.yaml"))
from core.os_win import detached_spawn_kwargs  # noqa: E402

PORT = 8799
u32 = ctypes.WinDLL("user32")
EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def windows():
    out = {}

    def cb(h, _):
        if u32.IsWindowVisible(h):
            pid, cls, title = wintypes.DWORD(), ctypes.create_unicode_buffer(256), ctypes.create_unicode_buffer(256)
            u32.GetWindowThreadProcessId(h, ctypes.byref(pid))
            u32.GetClassNameW(h, cls, 256)
            u32.GetWindowTextW(h, title, 256)
            out[h] = (pid.value, cls.value, title.value)
        return True

    u32.EnumWindows(EnumProc(cb), 0)
    return out


def post(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body or {}).encode(), method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


before = windows()
subprocess.Popen([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "core.server"], cwd=str(ROOT),
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 **detached_spawn_kwargs())
seen = {}
deadline = time.time() + 30
for _ in range(300):  # sample often: a flashing window is a failure too
    seen.update({h: v for h, v in windows().items() if h not in before})
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
        break
    except OSError:
        time.sleep(0.1)
for model, fn, inputs in (("echo-test", "echo", {"text": "x"}), ("torch-a", "alloc", {"gb": 0.2})):
    post("/run", {"model": model, "function": fn, "inputs": inputs})
    seen.update({h: v for h, v in windows().items() if h not in before})
post("/unload/torch-a")  # -> warm shell spawn + page
for _ in range(40):
    seen.update({h: v for h, v in windows().items() if h not in before})
    time.sleep(0.1)
post("/shutdown")
time.sleep(2)
print("new visible windows:", seen or "none")
sys.exit(1 if seen else 0)
