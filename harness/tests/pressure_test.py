"""VRAM watcher, against a core started with HARNESS_CONFIG=tests/pressure.yaml."""
from __future__ import annotations

import sys
import threading
import time

from selftest import check, http, results, run, state

GONE = ("PAGED", "SHELL", "SPAWNING", "GONE")


def wait_for(model, states, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if state(model) in states:
            return True
        time.sleep(0.2)
    return False


# running: must not be touched, even under pressure
box = {}
t = threading.Thread(target=lambda: box.update(r=run("torch-a", "alloc", {"gb": 0.2, "seconds": 5})))
t.start()
while state("torch-a") not in ("BUSY",):
    time.sleep(0.2)
time.sleep(3)
check("busy model is not unloaded", state("torch-a") == "BUSY", state("torch-a"))
t.join()
check("its request finishes", box["r"].get("event") == "result", str(box["r"])[:120])

# idle: unloaded by the watcher within a few seconds, though idle_timeout is far away
check("idle model is unloaded", wait_for("torch-a", GONE, 5), state("torch-a"))
r = run("torch-a", "alloc", {"gb": 0.1})
check("it serves the next request", r.get("event") == "result", str(r)[:120])

# no workers left: the watcher stops by itself
http("POST", "/unload/torch-a")
bad = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(bad)}/{len(results)} passed" + (f"; failed: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
