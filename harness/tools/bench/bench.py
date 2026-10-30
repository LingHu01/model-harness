"""Find the largest useful batch for a model. Run manually against a running core:

    harness-bench <model> [--max 16] [--rounds 2] [--port 8765] [--inputs bench.json] [--yes]

Inputs: <plugin folder>/tests/bench/bench.json, a list of {"function": ..., "inputs": {...}}, one per function;
relative file paths inside inputs resolve against that folder. Use real inputs: VRAM per extra request grows with
input size.

Each function is tuned on its own, on its one fixed input, so every request in every step is identical: same image,
same tokens. Requests/s and tokens/s then compare the same way, and a difference between steps is the batch size.

A step sends exactly C requests together, so they run as one batch of C, for --rounds whole batches. It watches
two flags live:
- VRAM against what was free on the card when the bench started.
- Memory bandwidth against the best seen so far. GPU utilization is not used: it reads high at low clock or
  power too. Bandwidth falling means the GPU can't keep up with its memory, or is waiting on system RAM.
Going over VRAM, or bandwidth collapsing, cancels the step at once: finishing it would only waste time. A smaller
drop across the whole step fails it when it ends. Bandwidth only counts once the plugin reports "generating", so
loading and preprocessing are never judged; a plugin that doesn't report it is judged on VRAM alone.

Steps are chosen, not swept: C=1 is the baseline, then mid-range, then half of that. If the lower one matches
mid-range, the limit is below it, so keep halving down. If mid-range is better, halve upward, up to where VRAM is
predicted to run out. High steps are the slowest, so they only run once a lower step has shown they could help.
A C predicted to exceed the available VRAM is never run.

Saves concurrency.max + vram_per_extra_gb to DATA_ROOT/state/tuning.yaml (per machine, untracked) ONLY after
explicit confirmation; the manifest is never edited. The core picks it up at once (/reload).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from core import config as config_mod  # noqa: E402
from core import tuning  # noqa: E402
from sdk import nvml  # noqa: E402

VRAM_MARGIN = 0.95    # of the VRAM free when the bench started
BANDWIDTH_DROP = 15   # points below the best step: compute can't keep up with the memory
COLLAPSE = 0.5        # live cancel: bandwidth under this share of the best step
ABORT_WINDOW = 3.0    # seconds of generation it must stay there
PLATEAU = 0.95        # throughput within this share of the best: a higher C gains nothing


class Core:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"

    def call(self, method, path, body=None, timeout=60):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    def run(self, model, function, inputs, rid, on_stage=None, limit_s=None):
        body = json.dumps({"model": model, "function": function, "inputs": inputs, "request_id": rid,
                           "limit_s": limit_s}).encode()
        req = urllib.request.Request(self.base + "/run", body, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                ev = None
                for line in r:
                    ev = json.loads(line)
                    if on_stage and ev.get("event") == "stage":
                        on_stage()
                return ev
        except urllib.error.HTTPError as e:
            return {"event": "error", **json.loads(e.read()).get("error", {})}


def load_inputs(model_dir: Path, spec: Path | None = None) -> list[dict]:
    bdir = spec.parent if spec else model_dir / "tests" / "bench"
    spec = spec or bdir / "bench.json"
    if not spec.exists():
        sys.exit(f"no {spec}: add a list of {{function, inputs}} (fallback to the smoke sample is plugin-specific)")
    items = json.loads(spec.read_text(encoding="utf-8-sig"))
    for it in items:
        for k, v in it["inputs"].items():
            if isinstance(v, str) and (bdir / v).exists():
                it["inputs"][k] = str((bdir / v).resolve())
    return items


def others_active(core, model) -> bool:
    st = core.call("GET", "/status")["models"]
    return any((s["queued"] or s["running"]) for n, s in st.items() if n != model)


class StepWatch:
    """Samples memory bandwidth and VRAM while a step runs, and calls `abort(reason)` once when a flag trips."""

    def __init__(self, index, limit_gb, base_used_gb, floor, abort, period=0.1):
        self.index, self.limit_gb, self.base_used_gb = index, limit_gb, base_used_gb
        self.floor, self.abort, self.period = floor, abort, period
        self.bandwidth: list[int] = []
        self.recent: list[int] = []      # this batch, since it started generating
        self.generating = threading.Event()
        self.reason: str | None = None
        self._stop = threading.Event()

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join()

    def new_batch(self):
        self.generating.clear()
        self.recent = []

    def _trip(self, reason):
        self.reason = reason
        self.abort(reason)

    def _loop(self):
        window = int(ABORT_WINDOW / self.period)
        while not self._stop.wait(self.period):
            u, mem = nvml.utilization(self.index), nvml.memory(self.index)
            if u and self.generating.is_set():
                self.bandwidth.append(u[1])
                self.recent.append(u[1])
            if self.reason:
                continue
            if mem and mem[2] - self.base_used_gb > self.limit_gb:
                self._trip("vram")
            elif self.floor is not None and len(self.recent) >= window:
                if statistics.mean(self.recent[-window:]) < (self.floor + BANDWIDTH_DROP) * COLLAPSE:
                    self._trip("bandwidth")


def measure_step(core, model, item, c, rounds, index, limit_gb, base_used_gb, floor, tag) -> dict:
    """`rounds` batches of exactly c identical requests. A tripped flag cancels the step at once."""
    latencies, peaks, errors, ks = [], [], [], []
    lock = threading.Lock()
    spilled, done, started = False, 0, time.perf_counter()
    tokens, generating = 0, 0.0
    batch: dict = {}          # metrics of this batch: every request in it reports the same ones
    live: list[str] = []

    def one(i, rid, gate):
        nonlocal spilled
        it = item
        gate.wait()
        t = time.perf_counter()
        ev = core.run(model, it["function"], it["inputs"], rid, on_stage=watch.generating.set)
        with lock:
            if ev["event"] == "result" and ev["result"].get("stopped") and not ev["metrics"].get("tokens"):
                errors.append("stopped_without_work")   # a measurement of nothing must never count
            elif ev["event"] == "result":
                latencies.append(time.perf_counter() - t)
                m = ev["metrics"]
                peaks.append(m.get("vram_peak_gb") or 0.0)
                ks.append(m.get("k") or 1)
                spilled = spilled or bool(m.get("vram_spill"))
                batch.update(m)
            else:
                errors.append(ev.get("code"))

    def abort(reason):
        print(f"  C={c}: {reason} flag tripped, cancelling the step")
        for rid in list(live):
            core.call("POST", f"/cancel/{rid}")

    with StepWatch(index, limit_gb, base_used_gb, floor, abort) as watch:
        r = -1
        while True:
            r += 1
            watch.new_batch()
            batch.clear()
            gate = threading.Barrier(c)
            live[:] = [f"bench-{tag}-{c}-{r}-{i}-{os.getpid()}" for i in range(c)]
            threads = [threading.Thread(target=one, args=(i + r * c, live[i], gate), daemon=True) for i in range(c)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            done += 1
            tokens += batch.get("tokens") or 0
            generating += batch.get("generating_s") or 0.0
            if watch.reason or errors or spilled or (peaks and max(peaks) > limit_gb):
                break
            if done >= rounds:
                break
    wall = time.perf_counter() - started
    latencies.sort()
    n = len(latencies)
    return {"c": c, "tput": n / wall if wall else 0.0, "tok_s": tokens / generating if generating else None, "p95": latencies[int(round(0.95 * (n - 1)))] if n else None,
            "vram": max(peaks, default=0.0), "bandwidth": statistics.mean(watch.bandwidth) if watch.bandwidth else None,
            "k": max(ks, default=0), "rounds": done, "errors": errors, "spilled": spilled, "abort": watch.reason}


# ---- search: only this part decides, so it can be tested on simulated curves --------------------------
def tripped(row, limit_gb, floor=None) -> str | None:
    """Why this step is past the maximum batch, or None."""
    if row.get("abort"):
        return row["abort"]
    if row["errors"]:
        return "errors"
    if row["spilled"] or row["vram"] > limit_gb:
        return "vram"
    if floor is not None and row["bandwidth"] is not None and row["bandwidth"] < floor:
        return "bandwidth"
    return None


def vram_slope(rows) -> float:
    """GB per extra request, from the widest pair of steps that stayed on the card."""
    if len(rows) < 2:
        return 0.0
    a, b = rows[0], rows[-1]
    return max(0.0, (b["vram"] - a["vram"]) / (b["c"] - a["c"]))


def predicted_top(rows, limit_gb, cmax) -> int:
    """Highest C that should fit, from the steps so far (rows are sorted by c and all fit)."""
    slope = vram_slope(rows)
    if slope <= 0:
        return cmax
    last = rows[-1]
    return max(last["c"], min(cmax, last["c"] + math.floor((limit_gb - last["vram"]) / slope)))


def choose_max(measure, cmax: int, limit_gb: float, drop: float = BANDWIDTH_DROP, say=print) -> tuple[int, list[dict]]:
    """Returns (best C, every step measured). `measure(c, floor)` runs one step; floor is the live bandwidth limit."""
    seen: dict[int, dict] = {}

    def floor():
        clean = [r["bandwidth"] for r in seen.values() if r["bandwidth"] is not None and not tripped(r, limit_gb)]
        return max(clean) - drop if clean else None

    def step(c):
        seen[c] = row = measure(c, floor())
        tok = f"  {row['tok_s']:6.1f} tok/s" if row.get("tok_s") else ""
        say(f"  C={c:<3} {row['tput']:6.2f} req/s{tok}  p95 {row['p95'] if row['p95'] is None else round(row['p95'], 2)} s  "
            f"vram {row['vram']:5.2f}/{limit_gb:.2f} GB  bandwidth {row['bandwidth'] or 0:3.0f} %  batch {row['k']}"
            + (f"  errors {sorted(set(map(str, row['errors'])))}" if row["errors"] else "")
            + ("  spilled" if row["spilled"] else "") + (f"  cancelled: {row['abort']}" if row.get("abort") else ""))
        return row

    def good():
        f = floor()
        return sorted((r for r in seen.values() if not tripped(r, limit_gb, f)), key=lambda r: r["c"])

    if tripped(step(1), limit_gb):
        say("stop: one request already exceeds the available VRAM")
        return 0, list(seen.values())
    if cmax == 1:
        return 1, list(seen.values())
    # C=1 is the baseline; mid-range gives a good VRAM slope. Then compare downward before going higher: high steps
    # are the slow ones, so they only run once a lower step has shown that more requests still help.
    second = max(2, cmax // 2)
    lo, hi = 1, second                   # hi: lowest C known to be no better, or one past the highest C worth trying
    probe = None
    if not tripped(step(second), limit_gb, floor()):
        lo, hi = second, min(cmax, predicted_top(good(), limit_gb, cmax)) + 1
        if hi - 1 < cmax:
            say(f"VRAM predicts a limit at C={hi - 1}: nothing above it will run")
        if second >= 4:
            lo, probe = 1, second // 2
    else:
        say(f"  C={second} tripped a flag: searching below it")

    while probe or hi - lo > max(1, hi // 8):     # within one eighth is close enough
        c, probe = probe or (lo + hi) // 2, None  # halve the range: the limit is found in a few steps
        row = seen.get(c) or step(c)
        why = tripped(row, limit_gb, floor())
        above = [r for r in good() if r["c"] > c]
        below = [r for r in good() if r["c"] < c]
        if why:
            hi = c
            say(f"  C={c} tripped {why}: searching below it")
        elif above and row["tput"] >= PLATEAU * max(r["tput"] for r in above):
            hi = c
            say(f"  C={c} matches the higher steps: searching below it")
        elif above:                              # the higher step is really better
            lo = c if hi in seen else max(r["c"] for r in above)   # bounded: halve between; open: move past it
            say(f"  C={c} is below the plateau: searching above C={lo}")
        elif below and below[-1]["tput"] >= PLATEAU * row["tput"]:
            hi = below[-1]["c"]                  # the top added nothing: the plateau starts lower
            lo = below[-2]["c"] if len(below) > 1 else 1
            say(f"  C={c} adds nothing over C={hi}: searching below it")
        else:
            lo = c
            hi = min(hi, predicted_top(good(), limit_gb, cmax) + 1)   # new evidence can lower the ceiling
            say(f"  C={c} still gains: searching above it")

    usable = good()
    best = max(r["tput"] for r in usable)
    # smallest C that reaches the plateau: more only adds latency and VRAM
    return min(r["c"] for r in usable if r["tput"] >= PLATEAU * best), list(seen.values())


def cat_device(raw: dict, variant: str | None) -> str:
    rt = {**(raw.get("runtime") or {}), **((((raw.get("variants") or {}).get(variant) or {}).get("runtime")) or {})}
    return rt.get("device", "cuda:0")


def tune_function(core, a, item, cmax, index) -> dict | None:
    holder, fn = f"bench:{a.model}", item["function"]
    lock = core.call("POST", "/bench/lock", {"holder": holder})
    if "error" in lock:
        sys.exit(lock["error"]["message"])
    try:
        core.call("POST", f"/unload/{a.model}")
        time.sleep(1)
        mem = nvml.memory(index)
        limit_gb = mem[1] * VRAM_MARGIN if mem else float("inf")
        base_used_gb = mem[2] if mem else 0.0
        print(f"up to C={cmax}, {mem[1]:.2f} GB free of {mem[0]:.2f} GB, limit {limit_gb:.2f} GB")
        core.call("POST", "/bench/lock", {"holder": holder, "model": a.model, "max": 1})

        def warm(tag):  # load outside the measurement: a reload is not a slow step
            core.run(a.model, fn, item["inputs"], f"bench-warm-{fn}-{tag}-{os.getpid()}")

        warm(0)
        cancelled = 0

        def measure(c, floor):
            nonlocal cancelled
            while others_active(core, a.model):
                print("  real requests in flight: bench paused …")
                time.sleep(5)
            core.call("POST", "/bench/lock", {"holder": holder, "model": a.model, "max": c})
            row = measure_step(core, a.model, item, c, a.rounds, index, limit_gb, base_used_gb, floor, fn)
            if row["abort"] or row["errors"]:   # cancelling killed the worker
                cancelled += 1
                core.call("POST", "/bench/lock", {"holder": holder, "model": a.model, "max": 1})
                warm(cancelled)
            return row

        best, rows = choose_max(measure, cmax, limit_gb)
    finally:
        core.call("POST", "/bench/unlock")
    print(f"{len(rows)} steps measured")
    if best < 1:
        return None
    fit = sorted((r for r in rows if not tripped(r, limit_gb)), key=lambda r: r["c"])
    per_extra = round(vram_slope(fit), 3)
    if best > 1 and per_extra <= 0:
        per_extra = 0.05  # max > 1 requires vram_per_extra_gb > 0 (spec 4.2); measured growth was ~0
    return {"max": best, "vram_per_extra_gb": per_extra}


def main():
    ap = argparse.ArgumentParser(prog="harness-bench")
    ap.add_argument("model")
    ap.add_argument("--max", type=int, default=16, help="highest batch to consider")
    ap.add_argument("--rounds", type=int, default=2, help="batches per step")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--yes", action="store_true", help="save the recommendation without asking")
    ap.add_argument("--inputs", type=Path, help="bench.json to use instead of the plugin's tests/bench/bench.json")
    a = ap.parse_args()

    cfg = config_mod.load(Path(os.environ["HARNESS_CONFIG"]) if os.environ.get("HARNESS_CONFIG") else None)
    core = Core(a.port or cfg.port)
    if not core.call("GET", "/health").get("ok"):
        sys.exit("core is not running")
    reg = yaml.safe_load(cfg.registry.read_text(encoding="utf-8-sig"))
    entry = next((m for m in reg["models"] if m["name"] == a.model), None)
    if entry is None:
        sys.exit(f"{a.model} not in registry")
    model_dir = cfg.registry.parent / entry["folder"]
    manifest_path = cfg.registry.parent / entry["manifest"]
    items = load_inputs(model_dir, a.inputs)
    if core.call("GET", "/catalog")["models"].get(a.model) is None:
        sys.exit(f"{a.model} is not in the core catalog")
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8-sig"))
    variant = entry.get("variant", entry["name"]) if raw.get("variants") else None
    conc = raw.get("concurrency") or {}
    cmax = a.max if conc.get("batching") or conc.get("thread_safe") else 1
    device = cat_device(raw, variant)
    index = nvml.device_index(device)

    per_function: dict[str, dict] = {}
    for item in {it["function"]: it for it in items}.values():   # one fixed input per function
        print(f"\n=== {item['function']}")
        tuned = tune_function(core, a, item, cmax, index)
        if tuned is None:
            print(f"{item['function']}: not tuned")
        else:
            per_function[item["function"]] = tuned
    if not per_function:
        return 1
    print("\nrecommendation:")
    for fn, t in per_function.items():
        print(f"  {fn:10} max {t['max']:<3} vram_per_extra_gb {t['vram_per_extra_gb']}")
    where = tuning.path(cfg)
    if not a.yes:
        if input(f"save to {where} (this machine only)? [y/N] ").strip().lower() not in ("y", "yes"):
            print("not saved")
            return 0
    gpu = nvml.device_name(index) if device.startswith("cuda") else None
    tuning.save(cfg, a.model, per_function, gpu)
    r = core.call("POST", "/reload")
    print(f"saved to {where}; " + ("core reloaded" if "error" not in r else
                                   f"takes effect at next /reload ({r['error']['message']})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
