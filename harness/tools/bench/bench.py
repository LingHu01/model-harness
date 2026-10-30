"""Companion throughput tool (spec 15). Run manually against a running core:

    harness-bench <model> [--max 16] [--burst 20] [--seconds 30] [--port 8765] [--yes]

Inputs: <plugin folder>/tests/bench/bench.json, a list of {"function": ..., "inputs": {...}};
relative file paths inside inputs resolve against that folder.
For C = 1, 2, 4, 8 ... --max it runs a burst of ~20 requests (or 30 s) with C in flight, measuring throughput,
p50/p95 latency and VRAM peak. All steps run (only an OOM or a step that would exceed the GPU stops early);
the recommendation is then chosen from all results: throughput >= 90 % of the best, lowest p95, tie -> lower C.
Writes concurrency.max + vram_per_extra_gb to DATA_ROOT/state/tuning.yaml (per machine, untracked) ONLY after
explicit confirmation; the manifest is never edited. The core picks it up at once (/reload).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from core import config as config_mod  # noqa: E402
from core import tuning  # noqa: E402
from sdk import nvml  # noqa: E402


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

    def run(self, model, function, inputs, rid):
        body = json.dumps({"model": model, "function": function, "inputs": inputs, "request_id": rid}).encode()
        req = urllib.request.Request(self.base + "/run", body, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                return [json.loads(l) for l in r][-1]
        except urllib.error.HTTPError as e:
            return {"event": "error", **json.loads(e.read()).get("error", {})}


def load_inputs(model_dir: Path, spec: Path | None = None) -> list[dict]:
    """Tune on inputs like the real workload: VRAM per extra request scales with input size."""
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


def burst(core, model, items, c, n, seconds, tag):
    """n requests (or `seconds`) with c in flight. Returns (throughput, p50, p95, vram_peak, errors)."""
    lat, peaks, errors = [], [], []
    lock = threading.Lock()
    counter = iter(range(n))
    deadline = time.monotonic() + seconds

    def worker():
        while time.monotonic() < deadline:
            with lock:
                i = next(counter, None)
            if i is None:
                return
            it = items[i % len(items)]
            t = time.perf_counter()
            ev = core.run(model, it["function"], it["inputs"], f"bench-{tag}-{c}-{i}-{os.getpid()}")
            dt = time.perf_counter() - t
            with lock:
                if ev["event"] == "result":
                    lat.append(dt)
                    peaks.append(ev["metrics"].get("vram_peak_gb") or 0.0)
                else:
                    errors.append(ev.get("code"))

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(c)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    if not lat:
        return 0.0, None, None, None, errors
    lat.sort()
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
    return len(lat) / wall, statistics.median(lat), p95, max(peaks), errors


def linear_slope(xs, ys) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return max(0.0, sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den else 0.0


def recommend(rows):
    best = max(r["tput"] for r in rows)
    cands = [r for r in rows if r["tput"] >= 0.9 * best]
    return min(cands, key=lambda r: (r["p95"], r["c"]))


def cat_device(raw: dict, variant: str | None) -> str:
    rt = {**(raw.get("runtime") or {}), **((((raw.get("variants") or {}).get(variant) or {}).get("runtime")) or {})}
    return rt.get("device", "cuda:0")


def main():
    ap = argparse.ArgumentParser(prog="harness-bench")
    ap.add_argument("model")
    ap.add_argument("--max", type=int, default=16)
    ap.add_argument("--burst", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=30)
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
    cat = core.call("GET", "/catalog")["models"].get(a.model)
    if cat is None:
        sys.exit(f"{a.model} is not in the core catalog")
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8-sig"))
    conc = raw.get("concurrency") or {}
    can_parallel = conc.get("batching") or conc.get("thread_safe")
    cmax = a.max if can_parallel else 1
    gpu_total = next(iter(core.call("GET", "/status")["gpu"].values()), {}).get("total_gb")

    lock = core.call("POST", "/bench/lock", {"holder": f"bench:{a.model}"})
    if "error" in lock:
        sys.exit(lock["error"]["message"])
    rows = []
    try:
        cs = [1]
        while cs[-1] * 2 <= cmax:
            cs.append(cs[-1] * 2)
        print(f"bench {a.model}: C in {cs}, {a.burst} requests or {a.seconds:.0f} s per step, "
              f"{len(items)} distinct inputs, GPU {gpu_total} GB")
        for c in cs:
            while others_active(core, a.model):
                print("  real requests in flight: bench paused …")
                time.sleep(5)
            core.call("POST", "/bench/lock", {"holder": f"bench:{a.model}", "model": a.model, "max": c})
            core.call("POST", f"/unload/{a.model}")
            burst(core, a.model, items, 1, 1, a.seconds, f"warm{c}")  # load + warm-up outside the measurement
            tput, p50, p95, peak, errs = burst(core, a.model, items, c, a.burst, a.seconds, "m")
            if p50 is None:
                print(f"  C={c}: all requests failed {errs}")
                break
            rows.append({"c": c, "tput": tput, "p50": p50, "p95": p95, "vram": peak, "errors": len(errs)})
            print(f"  C={c:<3} {tput:7.2f} req/s  p50 {p50:6.2f}s  p95 {p95:6.2f}s  vram {peak:6.2f} GB"
                  + (f"  errors {len(errs)} {sorted(set(errs))}" if errs else ""))
            if errs and any(e == "oom" for e in errs):
                break
            if gpu_total and peak > gpu_total * 0.95:
                # Windows sysmem fallback: no OOM, VRAM spills to RAM over PCIe and everything crawls
                rows[-1]["spilled"] = True
                print(f"  stop: VRAM peak {peak:.2f} GB exceeds the card ({gpu_total} GB) -> spilled; C={c} invalid")
                break
            # every C runs; the choice is made afterwards from all rows. Only safety stops here.
            if len(rows) > 1 and c * 2 <= cmax:
                slope = linear_slope([r["c"] for r in rows], [r["vram"] for r in rows])
                if gpu_total and peak + slope * c > gpu_total * 0.95:
                    print("  stop: next step would exceed the GPU")
                    break
    finally:
        core.call("POST", "/bench/unlock")
        core.call("POST", f"/unload/{a.model}")

    per_extra = round(linear_slope([r["c"] for r in rows], [r["vram"] for r in rows]), 3)
    rows = [r for r in rows if not r.get("spilled") and not r["errors"]]
    if not rows:
        return 1
    rec = recommend(rows)
    if rec["c"] > 1 and per_extra <= 0:
        per_extra = 0.05  # max > 1 requires vram_per_extra_gb > 0 (spec 4.2); measured growth was ~0
    print(f"\nrecommendation: concurrency.max = {rec['c']}  (vram_per_extra_gb = {per_extra})")
    variant = entry.get("variant", entry["name"]) if raw.get("variants") else None
    eff = {**conc, **(((raw.get("variants") or {}).get(variant) or {}).get("concurrency") or {})} if variant else conc
    eff = {**eff, **(tuning.for_model(cfg, tuning.load(cfg), a.model, cat_device(raw, variant)) or {})}
    if rec["c"] == eff.get("max", 1) and per_extra == eff.get("vram_per_extra_gb", 0.0):
        print("already tuned to this; nothing to save")
        return 0
    where = tuning.path(cfg)
    if not a.yes:
        if input(f"save to {where} (this machine only)? [y/N] ").strip().lower() not in ("y", "yes"):
            print("not saved")
            return 0
    device = cat_device(raw, variant)
    gpu = nvml.device_name(nvml.device_index(device)) if device.startswith("cuda") else None
    tuning.save(cfg, a.model, rec["c"], per_extra, gpu)
    r = core.call("POST", "/reload")
    print(f"saved to {where}; " + ("core reloaded" if "error" not in r else
                                   f"takes effect at next /reload ({r['error']['message']})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
