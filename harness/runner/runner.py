"""Generic worker. Runs inside a plugin's own .venv; speaks JSON lines with the core.

stdin  (core -> runner): load | run | shutdown
stdout (runner -> core): shell_ready | loaded | result | error | oom_recovered
stderr: logs (the core redirects it to logs/<model>.log)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

_T0 = time.perf_counter()

# Keep the protocol channel private: anything plugins print goes to stderr.
_proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HARNESS_ROOT not in sys.path:
    sys.path.insert(0, HARNESS_ROOT)

import sdk  # noqa: E402
from sdk import nvml, procmem  # noqa: E402

sdk.setup_offline_env()

_send_lock = threading.Lock()


def send(msg: dict) -> None:
    line = json.dumps(msg, default=str)
    with _send_lock:
        _proto.write(line + "\n")
        _proto.flush()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def _torch():
    return sys.modules.get("torch")


def _cuda_ok(device: str) -> bool:
    t = _torch()
    return bool(t is not None and device.startswith("cuda") and t.cuda.is_available())


class ExtSampler:
    """Samples device memory every N ms during ONE request; catches non-PyTorch allocators."""

    def __init__(self, index: int, interval_s: float):
        self.index, self.interval = index, interval_s
        self.base = self.peak = None
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        m = nvml.memory(self.index)
        if m is not None:
            self.base = self.peak = m[2]
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def _loop(self):
        while not self._stop.wait(self.interval):
            m = nvml.memory(self.index)
            if m is not None and m[2] > self.peak:
                self.peak = m[2]

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(1)
            m = nvml.memory(self.index)
            if m is not None and m[2] > self.peak:
                self.peak = m[2]

    @property
    def delta_gb(self):
        return None if self.base is None else max(0.0, self.peak - self.base)


class Runner:
    def __init__(self, args):
        self.args = args
        self.adapter = None
        self.ctx: sdk.Context | None = None
        self.ctx_overhead_gb = 0.0
        self.sampler_s = args.sampler_ms / 1000.0
        self.pool: ThreadPoolExecutor | None = None
        self.thread_safe = False
        self.first_request = True
        self.cold_start: dict = {}

    # ---- startup ------------------------------------------------------------
    def import_adapter(self):
        t = time.perf_counter()
        path = os.path.join(self.args.plugin_dir, self.args.entry)
        sys.path.insert(0, self.args.plugin_dir)
        spec = importlib.util.spec_from_file_location(f"plugin_{self.args.model.replace('-', '_')}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, "Adapter", None)
        if cls is None:
            cls = next(v for v in vars(mod).values()
                       if isinstance(v, type) and issubclass(v, sdk.BaseAdapter) and v is not sdk.BaseAdapter)
        self.adapter = cls()
        self.cold_start["imports_s"] = round(time.perf_counter() - _T0, 3)
        self.cold_start["adapter_import_s"] = round(time.perf_counter() - t, 3)
        send({"type": "shell_ready", "pid": os.getpid(), "imports_s": self.cold_start["imports_s"]})

    # ---- load ---------------------------------------------------------------
    def on_load(self, m: dict):
        t_resume = m.get("resume_s")
        if t_resume is not None:
            self.cold_start["resume_s"] = t_resume
        self.thread_safe = bool(m.get("thread_safe"))
        if self.thread_safe and int(m.get("max_concurrency", 1)) > 1:
            self.pool = ThreadPoolExecutor(int(m["max_concurrency"]), thread_name_prefix="req")
        self.ctx = sdk.Context(
            weights_path=m.get("weights_path", ""), device=m.get("device", "cuda:0"),
            vram_budget_gb=float(m.get("vram_budget_gb", 0)), ram_budget_gb=float(m.get("ram_budget_gb", 0)),
            offload=m.get("offload", "partial"), strategy=m.get("strategy", "standard"),
            engine_path=m.get("engine_path"), plugin_dir=self.args.plugin_dir, cache_dir=self.args.cache_dir)
        dev_idx = nvml.device_index(self.ctx.device)
        torch = _torch()
        try:
            t = time.perf_counter()
            if _cuda_ok(self.ctx.device):
                before = nvml.memory(dev_idx)
                torch.cuda.set_device(dev_idx)
                torch.zeros(1, device=self.ctx.device)
                torch.cuda.synchronize()
                after = nvml.memory(dev_idx)
                if before and after:
                    self.ctx_overhead_gb = max(0.0, after[2] - before[2] - torch.cuda.memory_reserved() / 1024 ** 3)
            self.cold_start["cuda_init_s"] = round(time.perf_counter() - t, 3)

            t = time.perf_counter()
            self.adapter.load(self.ctx)
            self.cold_start["weights_s"] = round(time.perf_counter() - t, 3)

            t = time.perf_counter()
            warm = getattr(self.adapter, "warmup", None)
            if callable(warm):
                warm(self.ctx)
            self.cold_start["warmup_s"] = round(time.perf_counter() - t, 3)
        except Exception as e:
            code = "oom" if sdk.is_cuda_oom(e) else getattr(e, "code", "load_failed")
            log("load failed:", traceback.format_exc())
            send({"type": "error", "code": code, "message": f"load: {e}",
                  "metrics": {"vram_peak_gb": self._torch_peak()}})
            return
        send({"type": "loaded", "cold_start": self.cold_start, "placement": self._placement(),
              "ctx_overhead_gb": round(self.ctx_overhead_gb, 3)})

    def _placement(self):
        p = getattr(self.adapter, "placement", None)
        if isinstance(p, dict):
            return p
        torch = _torch()
        on_gpu = torch.cuda.memory_allocated() / 1024 ** 3 if _cuda_ok(self.ctx.device) else 0.0
        return {"on_gpu_gb": round(on_gpu, 3), "in_ram_gb": 0.0, "offload_mode": "none"}

    def _torch_peak(self):
        torch = _torch()
        if self.ctx and _cuda_ok(self.ctx.device):
            return round(torch.cuda.max_memory_reserved() / 1024 ** 3 + self.ctx_overhead_gb, 3)
        return None

    # ---- run ----------------------------------------------------------------
    def on_run(self, m: dict):
        if self.pool is not None and len(m["ids"]) == 1:
            self.pool.submit(self._run_safe, m)
        else:
            self._run_safe(m)

    def _run_safe(self, m):
        try:
            self._run(m)
        except Exception as e:
            log("run crashed:", traceback.format_exc())
            for rid in m["ids"]:
                send({"type": "error", "id": rid, "code": getattr(e, "code", "worker_error"), "message": str(e)})

    def _run(self, m: dict):
        ids, inputs, fns = m["ids"], m["inputs"], m["functions"]
        ctx = sdk.Context(**{**self.ctx.__dict__, "out_dir": m["out_dir"],
                             "stage": lambda name: send({"type": "stage", "ids": ids, "name": name})})
        torch = _torch()
        cuda = _cuda_ok(ctx.device)
        others_gb = 0.0
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            dev = nvml.memory(nvml.device_index(ctx.device))
            if dev:  # VRAM other programs hold: a spill starts at total minus this, not at the card's total
                others_gb = max(0.0, dev[2] - torch.cuda.memory_reserved() / 1024 ** 3 - self.ctx_overhead_gb)
        recoveries: list[str] = []
        t = time.perf_counter()
        with ExtSampler(nvml.device_index(ctx.device), self.sampler_s) as samp:
            results = self._call_with_recovery(fns, ids, inputs, ctx, recoveries)
        duration = time.perf_counter() - t
        metrics = {
            "vram_peak_gb": self._torch_peak(),
            "vram_peak_ext_gb": None if samp.delta_gb is None else round(samp.delta_gb, 3),
            "ram_peak_gb": _r(procmem.peak_private_gb()),
            "vram_after_gb": _r(torch.cuda.memory_allocated() / 1024 ** 3) if cuda else None,
            "duration_s": round(duration, 3),
            "k": len(ids),
            "placement": self._placement(),
            "oom_recoveries": recoveries,
        }
        dev = nvml.memory(nvml.device_index(ctx.device)) if cuda else None
        if dev and metrics["vram_peak_gb"] and metrics["vram_peak_gb"] + others_gb > dev[0]:
            metrics["vram_spill"] = True  # Windows sysmem fallback: ran past the card's VRAM, very slow
        if cuda:  # an idle model holds its weights, not the allocator's cache of its last request
            torch.cuda.empty_cache()
        if self.first_request:
            metrics["cold_start"] = self.cold_start
            self.first_request = False
        for rid, res in zip(ids, results):
            if isinstance(res, Exception):
                code = "oom" if sdk.is_cuda_oom(res) else getattr(res, "code", "worker_error")
                send({"type": "error", "id": rid, "code": code, "message": str(res), "metrics": metrics})
            else:
                send({"type": "result", "id": rid, "result": res, "metrics": metrics})

    def _call(self, fns, inputs, ctx):
        if len(inputs) == 1:
            return [self.adapter.run(fns[0], inputs[0], ctx)]
        if len(set(fns)) == 1:
            out = self.adapter.run_batch(fns[0], inputs, ctx)
        else:
            out = self.adapter.run_mixed(fns, inputs, ctx)
        if len(out) != len(inputs):
            raise RuntimeError("run_batch returned wrong number of results")
        return out

    def _call_with_recovery(self, fns, ids, inputs, ctx, recoveries, depth=0):
        """OOM ladder (spec 9.4): empty cache -> split batch -> offload more -> fail."""
        torch = _torch()
        try:
            return self._call(fns, inputs, ctx)
        except Exception as e:
            if not sdk.is_cuda_oom(e):
                if len(inputs) == 1:
                    return [e]
                raise
        # 1. empty allocator cache
        torch.cuda.empty_cache()
        recoveries.append("empty_cache")
        send({"type": "oom_recovered", "id": ids[0], "step": "empty_cache"})
        try:
            return self._call(fns, inputs, ctx)
        except Exception as e:
            if not sdk.is_cuda_oom(e):
                return [e] * len(inputs)
        # 2. split the batch
        if len(inputs) > 1:
            recoveries.append(f"split:{len(inputs)}")
            send({"type": "oom_recovered", "id": ids[0], "step": f"split:{len(inputs)}"})
            h = len(inputs) // 2
            return (self._call_with_recovery(fns[:h], ids[:h], inputs[:h], ctx, recoveries, depth + 1)
                    + self._call_with_recovery(fns[h:], ids[h:], inputs[h:], ctx, recoveries, depth + 1))
        # 3. offload more layers (adapter opt-in)
        more = getattr(self.adapter, "offload_more", None)
        if callable(more) and ctx.offload != "none":
            try:
                more(ctx)
                torch.cuda.empty_cache()
                recoveries.append("offload_more")
                send({"type": "oom_recovered", "id": ids[0], "step": "offload_more"})
                return self._call(fns, inputs, ctx)
            except Exception as e:
                return [e]
        # 4. fail
        err = sdk.PluginError("oom", f"CUDA out of memory; peak {self._torch_peak()} GB")
        return [err]

    # ---- main loop ------------------------------------------------------------
    def serve(self):
        self.import_adapter()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                log("bad message:", line[:200])
                continue
            kind = m.get("type")
            if kind == "load":
                self.on_load(m)
            elif kind == "run":
                self.on_run(m)
            elif kind == "shutdown":
                break
        try:
            self.adapter.unload()
        except Exception:
            pass
        if self.pool:
            self.pool.shutdown(wait=True)


def _r(x):
    return None if x is None else round(x, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--plugin-dir", required=True)
    ap.add_argument("--entry", default="adapter.py")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--sampler-ms", type=int, default=75)
    args = ap.parse_args()
    try:
        Runner(args).serve()
    except Exception as e:
        log("fatal:", traceback.format_exc())
        send({"type": "error", "code": "worker_crashed", "message": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
