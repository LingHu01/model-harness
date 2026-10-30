"""Queues, admission, eviction, concurrency, cancel and all timers (spec 10-12).

Single asyncio loop. Every timer is a scheduled event (call_later); nothing polls.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import time
import uuid
from dataclasses import dataclass, field

import psutil

from core import workers as W
from core.artifacts import Artifacts
from core.catalog import Catalog, ModelDef
from core.config import Config
from core.errors import HarnessError
from core.io import IO
from core.memory import Memory
from core.metrics import Store
from core.osal import osx
from sdk import nvml

log = logging.getLogger("scheduler")


@dataclass
class Request:
    id: str
    model: str
    function: str
    inputs: dict
    fingerprint: str = ""           # plugin version the inputs were validated against
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    state: str = "queued"           # queued | running | done | failed
    ts_in: float = field(default_factory=time.monotonic)
    ts_start: float | None = None
    last_pos: int = -1
    last_move: float = field(default_factory=time.monotonic)
    last_report: float = field(default_factory=time.monotonic)
    tick: asyncio.TimerHandle | None = None
    requeue: bool = False           # set when a batch-mate is cancelled: go back to the queue head
    loaded_now: dict | None = None  # cold-start info when this request triggered the load

    def emit(self, event: str, **kw):
        self.events.put_nowait({"event": event, **kw})


@dataclass
class ModelState:
    queue: collections.deque = field(default_factory=collections.deque)
    driver: asyncio.Task | None = None
    blocked_by: dict | None = None
    loading: bool = False
    last_error: str | None = None


class Scheduler:
    def __init__(self, cfg: Config, catalog: Catalog, store: Store, memory: Memory, io: IO, artifacts: Artifacts):
        self.cfg, self.catalog, self.store, self.memory, self.io, self.artifacts = (
            cfg, catalog, store, memory, io, artifacts)
        self.group = osx.ProcessGroup()
        self.workers: dict[str, W.Worker] = {}   # loaded (LOADING/HOT/BUSY)
        self.shells: dict[str, W.Worker] = {}    # SPAWNING/SHELL/PAGED
        self.models: dict[str, ModelState] = collections.defaultdict(ModelState)
        self.requests: dict[str, Request] = {}
        self.admission = asyncio.Lock()
        self.stopping = False
        self.bench_locked = False
        self._wake = asyncio.get_running_loop().create_future()
        self._bg: set[asyncio.Task] = set()

    # ---- wake-ups -------------------------------------------------------------------
    def wake(self):
        if not self._wake.done():
            self._wake.set_result(None)
        self._wake = asyncio.get_running_loop().create_future()

    async def wait_change(self):
        await asyncio.shield(self._wake)

    def _spawn_bg(self, coro):
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)
        return t

    # ---- submit -------------------------------------------------------------------------
    def submit(self, model: str, function: str, inputs: dict, request_id: str | None = None) -> Request:
        if self.stopping:
            raise HarnessError("shutting_down", "harness is stopping")
        m = self.catalog.models.get(model)
        if m is None:
            hint = f" ({self.catalog.invalid[model]})" if model in self.catalog.invalid else ""
            raise HarnessError("unknown_model", f"unknown model {model!r}{hint}")
        fn = m.functions.get(function)
        if fn is None:
            raise HarnessError("unknown_function", f"{model} has no function {function!r}; "
                                                   f"available: {', '.join(m.functions)}")
        rid = request_id or uuid.uuid4().hex[:12]
        if rid in self.requests:
            raise HarnessError("invalid_input", f"duplicate request_id {rid}")
        norm = self.io.normalise(rid, fn, inputs)
        req = Request(rid, model, function, norm, m.fingerprint)
        self.requests[rid] = req
        self.store.request_in(rid, model, function)
        st = self.models[model]
        st.queue.append(req)
        req.last_pos = self._position(req)
        req.emit("queued", position=req.last_pos, request_id=rid)
        self._schedule_tick(req)
        self._kick(model)
        return req

    def _kick(self, model: str):
        st = self.models[model]
        if st.driver is None or st.driver.done():
            st.driver = asyncio.create_task(self._drive(model))
        self.wake()

    def _position(self, req: Request) -> int:
        """Requests ahead of this one: queued before it + running on its model's worker."""
        st, w = self.models[req.model], self.workers.get(req.model)
        idx = st.queue.index(req) if req in st.queue else 0
        return idx + (len(w.running) if w else 0)

    # ---- per-request timers (R4) ------------------------------------------------------
    def _schedule_tick(self, req: Request):
        interval = min(self.cfg.queue_report, self.cfg.queue_stall)
        req.tick = asyncio.get_running_loop().call_later(interval, self._on_tick, req)

    def _on_tick(self, req: Request):
        if req.state != "queued":
            return
        st = self.models[req.model]
        now = time.monotonic()
        pos = self._position(req)
        if pos != req.last_pos:
            req.last_pos, req.last_move = pos, now
        w = self.workers.get(req.model)
        loading = st.loading or (w is not None and w.state in (W.LOADING, W.SPAWNING, W.RESUMING))
        if loading:
            req.last_move = now  # own model is loading: progress, not a stall
        if now - req.last_move >= self.cfg.queue_stall:
            blocked = st.blocked_by or ({"kind": "model", "detail": f"{req.model} busy"} if w and w.running
                                        else {"kind": "memory", "detail": "waiting for memory"})
            self._fail(req, HarnessError("queue_stalled", f"queue position {pos} has not moved for "
                                         f"{int(self.cfg.queue_stall)} s", queue_position=pos, blocked_by=blocked))
            self.wake()
            return
        if now - req.last_report >= self.cfg.queue_report - 0.01:
            req.last_report = now
            req.emit("loading" if loading else "queued", position=pos,
                     **({"blocked_by": st.blocked_by} if st.blocked_by else {}))
        self._schedule_tick(req)

    # ---- driver: one task per model with queued work ----------------------------------
    async def _drive(self, model: str):
        st = self.models[model]
        m = self.catalog.models[model]
        while st.queue and not self.stopping:
            m = self.catalog.models.get(model, m)  # a reload may swap it; a removed model drains its queue
            w = self.workers.get(model)
            if (w is not None and w.stale and w.state in (W.HOT, W.BUSY)
                    and st.queue[0].fingerprint != w.model.fingerprint):
                if w.running:          # requests accepted before the reload finish on the old version first
                    await self.wait_change()
                else:
                    await self._kill_worker(w, "reload", shell=False)
                continue
            if w is None or w.state == W.KILLED:
                self._rebase_queue(model)
                if not st.queue:
                    break
                try:
                    w = await self._admit_and_load(m)
                except HarnessError as e:
                    st.last_error = e.code
                    for req in list(st.queue):
                        self._fail(req, e)
                    break
                if w is None:          # waiting for busy workers to release memory
                    await self.wait_change()
                    continue
            if w.state not in (W.HOT, W.BUSY):
                await self.wait_change()
                continue
            dispatched = self._dispatch_ready(w.model, w)
            if not dispatched:
                st.blocked_by = {"kind": "model", "detail": f"{model} busy"} if w.running else st.blocked_by
                await self.wait_change()
        st.blocked_by = None

    def _slots(self, m: ModelDef) -> int:
        return m.max_concurrency if m.thread_safe and not m.batching else 1

    def _choose_k(self, m: ModelDef, queued: int) -> int:
        """Largest k <= max such that the extra VRAM for k fits what is free now (spec 11)."""
        k = min(queued, m.max_concurrency) if m.batching else 1
        if k <= 1 or m.vram_per_extra_gb <= 0:
            return max(1, k)
        gpu = self.memory.gpu(m.device)
        if gpu is None:
            return k
        free = gpu[1]
        while k > 1 and (k - 1) * m.vram_per_extra_gb > free:
            k -= 1
        return k

    def _dispatch_ready(self, m: ModelDef, w: W.Worker) -> bool:
        st = self.models[m.name]
        did = False
        while st.queue and w.state in (W.HOT, W.BUSY):
            if m.batching:
                if w.running:
                    break
                k = self._choose_k(m, len(st.queue))
            else:
                if len(w.running) >= self._slots(m):
                    break
                if w.running and m.vram_per_extra_gb > 0:
                    gpu = self.memory.gpu(m.device)
                    if gpu is not None and gpu[1] < m.vram_per_extra_gb:
                        break
                k = 1
            # a batch is one run_batch call: only consecutive requests for the same function
            # and only requests validated against the version this worker runs
            fn = st.queue[0].function
            batch = []
            while (st.queue and len(batch) < k and st.queue[0].function == fn
                   and st.queue[0].fingerprint == m.fingerprint):
                batch.append(st.queue.popleft())
            if not batch:
                break
            self._run_batch(m, w, batch)
            did = True
        return did

    def _run_batch(self, m: ModelDef, w: W.Worker, batch: list[Request]):
        now = time.monotonic()
        for r in batch:
            r.state, r.ts_start = "running", now
            if r.tick:
                r.tick.cancel()
            self.store.request_start(r.id)
            r.emit("running")
        if w.idle_handle:
            w.idle_handle.cancel()
            w.idle_handle = None
        out_dirs = [self.io.out_dir(r.id) for r in batch]
        busy_others = self._busy_on_device(m.device, exclude=m.name)
        futs = w.run([r.id for r in batch], batch[0].function, [r.inputs for r in batch], out_dirs)
        runaway = asyncio.get_running_loop().call_later(m.runaway, self._on_runaway, w, batch)
        self._spawn_bg(self._await_batch(m, w, batch, futs, runaway, busy_others))
        # others waiting on this queue position moved
        for r in self.models[m.name].queue:
            r.last_pos, r.last_move = self._position(r), now

    async def _await_batch(self, m, w, batch, futs, runaway, busy_others_start):
        results = await asyncio.gather(*futs)
        runaway.cancel()
        ext_valid = not busy_others_start and not self._busy_on_device(m.device, exclude=m.name)
        recorded = False
        for req, res in zip(batch, results):
            if res.get("type") == "result":
                metrics = dict(res.get("metrics") or {})
                metrics["queue_wait_s"] = round(req.ts_start - req.ts_in, 3)
                if req.loaded_now:
                    metrics["cold_start"] = {**req.loaded_now, **(metrics.get("cold_start") or {})}
                if not recorded:
                    self.store.add_run(m.name, req.function, metrics, m.plugin_version,
                                       self.memory.whash(m), ext_valid)
                    recorded = True
                req.state = "done"
                self.store.request_end(req.id)
                req.emit("result", result=res.get("result"), metrics=metrics)
                self.io.remove_inputs(req.id)
                self._forget_later(req)
            elif req.requeue and req.state == "running":
                req.requeue, req.state = False, "queued"
                self.models[m.name].queue.appendleft(req)
                self._schedule_tick(req)
            elif req.state == "running":
                code = res.get("code", "worker_error")
                metrics = res.get("metrics")
                if code == "oom" and metrics and metrics.get("vram_peak_gb"):
                    self.memory.raise_need(m, metrics["vram_peak_gb"] * (1 + self.cfg.need_margin))
                self._fail(req, HarnessError(code, res.get("message", code), metrics=metrics))
        if w.state in (W.HOT,) and not w.running and not self.models[m.name].queue:
            if w.stale:
                self._spawn_bg(self._kill_worker(w, "reload"))
            else:
                self._arm_idle(w)
        self._kick(m.name) if self.models[m.name].queue else self.wake()

    def _busy_on_device(self, device: str, exclude: str) -> bool:
        return any(o.running for n, o in self.workers.items() if n != exclude and o.model.device == device)

    # ---- R5 runaway --------------------------------------------------------------------
    def _on_runaway(self, w: W.Worker, batch: list[Request]):
        log.warning("runaway: %s exceeded %.0f s", w.name, w.model.runaway)
        err = HarnessError("runaway", f"{w.name}: request exceeded {int(w.model.runaway)} s; worker killed")
        for r in batch:
            self._fail(r, err)
        self._spawn_bg(self._kill_worker(w, "runaway"))

    # ---- R1 idle / R2 shell expiry -------------------------------------------------------
    def _arm_idle(self, w: W.Worker):
        if w.idle_handle:
            w.idle_handle.cancel()
        w.idle_handle = asyncio.get_running_loop().call_later(self.cfg.idle_timeout, self._on_idle, w)

    def _on_idle(self, w: W.Worker):
        if w.idle and not self.models[w.name].queue and self.workers.get(w.name) is w:
            log.info("idle timeout: %s", w.name)
            self._spawn_bg(self._kill_worker(w, "idle"))

    async def _kill_worker(self, w: W.Worker, reason: str, shell: bool = True):
        """Kill a loaded worker; replace it with a paged warm shell (R1)."""
        if self.workers.get(w.name) is w:
            del self.workers[w.name]
        await w.kill(reason)
        self.wake()
        if shell and not self.stopping and w.name in self.catalog.models:
            self._spawn_bg(self._make_shell(self.catalog.models[w.name]))

    async def _make_shell(self, m: ModelDef):
        if m.name in self.shells or self.stopping:
            return
        try:
            await self.artifacts.ensure_venv(m)
            sh = W.Worker(self.cfg, m, self.artifacts.python(m), self.group)
            sh.on_exit = self._on_shell_exit
            self.shells[m.name] = sh
            await sh.spawn()
            if self.shells.get(m.name) is sh and sh.state == W.SHELL:
                sh.page()
                sh.expiry_handle = asyncio.get_running_loop().call_later(
                    self.cfg.shell_expiry, self._on_shell_expiry, sh)
                log.info("warm shell for %s %s (pid %s)", m.name, sh.state, sh.pid)
        except Exception as e:
            log.warning("shell for %s failed: %s", m.name, e)
            self.shells.pop(m.name, None)

    def _on_shell_expiry(self, sh: W.Worker):
        if self.shells.get(sh.name) is sh:
            log.info("shell expiry: %s", sh.name)
            del self.shells[sh.name]
            self._spawn_bg(sh.kill("expired"))

    def _on_shell_exit(self, sh: W.Worker):
        if self.shells.get(sh.name) is sh:
            del self.shells[sh.name]

    def _on_worker_exit(self, w: W.Worker):
        """Unexpected runner exit while loaded: requests already failed via futures; model gets a shell."""
        if self.workers.get(w.name) is w:
            del self.workers[w.name]
            log.warning("%s exited unexpectedly (%s)", w.name, w.kill_reason or "crash")
            if not self.stopping and w.kill_reason is None and w.name in self.catalog.models:
                self._spawn_bg(self._make_shell(self.catalog.models[w.name]))
        self.wake()

    # ---- admission + eviction (spec 12) ------------------------------------------------------
    async def _admit_and_load(self, m: ModelDef) -> W.Worker | None:
        st = self.models[m.name]
        async with self.admission:
            existing = self.workers.get(m.name)
            if existing and existing.state != W.KILLED:
                return existing
            st.loading = True
            try:
                for q in st.queue:
                    q.emit("loading")
                await self.artifacts.ensure_venv(m)
                for attempt in (1, 2):
                    plan = await self._make_room(m)
                    if plan is None:
                        return None
                    try:
                        return await self._start_worker(m, plan)
                    except HarnessError as e:
                        if e.code == "oom" and attempt == 1:
                            peak = (e.extra.get("metrics") or {}).get("vram_peak_gb")
                            need = self.memory.need(m).vram_gb
                            self.memory.raise_need(m, max(peak or 0, need * 1.5))
                            log.warning("%s: load OOM (bootstrap too low); retrying once", m.name)
                            continue
                        raise
            finally:
                st.loading = False

    async def _make_room(self, m: ModelDef) -> dict | None:
        """R3/R6: read real free memory now; evict idle LRU workers until M fits."""
        st = self.models[m.name]
        while True:
            need = self.memory.need(m)
            gpu = self.memory.gpu(m.device)
            free_ram = self.memory.free_ram_gb()
            fits_vram = gpu is None or need.vram_gb <= gpu[1]
            fits_ram = need.ram_gb <= free_ram
            if fits_vram and fits_ram:
                st.blocked_by = None
                return {"vram_budget_gb": gpu[1] if gpu else 0.0, "ram_budget_gb": free_ram * 0.9,
                        "offload": "none" if m.offload == "none" else m.offload, "need": need}
            # too large = bigger than the card itself; a measured peak already ran here, so use it unmargined
            if gpu is not None and (need.raw_vram_gb if need.measured else need.vram_gb) > gpu[0] * 0.98:
                if m.offload != "none":
                    # larger than the card: load partially offloaded within budgets (spec 9.3)
                    idle = self._idle_lru(exclude=m.name)
                    if idle:
                        await self._kill_worker(idle[0], "evicted")
                        continue
                    return {"vram_budget_gb": gpu[1] * 0.9, "ram_budget_gb": free_ram * 0.9,
                            "offload": "partial", "need": need}
                raise HarnessError("model_too_large", f"{m.name} needs {need.vram_gb:.1f} GB VRAM; card has "
                                                      f"{gpu[0]:.1f} GB and offload is disabled")
            idle = self._idle_lru(exclude=m.name)
            if idle:
                log.info("pressure eviction: %s for %s", idle[0].name, m.name)
                await self._kill_worker(idle[0], "evicted")
                continue
            busy = [w.name for w in self.workers.values() if w.name != m.name]
            if busy:
                st.blocked_by = {"kind": "memory", "detail": f"waiting for {', '.join(busy)} to finish"}
                return None
            if fits_ram and need.measured and gpu is not None and need.raw_vram_gb <= gpu[1] + 1.0:
                # nothing of ours is left to evict and the unmargined measured peak (nearly) fits: it ran
                # like this before, so load as a tight fit rather than blaming an external holder
                log.warning("%s: tight fit (need %.2f GB incl. margin, %.2f GB free)", m.name, need.vram_gb, gpu[1])
                st.blocked_by = None
                return {"vram_budget_gb": gpu[1], "ram_budget_gb": free_ram * 0.9,
                        "offload": "none" if m.offload == "none" else m.offload, "need": need}
            raise self._external_holder(m, need, gpu, free_ram, fits_vram)

    def _idle_lru(self, exclude: str) -> list[W.Worker]:
        idle = [w for n, w in self.workers.items() if n != exclude and w.idle
                and not (w.stale and self.models[n].queue)]  # the only process left for its queued requests
        return sorted(idle, key=lambda w: w.last_used)

    def _external_holder(self, m, need, gpu, free_ram, fits_vram) -> HarnessError:
        if not fits_vram:
            ours = {w.pid for w in list(self.workers.values()) + list(self.shells.values())}
            holders = []
            for pid, used in nvml.compute_processes(nvml.device_index(m.device)):
                if pid in ours:
                    continue
                try:
                    name = psutil.Process(pid).name()
                except psutil.Error:
                    name = "?"
                holders.append(f"{name} (pid {pid}{f', {used:.1f} GB' if used else ''})")
            detail = ", ".join(holders) or "held outside the harness"
            return HarnessError("external_memory_holder",
                                f"{m.name} needs {need.vram_gb:.1f} GB VRAM, {gpu[1]:.1f} GB free; {detail}",
                                blocked_by={"kind": "external", "detail": detail})
        return HarnessError("external_memory_holder",
                            f"{m.name} needs {need.ram_gb:.1f} GB RAM, {free_ram:.1f} GB available; "
                            "held outside the harness", blocked_by={"kind": "external", "detail": "RAM"})

    async def _start_worker(self, m: ModelDef, plan: dict) -> W.Worker:
        t0 = time.perf_counter()
        sh = self.shells.pop(m.name, None)
        w, resume_s = None, None
        if sh is not None:
            if sh.state == W.SPAWNING:
                try:
                    await asyncio.wait_for(sh._shell_ready, 300)
                    sh.state = W.SHELL if sh.state == W.SPAWNING else sh.state
                except Exception:
                    pass
            if sh.state in (W.SHELL, W.PAGED):
                if sh.expiry_handle:
                    sh.expiry_handle.cancel()
                try:
                    resume_s = sh.resume()
                    w = sh
                except OSError as e:
                    log.warning("resume of %s failed (%s); fresh spawn", m.name, e)
                    await sh.kill("resume_failed")
            else:
                await sh.kill("bad_shell")
        if w is None:
            w = W.Worker(self.cfg, m, self.artifacts.python(m), self.group)
            await w.spawn()
        w.on_exit = self._on_worker_exit
        w.on_oom_recovered = lambda wk, msg: log.warning("%s: OOM recovered (%s)", wk.name, msg.get("step"))
        self.workers[m.name] = w
        msg = {"weights_path": m.weights_path, "device": m.device,
               "vram_budget_gb": round(plan["vram_budget_gb"], 3), "ram_budget_gb": round(plan["ram_budget_gb"], 3),
               "strategy": "standard", "engine_path": None, "offload": plan["offload"],
               "thread_safe": m.thread_safe, "max_concurrency": m.max_concurrency}
        if resume_s is not None:
            msg["resume_s"] = resume_s
        try:
            await w.load(msg)
        except HarnessError:
            if self.workers.get(m.name) is w:
                del self.workers[m.name]
            await w.kill("load_failed")
            raise
        w.cold_start["total_s"] = round(time.perf_counter() - t0, 3)
        w.cold_start["from_shell"] = resume_s is not None
        for q in self.models[m.name].queue:
            if q.loaded_now is None:
                q.loaded_now = dict(w.cold_start)
                break
        log.info("%s loaded in %.2f s (%s)", m.name, time.perf_counter() - t0,
                 "warm shell" if resume_s is not None else "cold")
        return w

    # ---- failure / cancel / unload ---------------------------------------------------------
    def _fail(self, req: Request, err: HarnessError):
        if req.state in ("done", "failed"):
            return
        st = self.models[req.model]
        if req in st.queue:
            st.queue.remove(req)
        if req.tick:
            req.tick.cancel()
        req.state = "failed"
        self.store.request_end(req.id, err.code)
        req.emit("error", **err.to_dict(req.id)["error"])
        self.io.remove_inputs(req.id)
        self._forget_later(req)

    def _forget_later(self, req: Request):
        asyncio.get_running_loop().call_later(600, self.requests.pop, req.id, None)

    async def cancel(self, rid: str) -> dict:
        req = self.requests.get(rid)
        if req is None or req.state in ("done", "failed"):
            raise HarnessError("invalid_input", f"no active request {rid}")
        err = HarnessError("cancelled", f"request {rid} cancelled")
        if req.state == "queued":
            self._fail(req, err)
            self.wake()
            return {"cancelled": rid, "was": "queued"}
        w = self.workers.get(req.model)
        if w is not None and rid in w.running:
            for other in w.running:
                if other != rid and other in self.requests:
                    self.requests[other].requeue = True
            self._fail(req, err)
            await self._kill_worker(w, "cancelled")
            self._kick(req.model)
        return {"cancelled": rid, "was": "running"}

    async def unload(self, model: str, force: bool = False) -> dict:
        if model not in self.catalog.models:
            raise HarnessError("unknown_model", f"unknown model {model!r}")
        w = self.workers.get(model)
        if w is None:
            return {"unloaded": model, "was": "not loaded"}
        st = self.models[model]
        if (w.running or st.queue or w.state == W.LOADING) and not force:
            raise HarnessError("busy", f"{model} is busy ({len(w.running)} running, {len(st.queue)} queued); "
                                       "use force=true to cancel its requests")
        err = HarnessError("cancelled", f"{model} unloaded")
        for r in list(st.queue):
            self._fail(r, err)
        for rid in list(w.running):
            if rid in self.requests:
                self._fail(self.requests[rid], err)
        await self._kill_worker(w, "unloaded")
        return {"unloaded": model, "was": "loaded"}

    # ---- catalog reload ---------------------------------------------------------------------------
    def _rebase_queue(self, model: str):
        """Queued requests whose version has no process left (it crashed): that code is gone from disk,
        so each one moves to the current version if its inputs are still valid there, else it fails."""
        st, m = self.models[model], self.catalog.models.get(model)
        for req in list(st.queue):
            if m is not None and req.fingerprint == m.fingerprint:
                continue
            try:
                fn = m.functions.get(req.function) if m else None
                if fn is None:
                    raise HarnessError("plugin_changed", f"{model}.{req.function} no longer exists")
                self.io.normalise(req.id, fn, req.inputs)
                req.fingerprint = m.fingerprint
            except HarnessError as e:
                self._fail(req, HarnessError("plugin_changed", f"{model} changed while this request was queued "
                                                               f"and the old version is gone: {e.message}"))

    def sync_catalog(self, old: dict[str, ModelDef]) -> dict:
        """After a catalog swap: untouched models keep their processes; changed or removed ones retire once
        every request accepted before the reload is done (running or queued, on the old code).
        Requests accepted after the reload wait for them, then get a new process."""
        self.memory.forget_hashes()
        new = self.catalog.models
        removed = sorted(set(old) - set(new))
        changed = sorted(n for n in set(old) & set(new) if old[n].fingerprint != new[n].fingerprint)
        for name in removed + changed:
            sh = self.shells.pop(name, None)
            if sh:
                self._spawn_bg(sh.kill("reload"))
            w = self.workers.get(name)
            if w:
                w.stale = True
                if w.idle and not self.models[name].queue:
                    self._spawn_bg(self._kill_worker(w, "reload"))
        for name, m in new.items():  # same process, fresh limits (tuning, runaway)
            if name not in changed:
                for w in (self.workers.get(name), self.shells.get(name)):
                    if w:
                        w.model = m
        self.wake()
        return {"added": sorted(set(new) - set(old)), "removed": removed, "changed": changed}

    # ---- bench override (tools/bench) ---------------------------------------------------------------
    def bench_override(self, model: str, max_concurrency: int):
        """Temporarily run `model` at a given concurrency (manifest untouched). Takes effect at next load."""
        m = self.catalog.models.get(model)
        if m is None:
            raise HarnessError("unknown_model", f"unknown model {model!r}")
        if max_concurrency > 1 and not (m.batching or m.thread_safe):
            raise HarnessError("invalid_input", f"{model} declares neither batching nor thread_safe")
        self._bench_saved = getattr(self, "_bench_saved", {})
        self._bench_saved.setdefault(model, m.max_concurrency)
        m.max_concurrency = max(1, int(max_concurrency))
        self.wake()

    def bench_restore(self):
        for model, cmax in getattr(self, "_bench_saved", {}).items():
            if model in self.catalog.models:
                self.catalog.models[model].max_concurrency = cmax
        self._bench_saved = {}

    # ---- status ---------------------------------------------------------------------------------
    def status(self) -> dict:
        out = {"models": {}, "invalid": self.catalog.invalid}
        for name, m in self.catalog.models.items():
            w, sh, st = self.workers.get(name), self.shells.get(name), self.models.get(name)
            need = self.memory.need(m)
            out["models"][name] = {
                "state": (w.state if w else sh.state if sh else "GONE"),
                "pid": (w.pid if w else sh.pid if sh else None),
                "running": len(w.running) if w else 0,
                "queued": len(st.queue) if st else 0,
                "placement": w.placement if w else None,
                "cold_start": w.cold_start if w else None,
                "need": {"vram_gb": need.vram_gb, "ram_gb": need.ram_gb, "measured": need.measured},
                "concurrency": {"max": m.max_concurrency, "batching": m.batching, "thread_safe": m.thread_safe},
                "compile": self.store.compile_state(name, self.memory.key(m)) or {"state": "off",
                                                                                   "reason": "not built (phase 7)"},
                "flags": self.memory.flags(m),
                "venv": "ready" if self.artifacts.venv_valid(m) else "build on first request",
                "blocked_by": st.blocked_by if st else None,
                "last_error": st.last_error if st else None,
            }
        devices = {}
        for dev in sorted({m.device for m in self.catalog.models.values() if m.device.startswith("cuda")}):
            g = self.memory.gpu(dev)
            if g:
                devices[dev] = {"total_gb": round(g[0], 2), "free_gb": round(g[1], 2),
                                "name": nvml.device_name(nvml.device_index(dev))}
        out["gpu"] = devices
        out["ram_free_gb"] = round(self.memory.free_ram_gb(), 2)
        return out

    # ---- shutdown -------------------------------------------------------------------------------
    async def shutdown(self):
        self.stopping = True
        err = HarnessError("cancelled", "harness stopping")
        for st in self.models.values():
            for r in list(st.queue):
                self._fail(r, err)
        deadline = time.monotonic() + self.cfg.shutdown_grace
        while any(w.running for w in self.workers.values()) and time.monotonic() < deadline:
            try:
                await asyncio.wait_for(self.wait_change(), max(0.1, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                break
        for r in self.requests.values():
            if r.state == "running":
                self._fail(r, err)
        await asyncio.gather(*(w.kill("shutdown") for w in list(self.workers.values()) + list(self.shells.values())),
                             return_exceptions=True)
        self.workers.clear()
        self.shells.clear()
