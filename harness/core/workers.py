"""Runner processes and their state machine (spec 10.1).

GONE -> SPAWNING -> SHELL -> PAGED <-> RESUMING -> LOADING -> HOT <-> BUSY -> KILLED
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from core.catalog import ModelDef
from core.config import HARNESS_ROOT, Config
from core.errors import HarnessError
from core.osal import osx

log = logging.getLogger("workers")

SPAWNING, SHELL, PAGED, RESUMING, LOADING, HOT, BUSY, KILLED = (
    "SPAWNING", "SHELL", "PAGED", "RESUMING", "LOADING", "HOT", "BUSY", "KILLED")

RUNNER = HARNESS_ROOT / "runner" / "runner.py"
STREAM_LIMIT = 64 * 1024 * 1024  # results are JSON lines; big outputs go to files, but OCR lists can be long


def _rotate(path: Path, max_bytes: int, backups: int):
    try:
        if path.exists() and path.stat().st_size > max_bytes:
            for i in range(backups - 1, 0, -1):
                src = path.with_suffix(path.suffix + f".{i}")
                if src.exists():
                    src.replace(path.with_suffix(path.suffix + f".{i + 1}"))
            path.replace(path.with_suffix(path.suffix + ".1"))
    except OSError:
        pass


class Worker:
    def __init__(self, cfg: Config, model: ModelDef, python: str, group):
        self.cfg, self.model, self.python, self.group = cfg, model, python, group
        self.name = model.name
        self.state = "GONE"
        self.proc: asyncio.subprocess.Process | None = None
        self.pid: int | None = None
        self.running: dict[str, asyncio.Future] = {}   # request id -> future(result msg)
        self.last_used = time.monotonic()
        self.idle_handle: asyncio.TimerHandle | None = None
        self.expiry_handle: asyncio.TimerHandle | None = None
        self.kill_reason: str | None = None
        self.stale = False           # catalog reload changed or removed this model: retire when idle
        self.placement: dict = {}
        self.cold_start: dict = {}
        self.ctx_overhead_gb = 0.0
        self.exited = asyncio.Event()
        self._shell_ready: asyncio.Future | None = None
        self._loaded: asyncio.Future | None = None
        self._reader: asyncio.Task | None = None
        self._logf = None
        self.on_exit = None       # callback(worker)
        self.on_oom_recovered = None
        self.on_stage = None

    # ---- spawn ----------------------------------------------------------------
    async def spawn(self) -> None:
        """Start the runner in the plugin's venv; returns once imports are done (SHELL)."""
        loop = asyncio.get_running_loop()
        self.state = SPAWNING
        self._shell_ready = loop.create_future()
        logp = self.cfg.path("logs", f"{self.name}.log")
        _rotate(logp, self.cfg.log_max_bytes, self.cfg.log_backups)
        self._logf = open(logp, "ab", buffering=0)
        self._logf.write(f"\n==== spawn {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n".encode())
        env = {k: v for k, v in os.environ.items()
               if k.upper() not in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "PYTHONPATH", "PYTHONHOME",
                                    "VIRTUAL_ENV")}
        env.update({"PYTHONPATH": str(HARNESS_ROOT), "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1", "NO_PROXY": "*", "PYTHONIOENCODING": "utf-8",
                    "HARNESS_CACHE_DIR": str(self.cache_dir)})
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # persistent kernel/compile caches under the model's cache dir (spec 9.1)
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(self.cache_dir / "inductor"))
        env.setdefault("TRITON_CACHE_DIR", str(self.cache_dir / "triton"))
        args = [self.python, "-u", str(RUNNER), "--model", self.name, "--plugin-dir", str(self.model.folder),
                "--entry", self.model.entry, "--cache-dir", str(self.cache_dir),
                "--sampler-ms", str(self.cfg.sampler_ms)]
        self.proc = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=self._logf,
            cwd=str(self.model.folder), env=env, limit=STREAM_LIMIT, **osx.runner_spawn_kwargs())
        self.pid = self.proc.pid
        try:
            self.group.add(self.pid)
        except OSError as e:
            log.warning("could not add %s (pid %s) to job object: %s", self.name, self.pid, e)
        self.cfg.path("run", f"{self.name}.pid").write_text(str(self.pid))
        self._reader = asyncio.create_task(self._read_loop())
        t = time.perf_counter()
        try:
            ready = await asyncio.wait_for(asyncio.shield(self._shell_ready), timeout=300)
        except asyncio.TimeoutError:
            await self.kill("spawn_timeout")
            raise HarnessError("worker_crashed", f"{self.name}: runner did not become ready in 300 s")
        if ready.get("type") != "shell_ready":
            await self.kill("spawn_failed")
            raise HarnessError(ready.get("code", "worker_crashed"),
                               f"{ready.get('message', 'runner failed to start')} (see logs/{self.name}.log)")
        self.cold_start["spawn_s"] = round(time.perf_counter() - t, 3)
        self.state = SHELL

    @property
    def cache_dir(self) -> Path:
        return self.cfg.path("cache", self.name)

    # ---- paging -----------------------------------------------------------------
    def page(self) -> None:
        """Warm shell -> disk: trim working set and suspend (Windows)."""
        try:
            osx.trim_working_set(self.pid)
            osx.suspend(self.pid)
            osx.trim_working_set(self.pid)
            self.state = PAGED
        except OSError as e:
            log.warning("paging %s failed: %s", self.name, e)
            self.state = SHELL

    def resume(self) -> float:
        t = time.perf_counter()
        was = self.state
        self.state = RESUMING
        if was == PAGED:
            osx.resume(self.pid)
        return round(time.perf_counter() - t, 3)

    # ---- load / run ---------------------------------------------------------------
    async def load(self, msg: dict) -> dict:
        loop = asyncio.get_running_loop()
        self.state = LOADING
        self._loaded = loop.create_future()
        self.send({"type": "load", **msg})
        res = await self._loaded
        if res.get("type") == "error":
            raise HarnessError(res.get("code", "load_failed"), res.get("message", "load failed"),
                               metrics=res.get("metrics"))
        self.placement = res.get("placement", {})
        self.cold_start.update(res.get("cold_start", {}))
        self.ctx_overhead_gb = res.get("ctx_overhead_gb", 0.0)
        self.state = HOT
        return res

    def run(self, ids: list[str], functions: list[str], inputs: list[dict], out_dirs: list[str],
            limit_s: float | None = None) -> list[asyncio.Future]:
        """One run message (k ids). Each request writes into its own out_dir (single-id) or a batch dir."""
        loop = asyncio.get_running_loop()
        futs = []
        for rid in ids:
            f = loop.create_future()
            self.running[rid] = f
            futs.append(f)
        self.state = BUSY
        self.last_used = time.monotonic()
        self.send({"type": "run", "ids": ids, "functions": functions, "inputs": inputs, "out_dir": out_dirs[0],
                   "out_dirs": out_dirs, "limit_s": limit_s})
        return futs

    def send(self, msg: dict) -> None:
        if self.proc is None or self.proc.stdin is None or self.proc.returncode is not None:
            raise HarnessError("worker_crashed", f"{self.name}: runner is not running")
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())

    @property
    def idle(self) -> bool:
        return self.state == HOT and not self.running

    # ---- reader ---------------------------------------------------------------------
    async def _read_loop(self):
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("%s: non-JSON on protocol channel: %r", self.name, line[:200])
                    continue
                self._dispatch(m)
        except Exception as e:  # pragma: no cover
            log.exception("%s reader failed: %s", self.name, e)
        await self.proc.wait()
        self._on_exit()

    def _dispatch(self, m: dict):
        t = m.get("type")
        if t == "shell_ready":
            if self._shell_ready and not self._shell_ready.done():
                self._shell_ready.set_result(m)
        elif t == "loaded":
            if self._loaded and not self._loaded.done():
                self._loaded.set_result(m)
        elif t in ("result", "error"):
            rid = m.get("id")
            if rid is None:
                for fut in (self._loaded, self._shell_ready):
                    if fut and not fut.done():
                        fut.set_result(m)
                        return
                log.warning("%s: error without id: %s", self.name, m)
                return
            fut = self.running.pop(rid, None)
            if fut and not fut.done():
                fut.set_result(m)
            if not self.running and self.state == BUSY:
                self.state = HOT
                self.last_used = time.monotonic()
        elif t == "stage":
            if self.on_stage:
                self.on_stage(self, m)
        elif t == "oom_recovered":
            if self.on_oom_recovered:
                self.on_oom_recovered(self, m)

    def _on_exit(self):
        code = self.proc.returncode if self.proc else None
        reason = self.kill_reason or "worker_crashed"
        msg = {"type": "error", "code": reason, "message": f"{self.name}: runner exited (code {code}, {reason})"}
        for fut in (self._shell_ready, self._loaded):
            if fut and not fut.done():
                fut.set_result(msg)
        for rid, fut in list(self.running.items()):
            if not fut.done():
                fut.set_result({**msg, "id": rid})
        self.running.clear()
        self.state = KILLED
        for h in (self.idle_handle, self.expiry_handle):
            if h:
                h.cancel()
        try:
            self.cfg.path("run", f"{self.name}.pid").unlink(missing_ok=True)
        except OSError:
            pass
        if self._logf:
            self._logf.close()
        self.exited.set()
        if self.on_exit:
            self.on_exit(self)

    # ---- kill -------------------------------------------------------------------
    async def kill(self, reason: str = "killed") -> None:
        if self.proc is None or self.proc.returncode is not None:
            return
        self.kill_reason = self.kill_reason or reason
        try:
            if self.state == PAGED:
                osx.resume(self.pid)  # a suspended process may not process termination promptly
        except OSError:
            pass
        try:
            self.proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(self.exited.wait(), 15)
        except asyncio.TimeoutError:
            log.error("%s (pid %s) did not exit after kill", self.name, self.pid)
