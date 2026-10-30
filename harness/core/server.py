"""Core: localhost HTTP API (stdlib asyncio, no web framework). Run windowless via pythonw.exe.

    python -m core.server            (harness root as cwd, or harness .venv on PATH)
"""
from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import shutil
import sys
import time
from urllib.parse import parse_qs, unquote, urlsplit

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from core import config as config_mod  # noqa: E402
from core.artifacts import Artifacts  # noqa: E402
from core.catalog import Catalog  # noqa: E402
from core.errors import HarnessError  # noqa: E402
from core.io import IO  # noqa: E402
from core.memory import Memory  # noqa: E402
from core.metrics import Store  # noqa: E402
from core.osal import osx  # noqa: E402
from core.scheduler import Scheduler  # noqa: E402

log = logging.getLogger("core")
MAX_BODY = 256 * 1024 * 1024
REASONS = {200: "OK", 400: "Bad Request", 404: "Not Found", 409: "Conflict", 413: "Too Large",
           500: "Internal Server Error", 503: "Service Unavailable"}
STATUS_FOR = {"unknown_model": 404, "unknown_function": 404, "invalid_input": 400, "busy": 409,
              "shutting_down": 503}


class Core:
    def __init__(self, cfg):
        self.cfg = cfg
        self.started = time.time()
        self.store = Store(cfg.path("state", "harness.db"))
        self.catalog = Catalog(cfg).load()
        self.memory = Memory(cfg, self.store)
        self.io = IO(cfg)
        self.artifacts = Artifacts(cfg)
        self.sched: Scheduler | None = None
        self.server: asyncio.base_events.Server | None = None
        self.stop_event = asyncio.Event()
        self.warnings: list[str] = []
        self.bench_holder: str | None = None

    # ---- lifecycle --------------------------------------------------------------------
    async def start(self):
        self.sched = Scheduler(self.cfg, self.catalog, self.store, self.memory, self.io, self.artifacts)
        self._startup_cleanup()
        self._pagefile_check()
        self._write_catalog()
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", self.cfg.port, limit=1 << 20)
        self._schedule_retention()
        log.info("core up on 127.0.0.1:%s (pid %s); %d models, %d invalid",
                 self.cfg.port, os.getpid(), len(self.catalog.models), len(self.catalog.invalid))

    async def stop(self):
        log.info("shutdown requested")
        await self.sched.shutdown()
        self.server.close()
        self.stop_event.set()

    def _startup_cleanup(self):
        for p in self.cfg.path("run").glob("*.pid"):
            p.unlink(missing_ok=True)
        for p in self.cfg.path("in").iterdir():  # no request survives a restart
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        for d in self.cfg.path("cache").glob("*/*"):  # half-built artifacts: no key.json
            if d.is_dir() and d.name not in ("inductor", "triton") and not (d / "key.json").exists():
                shutil.rmtree(d, ignore_errors=True)

    def _pagefile_check(self):
        info = osx.pagefile_info()
        files = info.get("paging_files", [])
        if not files:
            self.warnings.append("no paging file configured: paged warm shells need one")
        else:
            data_drive = str(self.cfg.data_root)[:2].upper()
            if sys.platform == "win32" and not any(f.strip().upper().startswith(data_drive) or
                                                   f.strip().startswith("?:") for f in files):
                self.warnings.append(f"pagefile not on {data_drive} (configured: {'; '.join(files)})")
        for w in self.warnings:
            log.warning(w)

    def _check_bench(self):
        if self.bench_holder:  # a reload would drop the bench's temporary concurrency override
            raise HarnessError("busy", f"reload refused during bench ({self.bench_holder})")

    def _reload(self) -> dict:
        self._check_bench()
        old = self.catalog.models
        try:
            self.catalog.load()
        except Exception as e:
            raise HarnessError("invalid_input", f"catalog not reloaded, previous one still active: {e}")
        diff = self.sched.sync_catalog(old)
        self._write_catalog()
        log.info("catalog reloaded: %s", diff)
        return {**diff, "models": list(self.catalog.models), "invalid": self.catalog.invalid,
                "catalog": self.catalog.text}

    def _write_catalog(self):
        """The shim reads this at each MCP session start (it may start before the core)."""
        self.cfg.path("state", "catalog.json").write_text(
            json.dumps({"text": self.catalog.text, **self.catalog.structured()}, indent=1), encoding="utf-8")

    def _schedule_retention(self):
        n = self.io.cleanup()
        if n:
            log.info("retention: removed %d old in/out entries", n)
        asyncio.get_running_loop().call_later(3600, self._schedule_retention)

    # ---- HTTP -------------------------------------------------------------------------
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = {k.strip().lower(): v.strip() for k, v in (l.split(":", 1) for l in lines[1:] if ":" in l)}
            n = int(headers.get("content-length", "0") or 0)
            if n > MAX_BODY:
                return await self._json(writer, 413, {"error": {"code": "invalid_input", "message": "body too large"}})
            body = await reader.readexactly(n) if n else b""
            url = urlsplit(target)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
            await self._route(method, parts, q, body, reader, writer)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except HarnessError as e:
            await self._json(writer, STATUS_FOR.get(e.code, 500), e.to_dict())
        except Exception as e:
            log.exception("request failed")
            await self._json(writer, 500, {"error": {"code": "internal", "message": str(e)}})
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _route(self, method, parts, q, body, reader, writer):
        route = (method, parts[0] if parts else "")
        data = json.loads(body) if body else {}
        if route == ("GET", "health"):
            return await self._json(writer, 200, {"ok": True, "pid": os.getpid(), "uptime_s": int(time.time() - self.started),
                                                  "version": config_mod.VERSION})
        if route == ("GET", "catalog"):
            return await self._json(writer, 200, {"text": self.catalog.text, **self.catalog.structured()})
        if route == ("GET", "schema") and len(parts) == 3:
            m = self.catalog.models.get(parts[1])
            if m is None:
                raise HarnessError("unknown_model", f"unknown model {parts[1]!r}")
            fn = m.functions.get(parts[2])
            if fn is None:
                raise HarnessError("unknown_function", f"{parts[1]} has no function {parts[2]!r}")
            contract = self.catalog.contracts[fn.capability]
            return await self._json(writer, 200, {"model": m.name, "function": fn.name, "capability": fn.capability,
                                                  "input": fn.input_schema, "output": contract.get("output"),
                                                  **fn.schema})
        if route == ("GET", "status"):
            st = self.sched.status()
            st.update({"pid": os.getpid(), "uptime_s": int(time.time() - self.started), "warnings": self.warnings,
                       "bench_lock": self.bench_holder, "data_root": str(self.cfg.data_root)})
            return await self._json(writer, 200, st)
        if route == ("POST", "run"):
            return await self._run(data, q, reader, writer)
        if route == ("POST", "unload") and len(parts) == 2:
            res = await self.sched.unload(parts[1], force=q.get("force", "").lower() in ("1", "true", "yes"))
            return await self._json(writer, 200, res)
        if route == ("POST", "cancel") and len(parts) == 2:
            return await self._json(writer, 200, await self.sched.cancel(parts[1]))
        if route == ("POST", "shutdown"):
            await self._json(writer, 200, {"ok": True, "stopping": True})
            asyncio.get_running_loop().create_task(self.stop())
            return
        if route == ("POST", "reload"):
            return await self._json(writer, 200, self._reload())
        if route == ("POST", "add_model"):
            self._check_bench()
            if not data.get("folder"):
                raise HarnessError("invalid_input", "folder is required")
            try:
                names = self.catalog.add_plugin(data["folder"], data.get("tags"))
            except (ValueError, OSError, yaml.YAMLError) as e:
                raise HarnessError("invalid_input", f"plugin not added: {e}")
            return await self._json(writer, 200, {"registered": names, **self._reload()})
        if route == ("POST", "bench") and len(parts) == 2:
            holder = data.get("holder", "bench")
            if parts[1] == "lock":
                if self.bench_holder and self.bench_holder != holder:
                    raise HarnessError("busy", f"bench lock held by {self.bench_holder}")
                self.bench_holder = holder
                if data.get("model"):
                    self.sched.bench_override(data["model"], int(data.get("max", 1)))
            elif parts[1] == "unlock":
                self.bench_holder = None
                self.sched.bench_restore()
            return await self._json(writer, 200, {"bench_lock": self.bench_holder})
        return await self._json(writer, 404, {"error": {"code": "not_found", "message": f"{method} /{'/'.join(parts)}"}})

    async def _run(self, data, q, reader, writer):
        req = self.sched.submit(data.get("model", ""), data.get("function", ""), data.get("inputs") or {},
                                data.get("request_id"), data.get("limit_s"))
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nCache-Control: no-cache\r\n"
                     b"Connection: close\r\n\r\n")
        await writer.drain()
        # client hang-up while queued/running cancels the request
        gone = asyncio.create_task(reader.read(1))
        try:
            while True:
                get = asyncio.create_task(req.events.get())
                done, _ = await asyncio.wait({get, gone}, return_when=asyncio.FIRST_COMPLETED)
                if get not in done:
                    get.cancel()
                    if req.state in ("queued", "running"):
                        log.info("client went away; cancelling %s", req.id)
                        await self.sched.cancel(req.id)
                    return
                ev = get.result()
                writer.write((json.dumps(ev, default=str) + "\n").encode())
                await writer.drain()
                if ev["event"] in ("result", "error"):
                    return
        finally:
            gone.cancel()

    async def _json(self, writer, status, obj):
        body = json.dumps(obj, default=str).encode()
        writer.write(f"HTTP/1.1 {status} {REASONS.get(status, 'OK')}\r\nContent-Type: application/json\r\n"
                     f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
        await writer.drain()


# ---- single instance ------------------------------------------------------------------------
def acquire_lock(cfg) -> bool:
    lock = cfg.path("run", "core.lock")
    if lock.exists():
        try:
            info = json.loads(lock.read_text())
            if osx.pid_alive(int(info["pid"])) and int(info["pid"]) != os.getpid():
                return False
        except Exception:
            pass
        lock.unlink(missing_ok=True)  # stale lock (dead PID)
    lock.write_text(json.dumps({"pid": os.getpid(), "port": cfg.port, "started": time.time()}))
    return True


def release_lock(cfg):
    lock = cfg.path("run", "core.lock")
    try:
        if json.loads(lock.read_text()).get("pid") == os.getpid():
            lock.unlink()
    except Exception:
        pass


def setup_logging(cfg):
    h = logging.handlers.RotatingFileHandler(cfg.path("logs", "core.log"), maxBytes=cfg.log_max_bytes,
                                             backupCount=cfg.log_backups, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(logging.INFO)


async def amain(cfg):
    core = Core(cfg)
    await core.start()
    await core.stop_event.wait()
    core.store.close()
    log.info("core stopped")


def main():
    cfg = config_mod.load(config_mod.REPO_ROOT / "config.yaml"
                          if not os.environ.get("HARNESS_CONFIG") else config_mod.Path(os.environ["HARNESS_CONFIG"]))
    setup_logging(cfg)
    if sys.stdout is None or sys.stderr is None:  # pythonw: no console streams at all
        sys.stdout = sys.stderr = open(os.devnull, "w")
    if not acquire_lock(cfg):
        log.info("another core is running; exiting")
        return 0
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        asyncio.run(amain(cfg))
    except Exception:
        log.exception("core crashed")
        raise
    finally:
        release_lock(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
