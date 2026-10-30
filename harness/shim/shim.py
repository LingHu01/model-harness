"""MCP stdio shim. Launched by the agent (QwenPaw); tiny; forwards to the core over localhost HTTP.

start / stop / is_running are handled here; everything else goes to the core.
The catalog (state/catalog.json, written by the core) is placed in the `run` tool description at
session start, so it stays static for the session (prompt-cache friendly).
"""
from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HARNESS_ROOT = Path(__file__).resolve().parent.parent
SHARED = "--http" in sys.argv     # one server for several agents
sys.path.insert(0, str(HARNESS_ROOT))

from mcp.server.mcpserver import Context, MCPServer  # noqa: E402

from core import config as config_mod  # noqa: E402  (stdlib + yaml only)
from core.osal import osx  # noqa: E402

cfg = config_mod.load(Path(os.environ["HARNESS_CONFIG"]) if os.environ.get("HARNESS_CONFIG") else None)
log = logging.getLogger("shim")
_h = logging.handlers.RotatingFileHandler(cfg.path("logs", "shim.log"), maxBytes=cfg.log_max_bytes,
                                          backupCount=cfg.log_backups, encoding="utf-8")
_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_h])

NOT_RUNNING = {"error": {"code": "harness_not_running", "message": "the harness core is not running; call start"}}


# ---- tiny async HTTP client (asyncio streams, so cancelling a call closes the socket) ---------------
async def _http(method: str, path: str, body: dict | None = None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else b""
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", cfg.port), 3)
    except (OSError, asyncio.TimeoutError):
        return None
    try:
        writer.write(f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                     f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data)
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), timeout)
        _, _, payload = raw.partition(b"\r\n\r\n")
        return json.loads(payload) if payload else None
    finally:
        writer.close()


async def _healthy() -> bool:
    try:
        r = await _http("GET", "/health", timeout=3)
        return bool(r and r.get("ok"))
    except Exception:
        return False


def _lock_pid() -> int | None:
    try:
        pid = int(json.loads(cfg.path("run", "core.lock").read_text())["pid"])
        return pid if osx.pid_alive(pid) else None
    except Exception:
        return None


def _catalog_text() -> str:
    try:
        return json.loads(cfg.path("state", "catalog.json").read_text(encoding="utf-8"))["text"]
    except Exception:
        return "(catalog not built yet: call start, then status for the model list)"


async def _forward(method, path, body=None, timeout=60):
    if not await _healthy():
        return NOT_RUNNING
    r = await _http(method, path, body, timeout)
    return r if r is not None else NOT_RUNNING


# ---- tools --------------------------------------------------------------------------------------
async def start() -> dict:
    """Troubleshooting only: the harness starts by itself. Start the core if a request reports it is not running."""
    if await _healthy():
        return {"ok": True, "already_running": True, "pid": _lock_pid()}
    py = osx.venv_python(str(HARNESS_ROOT / ".venv"))  # console launcher + CREATE_NO_WINDOW: see os_win
    if not os.path.exists(py):
        py = sys.executable
    args = [py, "-m", "core.server"]
    log.info("launching core: %s", args)
    try:
        subprocess.Popen(args, cwd=str(HARNESS_ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, **osx.detached_spawn_kwargs())
    except OSError:  # agent runs inside a job that forbids breakaway
        subprocess.Popen(args, cwd=str(HARNESS_ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, **osx.detached_spawn_kwargs_fallback())
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if await _healthy():
            return {"ok": True, "already_running": False, "pid": _lock_pid()}
        await asyncio.sleep(0.25)
    return {"error": {"code": "harness_not_running", "message": f"core did not answer /health in 60 s; "
                                                                f"see {cfg.path('logs', 'core.log')}"}}


async def stop(force: bool = False) -> dict:
    """Troubleshooting only: shut the harness down. Every agent shares it, so this is refused while requests are
    running or queued, unless force=true. To free VRAM use unload."""
    if not await _healthy():
        return {"ok": True, "was_running": False}
    if SHARED and not force:
        models = (await _forward("GET", "/status")).get("models", {})
        busy = {m: s["running"] + s["queued"] for m, s in models.items() if s["running"] or s["queued"]}
        if busy:
            return {"error": {"code": "busy", "message": f"requests in progress: {busy}. Other agents share this "
                                                         "harness; use cancel, or force=true in an emergency"}}
    await _http("POST", "/shutdown")
    deadline = time.monotonic() + cfg.shutdown_grace + 20
    while time.monotonic() < deadline and await _healthy():
        await asyncio.sleep(0.25)
    return {"ok": True, "was_running": True}


async def is_running() -> dict:
    """Troubleshooting only, if a request fails: whether the harness core is up. No need to check before a call."""
    return {"running": await _healthy(), "pid": _lock_pid()}


async def run(model: str, function: str, inputs: dict[str, Any], ctx: Context) -> dict:
    """(description replaced at start-up with the catalog)"""
    if SHARED and not await _healthy():
        await start()
    if not await _healthy():
        return NOT_RUNNING
    body = json.dumps({"model": model, "function": function, "inputs": inputs}).encode()
    reader, writer = await asyncio.open_connection("127.0.0.1", cfg.port, limit=64 * 1024 * 1024)
    try:
        writer.write(b"POST /run HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                     + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
        await writer.drain()
        status = await reader.readline()
        while (await reader.readline()) not in (b"\r\n", b""):
            pass
        if b" 200 " not in status:
            payload = await reader.read()
            return json.loads(payload) if payload else {"error": {"code": "internal", "message": status.decode()}}
        step, rid = 0, None
        while True:
            line = await reader.readline()
            if not line:
                return {"error": {"code": "worker_crashed", "message": "core closed the stream", "request_id": rid}}
            ev = json.loads(line)
            kind = ev.pop("event")
            if kind == "result":
                return {"request_id": rid, "result": ev["result"], "metrics": ev["metrics"]}
            if kind == "error":
                return {"error": ev}
            rid = ev.get("request_id", rid)
            step += 1
            msg = {"queued": f"queued: {ev.get('position', 0)} ahead", "loading": "loading",
                   "running": "running"}.get(kind, kind)
            try:
                await ctx.report_progress(step, None, msg)
            except Exception:
                pass
    finally:
        writer.close()  # on cancellation this hangs up and the core cancels the request


async def get_schema(model: str, function: str) -> dict:
    """Full input/output schema, extras and examples for one model function."""
    return await _forward("GET", f"/schema/{model}/{function}")


async def status() -> dict:
    """Troubleshooting only, if a request fails: per-model state, queue, memory need, free VRAM/RAM, invalid models.
    No need to check before a call."""
    return await _forward("GET", "/status")


async def unload(model: str, force: bool = False) -> dict:
    """Free a model's VRAM, for example for another program. Models unload by themselves when idle or when another
    needs room. Busy -> error unless force=true."""
    return await _forward("POST", f"/unload/{model}?force={'true' if force else 'false'}")


async def cancel(request_id: str) -> dict:
    """Cancel a stuck queued or running request (running: its worker is killed)."""
    return await _forward("POST", f"/cancel/{request_id}")


RUN_DOC = """Run one model function in a single step. Inputs: file paths (preferred) or base64 \
({"base64": "...", "ext": ".png"} or a data: URL). Output files are returned as paths. \
Returns {request_id, result, metrics} or {error: {code, message, …}}. \
Optional params beyond the contract (extras) are listed per function; get_schema gives details.

"""


async def _publish_catalog(res: dict, ctx: Context) -> dict:
    """Put the new catalog in the `run` description and tell the client the tool list changed."""
    if "catalog" in res and _srv is not None:
        _srv.remove_tool("run")
        _srv.add_tool(run, description=RUN_DOC + res["catalog"])
        try:
            await ctx.notify_tools_changed()
        except Exception as e:  # the catalog is in this result anyway
            log.warning("tools changed notification failed: %s", e)
    return res


async def reload(ctx: Context) -> dict:
    """Maintenance: re-read the registry, manifests, schemas and tuning without restarting the core. Models that did not
    change stay loaded; changed or removed ones retire once their running requests finish. Returns
    {added, removed, changed, invalid, catalog}."""
    return await _publish_catalog(await _forward("POST", "/reload"), ctx)


async def add_model(folder: str, ctx: Context, tags: list[str] | None = None) -> dict:
    """Maintenance: register a plugin folder (manifest.yaml, env.lock, adapter, schemas, tests/smoke.py), available
    at once, no restart. Each variant becomes a model. cap: tags come from the manifest; pass the others in
    tags (lang:en, speed:fast, quality:high, license:...). An invalid plugin is rejected and not registered."""
    return await _publish_catalog(await _forward("POST", "/add_model", {"folder": folder, "tags": tags}), ctx)


def _catalog() -> dict:
    try:
        return json.loads(cfg.path("state", "catalog.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _named_tool(capability: str, contract: dict, models: dict):
    """A tool named for what it does, for one capability. A model picks tools by name: `run` hides all this."""
    tool = contract["tool"]

    async def call(image: str, ctx: Context, query: str = "", model: str = "") -> dict:
        name = model or next(iter(models))
        if name not in models:
            return {"error": {"code": "unknown_model", "message": f"{name}: choose one of {', '.join(models)}"}}
        inputs = {"image": image}
        if "query" in contract["inputs"]:
            inputs["query"] = query or tool.get("default_query", "")
        done = await run(name, models[name], inputs, ctx)
        if "result" not in done:
            return done
        out = {k: v for k, v in done["result"].items() if k != "response"}  # raw model text helps nobody
        if "cutout" in out:
            out.pop("instances", None)
        return {**out, "model": name, "seconds": done.get("metrics", {}).get("duration_s")}

    async def save(image: str, ctx: Context, query: str = "", model: str = "", output_path: str = "") -> dict:
        out = await call(image, ctx, query, model)
        if output_path and "cutout" in out:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out["cutout"] = shutil.copy2(out["cutout"], output_path)
        return out

    described = ", ".join(f"{m}" for m in models)
    return (save if tool.get("saves_file") else call, tool["name"],
            f"{tool['description']} Image: a file path. Models: {described} (default the first). {tool['returns']}")


def _named_tools() -> list:
    cat = _catalog()
    out = []
    for capability, contract in (cat.get("contracts") or {}).items():
        models = {m: next((f for f, d in spec["functions"].items() if d["capability"] == capability), None)
                  for m, spec in (cat.get("models") or {}).items()}
        models = {m: f for m, f in models.items() if f}
        if contract.get("tool") and models:
            out.append(_named_tool(capability, contract, models))
    return out


_srv: MCPServer | None = None


def build_server() -> MCPServer:
    global _srv
    srv = MCPServer("model-harness", instructions=(
        "Local models that are not LLMs. Call the tool named for the job, such as remove_background. It starts and "
        "loads what it needs, so no checks are needed first; a model's first call is slower. For anything else use "
        "run(model, function, inputs)."))
    srv.add_tool(start, description=start.__doc__)
    srv.add_tool(stop, description=stop.__doc__)
    srv.add_tool(is_running, description=is_running.__doc__)
    srv.add_tool(run, description=RUN_DOC + _catalog_text())
    srv.add_tool(get_schema, description=get_schema.__doc__)
    srv.add_tool(status, description=status.__doc__)
    srv.add_tool(unload, description=unload.__doc__)
    srv.add_tool(cancel, description=cancel.__doc__)
    srv.add_tool(reload, description=reload.__doc__)
    srv.add_tool(add_model, description=add_model.__doc__)
    for fn, name, description in _named_tools():
        srv.add_tool(fn, name=name, description=description)
    _srv = srv
    return srv


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    if SHARED:
        log.info("shared MCP server on 127.0.0.1:%s", cfg.mcp_port)
        log.info("core: %s", asyncio.run(start()))
        build_server().run("streamable-http", host="127.0.0.1", port=cfg.mcp_port)
        return
    log.info("shim session start")
    build_server().run("stdio")


if __name__ == "__main__":
    main()
