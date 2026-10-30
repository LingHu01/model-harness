"""MCP round trip: client -> shim (stdio) -> core. Uses HARNESS_CONFIG (default tests/selftest.yaml).

    .venv\\Scripts\\python.exe tests\\shim_test.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent


def payload(res):
    if getattr(res, "structured_content", None) or getattr(res, "structuredContent", None):
        sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    return json.loads(res.content[0].text)


async def main():
    env = {**os.environ, "HARNESS_CONFIG": os.environ.get("HARNESS_CONFIG", str(ROOT / "tests" / "selftest.yaml"))}
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "shim" / "shim.py")], env=env)
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        tools = {t.name: t for t in (await s.list_tools()).tools}
        print("tools:", sorted(tools))
        print("catalog in run description:", "echo-test" in tools["run"].description)
        print("is_running:", payload(await s.call_tool("is_running", {})))
        print("start:", payload(await s.call_tool("start", {})))
        print("start again:", payload(await s.call_tool("start", {})))
        progress = []

        async def on_progress(p, total, msg):
            progress.append(msg)

        res = payload(await s.call_tool("run", {"model": "echo-test", "function": "echo",
                                                "inputs": {"text": "via mcp"}}, progress_callback=on_progress))
        print("run:", res.get("result"), "progress:", progress)
        st = payload(await s.call_tool("status", {}))
        print("status echo-test:", st["models"]["echo-test"]["state"], "gpu:", st.get("gpu"))
        print("schema keys:", sorted(payload(await s.call_tool("get_schema", {"model": "echo-test",
                                                                             "function": "echo"}))))
        rl = payload(await s.call_tool("reload", {}))
        tools = {t.name: t for t in (await s.list_tools()).tools}
        print("reload:", {k: rl.get(k) for k in ("added", "removed", "changed")},
              "run description refreshed:", rl.get("catalog", "?") in tools["run"].description)
        bad = payload(await s.call_tool("add_model", {"folder": "E:/nope"}))
        print("add_model bad folder:", bad.get("error", {}).get("code"))
        print("stop:", payload(await s.call_tool("stop", {})))
        print("after stop run:", payload(await s.call_tool("run", {"model": "echo-test", "function": "echo",
                                                                   "inputs": {"text": "x"}})))


if __name__ == "__main__":
    asyncio.run(main())
