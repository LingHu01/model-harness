"""End-to-end self-test against a running core (stdlib only).

    set HARNESS_CONFIG=tests\\selftest.yaml
    .venv\\Scripts\\python.exe -m core.server          (in another terminal / background)
    .venv\\Scripts\\python.exe tests\\selftest.py [--gpu]
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

PORT = 8799
GPU = "--gpu" in sys.argv
results: list[tuple[str, bool, str]] = []


def http(method, path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def run(model, function, inputs, rid=None, events=None):
    body = {"model": model, "function": function, "inputs": inputs}
    if rid:
        body["request_id"] = rid
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/run", json.dumps(body).encode(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            evs = [json.loads(l) for l in r]
    except urllib.error.HTTPError as e:
        return {"event": "error", **json.loads(e.read())["error"]}
    if events is not None:
        events.extend(evs)
    return evs[-1]


def bg(fn, *a, **kw):
    box = {}
    t = threading.Thread(target=lambda: box.update(r=fn(*a, **kw)), daemon=True)
    t.start()
    return t, box


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def state(model):
    return http("GET", "/status")["models"][model]["state"]


def main():
    h = http("GET", "/health")
    check("health", h.get("ok"), f"pid {h.get('pid')}")
    cat = http("GET", "/catalog")
    check("catalog lists models", {"echo-test", "torch-a", "torch-b"} <= set(cat["models"]),
          f"invalid={cat['invalid']}")

    # phase 1/2: run, reuse, metrics
    r1 = run("echo-test", "echo", {"text": "hello"})
    check("echo result", r1.get("event") == "result" and r1["result"]["text"] == "hello", str(r1)[:200])
    check("metrics returned", "duration_s" in r1.get("metrics", {}) and "cold_start" in r1.get("metrics", {}),
          json.dumps(r1.get("metrics", {}).get("cold_start")))
    r2 = run("echo-test", "echo", {"text": "again"})
    check("worker reused", r2["result"]["pid"] == r1["result"]["pid"], f"{r1['result']['pid']} / {r2['result']['pid']}")

    check("invalid input rejected", run("echo-test", "echo", {"txt": 1}).get("code") == "invalid_input")
    check("unknown model", run("nope", "x", {}).get("code") == "unknown_model")
    check("unknown function", run("echo-test", "nope", {}).get("code") == "unknown_function")
    b64 = run("echo-test", "echo", {"text": "b64"})  # base64 path covered by the vision smoke test
    check("sanity", b64.get("event") == "result")

    # FIFO queue + progress events
    evs: list = []
    t1, b1 = bg(run, "echo-test", "echo", {"text": "a", "sleep_s": 3})
    time.sleep(0.3)
    t2, b2 = bg(run, "echo-test", "echo", {"text": "b", "sleep_s": 0.1}, events=evs)
    t1.join(); t2.join()
    check("queued request served after first", b1["r"]["event"] == "result" and b2["r"]["event"] == "result")
    check("queue position reported", any(e["event"] == "queued" and e.get("position", 0) >= 1 for e in evs),
          str([e["event"] for e in evs]))

    # cancel queued + running
    t1, b1 = bg(run, "echo-test", "echo", {"text": "long", "sleep_s": 5}, rid="run-long")
    time.sleep(0.5)
    t2, b2 = bg(run, "echo-test", "echo", {"text": "q"}, rid="queued-1")
    time.sleep(0.5)
    c = http("POST", "/cancel/queued-1")
    t2.join()
    check("cancel queued", c.get("was") == "queued" and b2["r"].get("code") == "cancelled", str(c))
    c = http("POST", "/cancel/run-long")
    t1.join()
    check("cancel running", c.get("was") == "running" and b1["r"].get("code") == "cancelled", str(c))
    r = run("echo-test", "echo", {"text": "after cancel"})
    check("works after cancel (new worker)", r.get("event") == "result")

    # R4 stall: position 0 behind a long request does not move for queue_stall (7 s)
    t1, b1 = bg(run, "echo-test", "echo", {"text": "slow", "sleep_s": 9.5})
    time.sleep(0.5)
    r = run("echo-test", "echo", {"text": "stuck"})
    t1.join()
    check("queue_stalled", r.get("code") == "queue_stalled" and r.get("blocked_by"), json.dumps(r)[:200])

    # R5 runaway (10 s)
    t0 = time.time()
    r = run("echo-test", "echo", {"text": "forever", "sleep_s": 60})
    check("runaway", r.get("code") == "runaway" and time.time() - t0 < 20, f"{time.time() - t0:.1f}s")

    # crash
    r = run("echo-test", "echo", {"text": "boom", "crash": True})
    check("worker_crashed", r.get("code") == "worker_crashed", str(r)[:200])
    r = run("echo-test", "echo", {"text": "alive"})
    check("recovers after crash", r.get("event") == "result")

    # R1 idle -> paged warm shell, then resume
    time.sleep(12)
    st = http("GET", "/status")["models"]["echo-test"]
    check("idle timeout -> paged shell", st["state"] in ("PAGED", "SHELL"), st["state"])
    r = run("echo-test", "echo", {"text": "from shell"})
    cs = r.get("metrics", {}).get("cold_start", {})
    check("resumed from warm shell", cs.get("from_shell") is True, json.dumps(cs))

    # unload
    u = http("POST", "/unload/echo-test")
    check("unload", u.get("was") == "loaded", str(u))

    test_reload()
    if GPU:
        gpu_tests()

    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed" + (f"; failed: {bad}" if bad else ""))
    return 1 if bad else 0


def gpu_tests():
    total = http("GET", "/status")["gpu"]["cuda:0"]["total_gb"]
    big = round(total * 0.55, 1)
    # measure torch-b first so its need is known
    r = run("torch-b", "alloc", {"gb": big})
    m = r.get("metrics", {})
    check("torch-b alloc + vram measured", r.get("event") == "result" and (m.get("vram_peak_gb") or 0) >= big,
          f"peak={m.get('vram_peak_gb')} ext={m.get('vram_peak_ext_gb')} ram={m.get('ram_peak_gb')}")
    need_b = http("GET", "/status")["models"]["torch-b"]["need"]
    check("need measured", need_b["measured"] and need_b["vram_gb"] >= big, json.dumps(need_b))
    http("POST", "/unload/torch-b")
    r = run("torch-a", "alloc", {"gb": big})
    check("torch-a alloc", r.get("event") == "result", str(r)[:200])
    # torch-a is idle but holds `big` GB in PyTorch's cache: torch-b must evict it (R3)
    r = run("torch-b", "alloc", {"gb": big})
    st = http("GET", "/status")["models"]
    check("pressure eviction (R3)", r.get("event") == "result" and st["torch-a"]["state"] in ("PAGED", "SHELL", "SPAWNING", "GONE"),
          f"torch-a={st['torch-a']['state']} torch-b={st['torch-b']['state']}")
    # busy model is never evicted: torch-a waits for torch-b's running request
    t1, b1 = bg(run, "torch-b", "alloc", {"gb": big, "seconds": 4})
    time.sleep(1.0)
    evs: list = []
    r = run("torch-a", "alloc", {"gb": big}, events=evs)
    t1.join()
    check("busy not evicted; waiter runs after", b1["r"].get("event") == "result" and r.get("event") == "result",
          f"{b1['r'].get('event')} / {r.get('event', r)}")
    http("POST", "/unload/torch-a")
    http("POST", "/unload/torch-b")


def test_reload():
    """Plugins come and go without restarting the core. Works on a throwaway copy of echo-test."""
    root = Path(__file__).parent
    registry = root / "registry.yaml"
    saved = registry.read_bytes()
    tmp = Path("E:/harness-data-test/tmp-plugin")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(root / "plugins" / "echo-test", tmp, ignore=shutil.ignore_patterns(".venv"))
    man = tmp / "manifest.yaml"
    man.write_text(man.read_text(encoding="utf-8-sig").replace("name: echo-test", "name: echo-new", 1), encoding="utf-8")
    try:
        pid = run("echo-test", "echo", {"text": "a"})["result"]["pid"]
        r = http("POST", "/reload")
        check("reload: nothing changed", r.get("added") == [] and r.get("changed") == [] and r.get("removed") == [], str(r)[:150])
        check("reload: untouched model keeps its process", run("echo-test", "echo", {"text": "b"})["result"]["pid"] == pid)

        t, b = bg(run, "echo-test", "echo", {"text": "long", "sleep_s": 3})
        time.sleep(1)
        http("POST", "/reload")
        t.join()
        check("reload: allowed while a request runs", b["r"].get("event") == "result", str(b["r"])[:150])

        r = http("POST", "/add_model", {"folder": "E:/harness-data-test/nope"})
        check("add_model: missing folder rejected", r.get("error", {}).get("code") == "invalid_input", str(r)[:150])
        (tmp / "env.lock").rename(tmp / "env.off")
        r = http("POST", "/add_model", {"folder": str(tmp)})
        (tmp / "env.off").rename(tmp / "env.lock")
        check("add_model: invalid plugin not registered",
              "error" in r and b"echo-new" not in registry.read_bytes(), str(r)[:150])
        r = http("POST", "/add_model", {"folder": str(tmp), "tags": ["lang:en"]})
        check("add_model: registered, no restart", r.get("added") == ["echo-new"], str(r)[:150])
        check("add_model: in the new catalog", "echo-new" in r.get("catalog", ""))
        r2 = run("echo-new", "echo", {"text": "new"})
        check("add_model: runs", r2.get("event") == "result", str(r2)[:150])
        r = http("POST", "/add_model", {"folder": str(tmp)})
        check("add_model: duplicate rejected", "already registered" in r.get("error", {}).get("message", ""), str(r)[:150])

        # new version drops an extra: requests accepted before the reload stay on the old version
        v1 = {f: (tmp / f).read_bytes() for f in ("manifest.yaml", "schemas/echo.json")}

        def drop_extra(name, keep):
            mtext = v1["manifest.yaml"].decode("utf-8-sig").replace("extras: [sleep_s, crash]", f"extras: [{keep}]")
            (tmp / "manifest.yaml").write_text(mtext, encoding="utf-8")
            schema = json.loads(v1["schemas/echo.json"].decode("utf-8-sig"))
            del schema["extras"][name]
            (tmp / "schemas" / "echo.json").write_text(json.dumps(schema), encoding="utf-8")

        def restore():
            for f, data in v1.items():
                (tmp / f).write_bytes(data)
            return http("POST", "/reload")

        t, b = bg(run, "echo-new", "echo", {"text": "old", "sleep_s": 3})
        time.sleep(1)
        t2, b2 = bg(run, "echo-new", "echo", {"text": "queued", "crash": False})
        time.sleep(0.3)
        drop_extra("crash", "sleep_s")
        r = http("POST", "/reload")
        check("reload: schema change detected", r.get("changed") == ["echo-new"], str(r)[:150])
        rej = run("echo-new", "echo", {"text": "x", "crash": False})
        check("reload: new requests use the new schema", rej.get("code") == "invalid_input", str(rej)[:150])
        t3, b3 = bg(run, "echo-new", "echo", {"text": "after"})
        t.join(); t2.join(); t3.join()
        ok = all(x["r"].get("event") == "result" for x in (b, b2, b3))
        check("reload: queued stays on old version, later gets new",
              ok and b["r"]["result"]["pid"] == b2["r"]["result"]["pid"] != b3["r"]["result"]["pid"],
              f"{str(b2['r'])[:90]} / {str(b3['r'])[:90]}")

        # old process dies with requests still queued: they are re-checked against the new version
        restore()
        run("echo-new", "echo", {"text": "load"})
        t, b = bg(run, "echo-new", "echo", {"text": "die", "sleep_s": 2, "crash": True})
        time.sleep(0.7)
        t2, b2 = bg(run, "echo-new", "echo", {"text": "fits"})
        t3, b3 = bg(run, "echo-new", "echo", {"text": "misfit", "sleep_s": 0.1})
        time.sleep(0.3)
        drop_extra("sleep_s", "crash")
        http("POST", "/reload")
        t.join(); t2.join(); t3.join()
        check("reload: old process gone, valid request moves on", b2["r"].get("event") == "result", str(b2["r"])[:150])
        check("reload: old process gone, invalid request fails clearly",
              b3["r"].get("code") == "plugin_changed", str(b3["r"])[:150])
        restore()

        good = registry.read_bytes()
        registry.write_bytes(good + b"\n  - : [broken")
        r = http("POST", "/reload")
        registry.write_bytes(good)
        check("reload: broken registry keeps the old catalog",
              "error" in r and run("echo-new", "echo", {"text": "x"}).get("event") == "result", str(r)[:150])

        registry.write_bytes(saved)
        r = http("POST", "/reload")
        check("reload: removed", r.get("removed") == ["echo-new"], str(r)[:150])
        time.sleep(1)
        check("reload: removed model unknown", run("echo-new", "echo", {"text": "x"}).get("code") == "unknown_model")
    finally:
        registry.write_bytes(saved)
        http("POST", "/reload")
        time.sleep(1)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
