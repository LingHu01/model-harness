"""Smoke test: detect the red circle in tests/sample.png through the core API.

    python tests/smoke.py [model] [port]
"""
import json
import pathlib
import sys
import urllib.request

model = sys.argv[1] if len(sys.argv) > 1 else "wevisone-vl-2b"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
img = str(pathlib.Path(__file__).with_name("sample.png").resolve())
body = json.dumps({"model": model, "function": "detect", "inputs": {"image": img, "query": "red circle"}}).encode()
with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/run", body), timeout=600) as r:
    last = [json.loads(l) for l in r][-1]
assert last["event"] == "result", last
inst = last["result"]["instances"]
assert inst, last["result"]
x0, y0, x1, y1 = inst[0]["box"]
assert x0 < 320 < x1 and y0 < 240 < y1, inst
print("ok", inst[0], last["metrics"].get("duration_s"))
