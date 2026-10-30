"""Smoke test: one quick call through the core API."""
import json
import sys
import urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
body = json.dumps({"model": "echo-test", "function": "echo", "inputs": {"text": "smoke"}}).encode()
with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/run", body), timeout=120) as r:
    last = [json.loads(l) for l in r][-1]
assert last["event"] == "result" and last["result"]["text"] == "smoke", last
print("ok")
