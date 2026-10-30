"""Run every WeVisOne function on tests/bench/scene.png for the given variants (core must be running).

    python tests/all_functions.py [wevisone-vl-2b wevisone-vl-4b] [--port 8765]
"""
import json
import pathlib
import sys
import urllib.request

args = [a for a in sys.argv[1:] if not a.startswith("--")]
port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8765
models = [a for a in args if not a.isdigit()] or ["wevisone-vl-2b", "wevisone-vl-4b"]
img = str(pathlib.Path(__file__).with_name("bench").joinpath("scene.png").resolve())
CALLS = [("detect", {"query": "red circle\nblue rectangle"}), ("refer", {"query": "the blue rectangle"}),
         ("point", {"query": "red circle"}), ("count", {"query": "red circles"}),
         ("segment", {"query": "the blue rectangle"}), ("matting", {"query": "the leftmost red circle"}),
         ("ocr", {}), ("detect", {"query": "giraffe"})]


def run(model, fn, inputs):
    body = json.dumps({"model": model, "function": fn, "inputs": {"image": img, **inputs}}).encode()
    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/run", body), timeout=900) as r:
        return [json.loads(l) for l in r][-1]


def summary(fn, res):
    if fn in ("detect", "refer"):
        return [(i["label"], i["box"]) for i in res["instances"]]
    if fn == "point":
        return [p["point"] for p in res["points"]]
    if fn == "count":
        return res["count"]
    if fn == "ocr":
        return [w["text"] for w in res["words"]]
    return [(i["box"], pathlib.Path(i.get("mask") or i.get("alpha")).name) for i in res["instances"]]


for model in models:
    for fn, inputs in CALLS:
        ev = run(model, fn, inputs)
        if ev["event"] != "result":
            print(f"{model:15} {fn:8} ERROR {ev.get('code')}: {ev.get('message')}")
            continue
        m = ev["metrics"]
        cs = m.get("cold_start") or {}
        load = f" load={cs.get('total_s')}s" if cs.get("total_s") else ""
        print(f"{model:15} {fn:8} {m['duration_s']:6.2f}s vram={m.get('vram_peak_gb')}{load}  {summary(fn, ev['result'])}")
