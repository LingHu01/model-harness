# Model Harness

Local harness for "particular" models (vision, TTS, STT, OCR…) — one process per loaded model, ~0 VRAM/RAM when idle, exposed to QwenPaw over MCP. Spec: `E:\Downloads\model-harness-spec.md`.

```
QwenPaw ─MCP stdio─> shim/shim.py ─HTTP 127.0.0.1:8765─> core (pythonw, single instance)
                                                           └─ JSON lines ─> runner/runner.py in plugins/<model>/.venv
```

## Layout on this machine

| What | Where |
| --- | --- |
| code + definitions (git) | `E:\harness` |
| DATA ROOT (cache, state db, logs, in/out) | `E:\harness-data` |
| WEIGHTS ROOT | `E:\harness-weights` |
| uv cache (same volume as plugins → hardlinks) | `E:\harness-data\uv-cache` |

## Hook up QwenPaw (MCP, stdio)

```json
{ "command": "E:\\harness\\.venv\\Scripts\\python.exe", "args": ["E:\\harness\\shim\\shim.py"] }
```

The agent calls `start()` once, then `run(model, function, inputs)`. The catalog is in the `run` tool description.

## What is in git

The repo is the core only: `core/`, `shim/`, `runner/`, `sdk/`, `tools/`, `schemas/base/`, `tests/`. Everything below stays on the machine and is ignored:

| Local | What | Start from |
| --- | --- | --- |
| `config.yaml` | paths, port, timers | `config.example.yaml` |
| `registry.yaml` | installed plugins | `registry.example.yaml` |
| `plugins/` | plugins (a plugin can be its own repo; paths in the registry may be absolute) | |
| `DATA_ROOT\state	uning.yaml` | concurrency tuned on this machine, written by the bench; ignored if the GPU changes | |

The self-test plugins are fixtures in `tests/plugins`, with their own `tests/registry.yaml`.

## Adding or changing a plugin (no restart)

The core only needs a restart when the core itself is upgraded, or if it hangs.

- `add_model(folder, tags)`: validates the plugin folder, adds it to `registry.yaml` (one entry per variant) and makes it available at once. `cap:` tags come from the manifest. An invalid plugin is rejected and nothing is written.
- `reload()`: after editing a plugin, the registry or tuning. Untouched models stay loaded, even mid-request. A changed plugin (manifest, adapter, `env.lock`, weights path, schemas) keeps its old process until every request accepted before the reload is done, running or queued, because those inputs were validated against the old schema. Requests accepted after the reload follow the new schema, wait for those, then get a new process. If the old process dies first, its code is gone from disk: each queued request moves to the new version if its inputs are still valid there, otherwise it fails with `plugin_changed`. A removed model finishes its queue the same way and goes away. A broken registry is refused and the previous catalog stays active. Refused during a bench.

Both are MCP tools; over HTTP they are `POST /add_model {"folder": ..., "tags": [...]}` and `POST /reload`. Both return the new catalog and update the `run` tool description; clients that ignore the MCP tool-list notification still get it in the result. Changing other adapter files is not detected: only the entry file is hashed, so unload the model by hand.

## Plugins on this machine

| Model(s) | Folder | Notes |
| --- | --- | --- |
| `wevisone-vl-2b`, `wevisone-vl-4b` | `plugins/wevisone-vl` | **variants**: one env/adapter, different weights; each loads in its own process. Tencent `wevisone_vl` pinned to commit 9a6bcd8 in `env.lock`. `tests/all_functions.py` runs all 7 functions |

Variants: a manifest may declare `variants: {name: {overrides…}}`; each registry entry names one (`variant:` or its `name`). Overrides are deep-merged onto the manifest (weights, description, limits…). Relative `weights.path` resolves against `weights_root`.

## Tests

```bash
# terminal 1
set HARNESS_CONFIG=E:\harness\tests\selftest.yaml && E:\harness\.venv\Scripts\python.exe -m core.server
# terminal 2
E:\harness\.venv\Scripts\python.exe E:\harness\tests\selftest.py --gpu
E:\harness\.venv\Scripts\python.exe E:\harness\tests\shim_test.py   # MCP round trip (start/run/stop)
```

## Status vs spec phases

- 1 Runner + plugin, metrics — done
- 2 Worker lifecycle, FIFO, R1/R4/R5, cancel — done
- 3 Memory manager (measured needs, bootstrap, R3 eviction, R6 external holders), shim, catalog, progress, Job Object, single instance, windowless — done
- 4 Warm shells: R1 replacement shell, paging (EmptyWorkingSet + NtSuspendProcess), R2 expiry — done
- 5 Concurrency: k-selection, thread-safe slots, same-function batches, batch split on OOM, bench tool — done
  (`.venv\Scripts\python.exe tools\bench\bench.py <model> [--max 16]`; writes the manifest only after you confirm)
- 6 OOM ladder in runner — done; partial offload is adapter-specific (hook: `offload_more`, budgets in `load`)
- 7 Artifacts: venv build from `env.lock` on request — done; keys computed; **compiled engines not built**

Windows "CUDA sysmem fallback": when VRAM runs out, the driver spills to system RAM over PCIe instead of
raising OOM, so the OOM ladder never fires and the run just crawls (seen: a 4B batch at 18 GB on a 16 GB card,
killed by the runaway timer). The harness flags `vram_spill` in metrics, ignores spilled runs for needs, and the
bench treats spilled steps as invalid. Tune with real inputs (`--inputs`), since VRAM per extra request grows
with image size. Optional: NVIDIA Control Panel → CUDA - Sysmem Fallback Policy → "Prefer No Sysmem Fallback"
for the plugin pythons turns spills back into OOM errors.

Deviations from the spec:
- The core is launched with the venv's console `python.exe` + `CREATE_NO_WINDOW`, not `pythonw.exe`. uv's pythonw launcher starts a console `python.exe` child, which opens a visible console window. `tests/window_test.py` checks that no window appears.
- `ram_peak_gb` uses peak working set, not peak private bytes. Under WDDM, CUDA allocations count as private commit, so private bytes would report VRAM as RAM.
- `plugin_version` includes the variant name, so each variant keeps its own measured need.
