# Model Harness

A local Windows server that lets an AI agent use models that aren't LLMs or diffusion models, such as vision, OCR and speech. The agent connects over MCP.

## What it does

- Runs each loaded model in its own process, with its own Python environment.
- Loads a model on the first request and frees it when idle, so an idle harness uses almost no VRAM or RAM.
- Measures how much memory each model needs and unloads idle models to make room.
- Queues requests and batches them where the plugin supports it.
- Adds or updates plugins without a restart.

## Setup

You need Windows, [uv](https://docs.astral.sh/uv/), Python 3.12, and an NVIDIA GPU for GPU plugins.

1. Clone the repo:

   ```bash
   git clone --filter=blob:none --sparse <repo-url>
   ```

2. Choose what to check out. Git leaves the rest out, on `git pull` too.

   | Target | Command |
   | --- | --- |
   | Core only | `git sparse-checkout set harness` |
   | Core and one plugin | `git sparse-checkout set harness plugins/wevisone-vl` |
   | Core and all plugins | `git sparse-checkout set harness plugins` |

3. Build the core and create your config:

   ```bash
   cd harness
   uv sync
   copy ..\config.example.yaml ..\config.yaml
   ```

   Keep `uv_cache_dir` on the same drive as `plugins/`.

4. Add the shim to your MCP client, using your own absolute paths:

   ```json
   { "command": "E:\\harness\\harness\\.venv\\Scripts\\python.exe", "args": ["E:\\harness\\harness\\shim\\shim.py"] }
   ```

5. Put the plugin's weights under `weights_root`. For WeVisOne, download `tencent/WeVisOne-VL-2B` and `tencent/WeVisOne-VL-4B` from Hugging Face into `wevisone-vl-2b` and `wevisone-vl-4b`.

6. From your agent, start the core and install the plugin:

   ```
   start()
   add_model("plugins/wevisone-vl")
   ```

   The plugin's environment builds on its first run, so that call is slow.

## Linux

Only Windows is implemented. A Linux version would replace `core/os_win.py` with `core/os_linux.py`:
child processes would use `PR_SET_PDEATHSIG` instead of a Job Object, and warm shells would pause with `SIGSTOP` and `SIGCONT`.
Peak memory would come from `ru_maxrss`, and PyTorch could use `expandable_segments`.
