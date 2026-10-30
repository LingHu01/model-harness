"""Helpers plugins import. Pure Python; torch is imported lazily and only when a helper needs it."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def setup_offline_env() -> None:
    """Offline mode + allocator settings. Must run before torch/transformers are imported."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if sys.platform != "win32":  # not supported by PyTorch on Windows
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None)


@dataclass
class Context:
    weights_path: str = ""
    device: str = "cuda:0"
    vram_budget_gb: float = 0.0
    ram_budget_gb: float = 0.0
    offload: str = "partial"
    strategy: str = "standard"
    engine_path: str | None = None
    plugin_dir: str = ""
    cache_dir: str = ""
    out_dir: str = ""                    # set per request
    extra: dict[str, Any] = field(default_factory=dict)

    def out_path(self, name: str) -> str:
        """Path for an output file inside this request's out_dir."""
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        p = (Path(self.out_dir) / name).resolve()
        if Path(self.out_dir).resolve() not in p.parents:
            raise ValueError(f"output name escapes out_dir: {name}")
        return str(p)


class BaseAdapter:
    """Plugin adapters subclass this and implement load / run (run_batch optional)."""

    def load(self, ctx: Context) -> None:
        raise NotImplementedError

    def run(self, function: str, inputs: dict, ctx: Context) -> dict:
        raise NotImplementedError

    def run_batch(self, function: str, inputs_list: list[dict], ctx: Context) -> list[dict]:
        return [self.run(function, i, ctx) for i in inputs_list]

    def unload(self) -> None:
        pass

    # Optional, for compile eligibility: def export_engine(self, ctx, out_path): ...


class PluginError(Exception):
    """Raise from an adapter to return a specific error code (e.g. invalid_input, model_too_large)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# ---- weights helpers ---------------------------------------------------------

def safetensors_files(weights_path: str) -> list[str]:
    p = Path(weights_path)
    files = [p] if p.is_file() else sorted(p.glob("*.safetensors"))
    if not files:
        raise PluginError("invalid_input", f"no .safetensors under {weights_path} (safetensors only)")
    return [str(f) for f in files]


def meta_init(build_fn):
    """Construct a torch module on the meta device (no memory, no init cost)."""
    import torch
    with torch.device("meta"):
        return build_fn()


def load_safetensors_into(module, weights_path: str, device: str, strict: bool = True, dtype=None):
    """Load safetensors straight to `device` and assign into a meta-initialised module."""
    from safetensors.torch import load_file
    state = {}
    for f in safetensors_files(weights_path):
        state.update(load_file(f, device=device))
    if dtype is not None:
        state = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in state.items()}
    module.load_state_dict(state, strict=strict, assign=True)
    return module.to(device)


def is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name == "OutOfMemoryError" or "CUDA out of memory" in str(exc)
