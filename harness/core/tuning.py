"""Per-machine concurrency tuning (written by tools/bench), kept in DATA_ROOT, not in git.

concurrency.max and vram_per_extra_gb depend on the card, the driver and the real input sizes, so they
are local measurements. The manifest keeps what the adapter can do (batching, thread_safe) and a
portable default. Tuning is not part of plugin_version: re-tuning keeps measured memory needs.

    # DATA_ROOT/state/tuning.yaml
    wevisone-vl-2b:
      gpu: NVIDIA GeForce RTX ...
      tuned: '2026-10-30'
      max: 8                     # highest of the functions
      vram_per_extra_gb: 0.22
      functions: {matting: {max: 4, vram_per_extra_gb: 0.22}, ocr: {max: 8, vram_per_extra_gb: 0.1}}
"""
from __future__ import annotations

import datetime
import logging

import yaml

from core.config import Config

log = logging.getLogger("tuning")

HEADER = "# Per-machine concurrency tuning written by tools/bench (not tracked). Delete an entry to fall back to the manifest.\n"


def path(cfg: Config):
    return cfg.path("state", "tuning.yaml")


def load(cfg: Config) -> dict:
    p = path(cfg)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8-sig")) or {}
    except Exception as e:
        log.error("bad %s, ignored: %s", p, e)
        return {}


def save(cfg: Config, model: str, functions: dict[str, dict], gpu: str | None):
    """`functions`: {name: {max, vram_per_extra_gb}}, each tuned on its own."""
    data = load(cfg)
    functions = {**(data.get(model) or {}).get("functions", {}), **functions}   # other functions keep their tuning
    data[model] = {"max": max(f["max"] for f in functions.values()),
                   "vram_per_extra_gb": max(f["vram_per_extra_gb"] for f in functions.values()),
                   "functions": functions, "gpu": gpu, "tuned": datetime.date.today().isoformat()}
    path(cfg).write_text(HEADER + yaml.safe_dump(data, sort_keys=True, default_flow_style=None), encoding="utf-8")


def for_model(cfg: Config, tuning: dict, model: str, device: str) -> dict | None:
    """The entry for `model`, or None when absent or tuned on a different GPU."""
    t = tuning.get(model)
    if not t:
        return None
    if t.get("gpu") and device.startswith("cuda"):
        from sdk import nvml
        cur = nvml.device_name(nvml.device_index(device))
        if cur and cur != t["gpu"]:
            log.warning("%s: tuning was for %s, this device is %s; using manifest defaults", model, t["gpu"], cur)
            return None
    return t
