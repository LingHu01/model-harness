"""Memory needs (measured / bootstrap), on-demand free-memory reads, leak/headroom flags (spec 12-13).

No background polling: every read happens at decision time.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import psutil

from core.catalog import ModelDef
from core.config import Config
from core.metrics import Store
from sdk import nvml


@dataclass
class Need:
    vram_gb: float
    ram_gb: float
    measured: bool
    raw_vram_gb: float = 0.0     # measured peak without the safety margin


def weights_size_gb(path: str) -> float:
    if not path or not os.path.exists(path):
        return 0.0
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1024 ** 3
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 ** 3


def weights_hash(path: str) -> str:
    """Cheap identity hash: names, sizes and mtimes (not content) of the weights folder."""
    if not path or not os.path.exists(path):
        return "none"
    p = Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
    h = hashlib.sha256()
    for f in files:
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


class Memory:
    def __init__(self, cfg: Config, store: Store):
        self.cfg, self.store = cfg, store
        self._whash: dict[str, str] = {}

    # ---- reads at decision time ----------------------------------------------
    @staticmethod
    def gpu(device: str):
        """(total_gb, free_gb) or None when no NVML / CPU device."""
        if not device.startswith("cuda"):
            return None
        m = nvml.memory(nvml.device_index(device))
        return None if m is None else (m[0], m[1])

    @staticmethod
    def free_ram_gb() -> float:
        return psutil.virtual_memory().available / 1024 ** 3

    # ---- needs --------------------------------------------------------------------
    def whash(self, m: ModelDef) -> str:
        if m.name not in self._whash:
            self._whash[m.name] = weights_hash(m.weights_path)
        return self._whash[m.name]

    def forget_hashes(self):
        self._whash.clear()

    def key(self, m: ModelDef) -> str:
        return f"{m.plugin_version}:{self.whash(m)}"

    def need(self, m: ModelDef) -> Need:
        """base_need = rolling max of recent peaks (+margin); bootstrap before the first run."""
        rows = self.store.recent_runs(m.name, m.plugin_version, self.whash(m), self.cfg.need_window)
        cpu = not m.device.startswith("cuda")
        gpu = None if cpu else self.gpu(m.device)
        if gpu:  # a peak above the physical card spilled to RAM (Windows sysmem fallback): not a real need
            rows = [r for r in rows if (r[1] or 0) <= gpu[0]]
        singles = [r for r in rows if r[0] == 1]
        if singles:  # single-request runs measure base_need directly; batches only estimate it
            rows = singles
        if rows:
            peaks, rams = [], []
            for k, vp, vpe, rp, _va, _oom, ext_valid in rows:
                base = (vp or 0.0) - max(0, k - 1) * m.vram_per_extra_gb
                ext = (vpe or 0.0) - max(0, k - 1) * m.vram_per_extra_gb if ext_valid else 0.0
                peaks.append(max(base, ext))
                rams.append(rp or 0.0)
            raw = 0.0 if cpu else max(peaks)
            vram = raw * (1 + self.cfg.need_margin)
            ram = max(rams) * (1 + self.cfg.need_margin)
            stored = self.store.get_need(m.name, self.key(m))
            if stored:  # load-time OOM may have raised the stored need above observed run peaks
                vram, ram = max(vram, stored[0] or 0), max(ram, stored[1] or 0)
            return Need(round(vram, 3), round(ram, 3), True, round(raw, 3))
        stored = self.store.get_need(m.name, self.key(m))
        if stored:
            return Need(stored[0], stored[1], True)
        w = weights_size_gb(m.weights_path)
        vram = 0.0 if cpu else w * self.cfg.bootstrap_factor + self.cfg.cuda_context_gb
        return Need(round(vram, 3), self.cfg.bootstrap_ram_gb, False)

    def raise_need(self, m: ModelDef, vram_gb: float, ram_gb: float | None = None):
        cur = self.need(m)
        self.store.set_need(m.name, self.key(m), max(cur.vram_gb, vram_gb), max(cur.ram_gb, ram_gb or 0))

    def need_k(self, m: ModelDef, k: int) -> float:
        return self.need(m).vram_gb + (k - 1) * m.vram_per_extra_gb

    # ---- flags ------------------------------------------------------------------------
    def flags(self, m: ModelDef) -> dict:
        rows = self.store.recent_runs(m.name, m.plugin_version, self.whash(m), self.cfg.need_window)
        after = [r[4] for r in rows[:5] if r[4] is not None]
        leak = len(after) >= 5 and all(after[i] > after[i + 1] for i in range(len(after) - 1))
        ooms = sum(1 for r in rows if r[5] and json.loads(r[5]))
        return {"leak": leak, "headroom": ooms >= 3}
