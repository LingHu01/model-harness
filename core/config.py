"""config.yaml loading with defaults. All paths resolved here (portable: nothing OS-specific)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

HARNESS_ROOT = Path(__file__).resolve().parent.parent
VERSION = "0.1.0"

_DUR = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)?\s*$")
_UNIT = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, None: 1}


def duration(v, default: float | None = None) -> float:
    """'30m' -> 1800.0; numbers are seconds."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    m = _DUR.match(str(v))
    if not m:
        raise ValueError(f"bad duration: {v!r}")
    return float(m.group(1)) * _UNIT[m.group(2)]


@dataclass
class Config:
    root: Path
    data_root: Path
    weights_root: Path
    uv_cache_dir: Path
    registry: Path               # local list of installed plugins (untracked)
    default_python: str
    port: int
    idle_timeout: float
    shell_expiry: float
    queue_report: float
    queue_stall: float
    runaway: float
    retention: float
    shutdown_grace: float
    need_margin: float
    need_window: int
    bootstrap_factor: float
    cuda_context_gb: float
    bootstrap_ram_gb: float
    sampler_ms: int
    compile_vram_cap: float
    compile_min_runs: int
    log_max_bytes: int
    log_backups: int

    def path(self, *parts) -> Path:
        return self.data_root.joinpath(*parts)


def _resolve(p, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def load(path: Path | None = None) -> Config:
    path = path or HARNESS_ROOT / "config.yaml"
    if not path.exists():  # settings are per machine and untracked
        raise SystemExit(f"{path} missing: copy {HARNESS_ROOT / 'config.example.yaml'} and edit it")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent
    t, mem, comp, logs = raw.get("timers", {}), raw.get("memory", {}), raw.get("compile", {}), raw.get("logs", {})
    data_root = _resolve(raw.get("data_root", "data"), base)
    cfg = Config(
        root=HARNESS_ROOT,
        data_root=data_root,
        weights_root=_resolve(raw.get("weights_root", "weights"), base),
        uv_cache_dir=_resolve(raw.get("uv_cache_dir", data_root / "uv-cache"), base),
        registry=_resolve(raw.get("registry", "registry.yaml"), base),
        default_python=str(raw.get("default_python", "3.12")),
        port=int(raw.get("port", 8765)),
        idle_timeout=duration(t.get("idle_timeout"), 1800),
        shell_expiry=duration(t.get("shell_expiry"), 3 * 86400),
        queue_report=duration(t.get("queue_report"), 30),
        queue_stall=duration(t.get("queue_stall"), 120),
        runaway=duration(t.get("runaway"), 300),
        retention=duration(t.get("retention"), 86400),
        shutdown_grace=duration(t.get("shutdown_grace"), 30),
        need_margin=float(mem.get("need_margin", 0.10)),
        need_window=int(mem.get("need_window", 20)),
        bootstrap_factor=float(mem.get("bootstrap_factor", 1.2)),
        cuda_context_gb=float(mem.get("cuda_context_gb", 0.5)),
        bootstrap_ram_gb=float(mem.get("bootstrap_ram_gb", 2.0)),
        sampler_ms=int(mem.get("sampler_ms", 75)),
        compile_vram_cap=float(comp.get("vram_cap", 0.70)),
        compile_min_runs=int(comp.get("min_runs_7d", 20)),
        log_max_bytes=int(logs.get("max_bytes", 5_000_000)),
        log_backups=int(logs.get("backups", 3)),
    )
    for d in ("cache", "state", "run", "in", "out", "logs"):
        cfg.path(d).mkdir(parents=True, exist_ok=True)
    return cfg
