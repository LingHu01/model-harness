"""Required artifacts built lazily on request (R7): plugin .venv from env.lock via uv.

Compiled engines (spec 14.3) are phase 7; artifact keys are computed here so status() can report them.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

from core.catalog import ModelDef
from core.config import VERSION, Config
from core.errors import HarnessError
from core.osal import osx
from sdk import nvml

log = logging.getLogger("artifacts")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "none"


def plugin_python_version(cfg: Config, m: ModelDef) -> str:
    pv = m.folder / ".python-version"
    return pv.read_text().strip() if pv.exists() else cfg.default_python


def artifact_key(cfg: Config, m: ModelDef, weights_hash: str) -> dict:
    idx = nvml.device_index(m.device) if m.device.startswith("cuda") else 0
    return {"gpu": nvml.device_name(idx), "driver": nvml.driver_version(),
            "env_lock": _sha(m.folder / "env.lock"), "python": plugin_python_version(cfg, m),
            "plugin_version": m.plugin_version, "weights": weights_hash, "harness": VERSION}


class Artifacts:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._locks: dict[str, asyncio.Lock] = {}

    def venv_dir(self, m: ModelDef) -> Path:
        return m.folder / ".venv"

    def python(self, m: ModelDef) -> str:
        return osx.venv_python(str(self.venv_dir(m)))

    def _stamp(self, m: ModelDef) -> dict:
        return {"env_lock": _sha(m.folder / "env.lock"), "python": plugin_python_version(self.cfg, m)}

    def venv_valid(self, m: ModelDef) -> bool:
        stamp = self.venv_dir(m) / ".harness-env.json"
        if not stamp.exists() or not os.path.exists(self.python(m)):
            return False
        try:
            return json.loads(stamp.read_text()) == self._stamp(m)
        except Exception:
            return False

    async def ensure_venv(self, m: ModelDef) -> None:
        if self.venv_valid(m):
            return
        lock = self._locks.setdefault(str(m.folder), asyncio.Lock())  # variants share one venv
        async with lock:
            if self.venv_valid(m):
                return
            await self._build_venv(m)

    async def _build_venv(self, m: ModelDef) -> None:
        venv = self.venv_dir(m)
        logf = self.cfg.path("logs", f"{m.name}.build.log")
        env = {**os.environ, "UV_CACHE_DIR": str(self.cfg.uv_cache_dir), "UV_LINK_MODE": "hardlink"}
        uv = shutil.which("uv") or "uv"
        reqs = [l for l in (m.folder / "env.lock").read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("--")]
        steps = [[uv, "venv", "--clear", "--python", plugin_python_version(self.cfg, m), str(venv)]]
        if reqs:
            steps.append([uv, "pip", "sync", str(m.folder / "env.lock"), "--python", self.python(m),
                          "--index-strategy", "unsafe-best-match", "--compile-bytecode"])
        log.info("building venv for %s", m.name)
        with open(logf, "ab") as lf:
            for cmd in steps:
                lf.write((" ".join(cmd) + "\n").encode())
                lf.flush()
                proc = await asyncio.create_subprocess_exec(*cmd, stdout=lf, stderr=lf, env=env,
                                                            cwd=str(m.folder), **osx.runner_spawn_kwargs())
                if await proc.wait() != 0:
                    raise HarnessError("env_build_failed", f"{m.name}: '{cmd[1]} {cmd[2]}' failed; see {logf}")
        (venv / ".harness-env.json").write_text(json.dumps(self._stamp(m)))
        log.info("venv for %s ready", m.name)
