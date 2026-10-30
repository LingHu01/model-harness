"""Input normalisation (base64 -> files in in/), validation, out/ dirs and retention (spec 18)."""
from __future__ import annotations

import base64
import binascii
import logging
import os
import re
import shutil
import time
from pathlib import Path

import jsonschema

from core.catalog import FunctionDef
from core.config import Config
from core.errors import HarnessError

log = logging.getLogger("io")
_DATA_URL = re.compile(r"^data:([\w/+.-]+)?;base64,(.*)$", re.S)
_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "audio/wav": ".wav",
        "audio/x-wav": ".wav", "audio/mpeg": ".mp3", "application/pdf": ".pdf"}


class IO:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def out_dir(self, rid: str) -> str:
        d = self.cfg.path("out", rid)
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    def normalise(self, rid: str, fn: FunctionDef, inputs: dict) -> dict:
        """base64 values ({'base64': ..., 'ext': '.png'} or data: URLs) become files; then validate."""
        if not isinstance(inputs, dict):
            raise HarnessError("invalid_input", "inputs must be an object")
        out = {}
        for k, v in inputs.items():
            out[k] = self._materialise(rid, k, v)
        try:
            jsonschema.validate(out, fn.input_schema)
        except jsonschema.ValidationError as e:
            raise HarnessError("invalid_input", f"{'/'.join(map(str, e.path)) or 'inputs'}: {e.message}")
        for k, spec in fn.input_schema.get("properties", {}).items():
            if spec.get("description") == "file path" and k in out and not os.path.exists(out[k]):
                raise HarnessError("invalid_input", f"{k}: file not found: {out[k]}")
        return out

    def _materialise(self, rid, key, v):
        data, ext = None, ".bin"
        if isinstance(v, dict) and "base64" in v:
            data, ext = v["base64"], v.get("ext", ".bin")
        elif isinstance(v, str) and v.startswith("data:"):
            m = _DATA_URL.match(v)
            if m:
                data, ext = m.group(2), _EXT.get(m.group(1) or "", ".bin")
        if data is None:
            return v
        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            raise HarnessError("invalid_input", f"{key}: bad base64")
        d = self.cfg.path("in", rid)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{key}{ext if ext.startswith('.') else '.' + ext}"
        p.write_bytes(raw)
        return str(p)

    def cleanup(self) -> int:
        """Delete in/ and out/ entries older than the retention window."""
        cutoff, n = time.time() - self.cfg.retention, 0
        for sub in ("in", "out"):
            for p in self.cfg.path(sub).iterdir():
                try:
                    if p.stat().st_mtime < cutoff:
                        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
                        n += 1
                except OSError:
                    pass
        return n

    def remove_inputs(self, rid: str):
        shutil.rmtree(self.cfg.path("in", rid), ignore_errors=True)
