"""SQLite state store: runs, needs, compile state, artifacts, request log (spec 13)."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(model TEXT, ts REAL, function TEXT, k INTEGER, vram_peak REAL,
    vram_peak_ext REAL, ram_peak REAL, vram_after REAL, duration REAL, cold_start_json TEXT,
    placement_json TEXT, oom_recoveries TEXT, plugin_version TEXT, weights_hash TEXT, ext_valid INTEGER);
CREATE INDEX IF NOT EXISTS runs_model_ts ON runs(model, ts);
CREATE TABLE IF NOT EXISTS needs(model TEXT, key TEXT, base_need_vram REAL, need_ram REAL, updated REAL,
    PRIMARY KEY(model, key));
CREATE TABLE IF NOT EXISTS compile_state(model TEXT, key TEXT, state TEXT, reason TEXT, updated REAL,
    PRIMARY KEY(model, key));
CREATE TABLE IF NOT EXISTS artifacts(model TEXT, kind TEXT, key_hash TEXT, path TEXT, built REAL, size INTEGER);
CREATE TABLE IF NOT EXISTS requests(request_id TEXT PRIMARY KEY, model TEXT, function TEXT, state TEXT,
    ts_in REAL, ts_start REAL, ts_end REAL, error_code TEXT);
"""


class Store:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    # ---- requests ---------------------------------------------------------------
    def request_in(self, rid, model, function):
        self.db.execute("INSERT OR REPLACE INTO requests VALUES(?,?,?,?,?,?,?,?)",
                        (rid, model, function, "queued", time.time(), None, None, None))

    def request_start(self, rid):
        self.db.execute("UPDATE requests SET state='running', ts_start=? WHERE request_id=?", (time.time(), rid))

    def request_end(self, rid, error_code=None):
        self.db.execute("UPDATE requests SET state=?, ts_end=?, error_code=? WHERE request_id=?",
                        ("failed" if error_code else "done", time.time(), error_code, rid))

    # ---- runs -------------------------------------------------------------------
    def add_run(self, model, function, m: dict, plugin_version, weights_hash, ext_valid: bool):
        self.db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            model, time.time(), function, m.get("k", 1), m.get("vram_peak_gb"), m.get("vram_peak_ext_gb"),
            m.get("ram_peak_gb"), m.get("vram_after_gb"), m.get("duration_s"),
            json.dumps(m.get("cold_start")) if m.get("cold_start") else None,
            json.dumps(m.get("placement")), json.dumps(m.get("oom_recoveries") or []),
            plugin_version, weights_hash, int(ext_valid)))

    def recent_runs(self, model, plugin_version, weights_hash, n):
        cur = self.db.execute(
            "SELECT k, vram_peak, vram_peak_ext, ram_peak, vram_after, oom_recoveries, ext_valid FROM runs "
            "WHERE model=? AND plugin_version=? AND weights_hash=? ORDER BY ts DESC LIMIT ?",
            (model, plugin_version, weights_hash, n))
        return cur.fetchall()

    def runs_since(self, model, ts) -> int:
        return self.db.execute("SELECT COUNT(*) FROM runs WHERE model=? AND ts>=?", (model, ts)).fetchone()[0]

    # ---- needs ------------------------------------------------------------------
    def set_need(self, model, key, vram, ram):
        self.db.execute("INSERT OR REPLACE INTO needs VALUES(?,?,?,?,?)", (model, key, vram, ram, time.time()))

    def get_need(self, model, key):
        r = self.db.execute("SELECT base_need_vram, need_ram FROM needs WHERE model=? AND key=?",
                            (model, key)).fetchone()
        return r

    # ---- compile state ----------------------------------------------------------
    def compile_state(self, model, key):
        r = self.db.execute("SELECT state, reason FROM compile_state WHERE model=? AND key=?", (model, key)).fetchone()
        return {"state": r[0], "reason": r[1]} if r else None

    def set_compile_state(self, model, key, state, reason=""):
        self.db.execute("INSERT OR REPLACE INTO compile_state VALUES(?,?,?,?,?)",
                        (model, key, state, reason, time.time()))

    def close(self):
        self.db.close()
