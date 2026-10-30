"""Registry (L0) + manifests (L1) + schemas (L2): load, validate (spec 4.4), build the agent catalog."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema
import yaml

from core import tuning as tuning_mod
from core.config import Config, duration

log = logging.getLogger("catalog")


@dataclass
class FunctionDef:
    name: str
    capability: str
    summary: str
    extras: list[str]
    schema: dict                 # L2
    input_schema: dict           # base input + extras, used for validation


@dataclass
class ModelDef:
    name: str
    tags: list[str]
    folder: Path
    description: str
    functions: dict[str, FunctionDef]
    entry: str = "adapter.py"
    device: str = "cuda:0"
    max_concurrency: int = 1
    batching: bool = False
    thread_safe: bool = False
    mixed_functions: bool = False  # one batch may hold different functions
    function_max: dict[str, int] = field(default_factory=dict)  # tuned batch limit per function
    vram_per_extra_gb: float = 0.0
    offload: str = "partial"
    compile_eligible: bool = True
    runaway: float = 300.0
    weights_path: str = ""
    plugin_version: str = ""     # hash of manifest + adapter
    fingerprint: str = ""        # plugin_version + env.lock + weights path + schemas: a change needs a new process
    raw_manifest: dict = field(default_factory=dict)


class Catalog:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.contracts: dict[str, dict] = {}
        self.models: dict[str, ModelDef] = {}
        self.invalid: dict[str, str] = {}
        self.text = ""

    # ---- loading --------------------------------------------------------------
    def load(self) -> "Catalog":
        """Build everything first, swap at the end: a broken registry leaves the running catalog intact."""
        root = self.cfg.root
        contracts, models, invalid = {}, {}, {}
        for f in sorted((root / "schemas" / "base").glob("*.json")):
            try:
                c = json.loads(f.read_text(encoding="utf-8-sig"))
                contracts[c["capability"]] = c
            except Exception as e:
                log.error("bad base contract %s: %s", f.name, e)
        if not self.cfg.registry.exists():
            log.warning("no registry at %s: no models", self.cfg.registry)
        reg = (yaml.safe_load(self.cfg.registry.read_text(encoding="utf-8-sig")) or {}
               if self.cfg.registry.exists() else {})
        self.contracts, self._tuning = contracts, tuning_mod.load(self.cfg)
        for entry in reg.get("models", []) or []:
            name = entry.get("name", "?")
            if entry.get("enabled", True) is False:
                continue
            if name in models or name in invalid:
                invalid[name] = "duplicate name"
                models.pop(name, None)
                continue
            try:
                models[name] = self._load_model(entry)
            except Exception as e:
                invalid[name] = str(e)
                log.warning("model %s excluded: %s", name, e)
        self.models, self.invalid = models, invalid
        self.text = self._render()
        return self

    def add_plugin(self, folder: str, tags: list[str] | None = None, manifest: str = "manifest.yaml") -> list[str]:
        """Validate a plugin folder and append it to the registry (one entry per variant). cap: tags come
        from the manifest's functions, the rest from its `tags:` (base and per variant) plus `tags`."""
        reg_dir = self.cfg.registry.parent
        path = (reg_dir / folder).resolve()
        mpath = path / manifest
        if not mpath.is_file():
            raise ValueError(f"no {manifest} in {path}")
        man = yaml.safe_load(mpath.read_text(encoding="utf-8-sig")) or {}
        names = list(man.get("variants") or []) or [man.get("name")]
        if not all(names):
            raise ValueError("manifest has no name")
        try:
            rel = path.relative_to(reg_dir).as_posix()
        except ValueError:
            rel = path.as_posix()
        caps = sorted({f"cap:{fn.get('capability')}" for fn in man.get("functions") or []})
        text = self.cfg.registry.read_text(encoding="utf-8-sig") if self.cfg.registry.exists() else "version: 1\nmodels:\n"
        known = yaml.safe_load(text) or {"version": 1}
        known["models"] = known.get("models") or []
        taken = {e.get("name") for e in known["models"]} & set(names)
        if taken:
            raise ValueError(f"already registered: {', '.join(sorted(taken))}")
        entries = []
        for n in names:
            own = ((man.get("variants") or {}).get(n) or {}).get("tags") or []
            extra = list(dict.fromkeys([*(man.get("tags") or []), *own, *(tags or [])]))
            e = {"name": n, "tags": caps + [t for t in extra if t not in caps],
                 "folder": rel, "manifest": f"{rel}/{manifest}"}
            if man.get("variants"):
                e["variant"] = n
            self._load_model(e)  # raises on anything invalid: nothing is written
            entries.append(e)
        new = text.rstrip() + "\n" + yaml.safe_dump(entries, sort_keys=False, default_flow_style=None)
        try:  # a plain append keeps comments; `models: []` or odd layouts fall back to a full rewrite
            ok = len(yaml.safe_load(new)["models"]) == len(known["models"]) + len(entries)
        except Exception:
            ok = False
        if not ok:
            new = yaml.safe_dump({**known, "models": known["models"] + entries}, sort_keys=False)
        self.cfg.registry.write_text(new, encoding="utf-8")
        return names

    def _load_model(self, entry: dict) -> ModelDef:
        root = self.cfg.registry.parent  # plugin paths are relative to the registry file
        for k in ("name", "tags", "folder", "manifest"):
            if k not in entry:
                raise ValueError(f"registry entry missing '{k}'")
        folder = (root / entry["folder"]).resolve()
        mpath = (root / entry["manifest"]).resolve()
        if not folder.is_dir():
            raise ValueError(f"folder missing: {folder}")
        if not mpath.is_file():
            raise ValueError(f"manifest missing: {mpath}")
        if not (folder / "env.lock").is_file():
            raise ValueError("env.lock missing")
        if not (folder / "tests" / "smoke.py").is_file():
            raise ValueError("tests/smoke.py missing")
        man = yaml.safe_load(mpath.read_text(encoding="utf-8-sig")) or {}
        variants = man.get("variants") or {}
        if variants:
            # One plugin folder / env / adapter, several weights: each variant is its own model+process.
            vname = entry.get("variant", entry["name"])
            if vname not in variants:
                raise ValueError(f"variant {vname!r} not in manifest variants {sorted(variants)}")
            man = _merge({k: v for k, v in man.items() if k != "variants"}, variants[vname] or {})
            man["name"] = entry["name"]
        if man.get("name") != entry["name"]:
            raise ValueError(f"manifest name {man.get('name')!r} != registry name {entry['name']!r}")
        tags = list(entry["tags"])
        cap_tags = {t[4:] for t in tags if t.startswith("cap:")}
        for t in cap_tags:
            if t not in self.contracts:
                raise ValueError(f"tag cap:{t} has no base contract")

        functions: dict[str, FunctionDef] = {}
        for fn in man.get("functions") or []:
            fname, cap = fn.get("name"), fn.get("capability")
            if cap not in self.contracts:
                raise ValueError(f"function {fname}: capability {cap} has no base contract")
            if cap not in cap_tags:
                raise ValueError(f"function {fname}: no matching cap:{cap} tag")
            spath = folder / fn.get("schema", f"schemas/{fname}.json")
            if not spath.is_file():
                raise ValueError(f"function {fname}: schema missing")
            schema = json.loads(spath.read_text(encoding="utf-8-sig"))
            contract = self.contracts[cap]
            if schema.get("base") != cap:
                raise ValueError(f"function {fname}: schema base {schema.get('base')} != {cap}")
            if schema.get("base_version") != contract.get("version"):
                raise ValueError(f"function {fname}: base_version {schema.get('base_version')} "
                                 f"!= contract version {contract.get('version')}")
            extras = list(fn.get("extras") or [])
            if set(extras) != set((schema.get("extras") or {}).keys()):
                raise ValueError(f"function {fname}: extras mismatch manifest vs schema")
            base_props = contract["input"].get("properties", {})
            if set(extras) & set(base_props):
                raise ValueError(f"function {fname}: extras redefine base params")
            inp = copy.deepcopy(contract["input"])
            inp.setdefault("properties", {}).update(schema.get("extras") or {})
            inp["additionalProperties"] = False
            jsonschema.Draft202012Validator.check_schema(inp)
            functions[fname] = FunctionDef(fname, cap, fn.get("summary", ""), extras, schema, inp)
        if not functions:
            raise ValueError("no functions")

        rt = man.get("runtime") or {}
        conc = man.get("concurrency") or {}
        cmax = int(conc.get("max", 1))
        per_extra = float(conc.get("vram_per_extra_gb", 0.0))
        batching, thread_safe = bool(conc.get("batching", False)), bool(conc.get("thread_safe", False))
        function_max: dict[str, int] = {}
        tuned = tuning_mod.for_model(self.cfg, self._tuning, entry["name"], rt.get("device", "cuda:0"))
        if tuned and (batching or thread_safe):  # local bench result; the manifest says what is possible
            cmax, per_extra = int(tuned.get("max", cmax)), float(tuned.get("vram_per_extra_gb", per_extra))
            function_max = {f: int(v["max"]) for f, v in (tuned.get("functions") or {}).items()}
        if cmax < 1 or (cmax > 1 and (per_extra <= 0 or not (batching or thread_safe))):
            raise ValueError("invalid concurrency block (max>1 needs vram_per_extra_gb>0 and batching or thread_safe)")
        entry_file = rt.get("entry", "adapter.py")
        if not (folder / entry_file).is_file():
            raise ValueError(f"entry {entry_file} missing")
        weights = (man.get("weights") or {}).get("path", "")
        if weights and not Path(weights).is_absolute():
            weights = str((self.cfg.weights_root / weights).resolve())
        # tuning is deliberately not hashed: re-tuning must not reset measured needs
        h = hashlib.sha256(mpath.read_bytes() + (folder / entry_file).read_bytes()
                           + entry["name"].encode()).hexdigest()[:16]
        return ModelDef(
            name=entry["name"], tags=tags, folder=folder, description=man.get("description", ""),
            functions=functions, entry=entry_file, device=rt.get("device", "cuda:0"),
            max_concurrency=cmax, batching=batching, thread_safe=thread_safe, vram_per_extra_gb=per_extra,
            mixed_functions=bool(conc.get("mixed_functions", False)), function_max=function_max,
            offload=man.get("offload", "partial"),
            compile_eligible=(man.get("compile") or {}).get("eligible", True) is not False,
            runaway=duration((man.get("limits") or {}).get("runaway"), self.cfg.runaway),
            weights_path=weights, plugin_version=h, raw_manifest=man,
            fingerprint=hashlib.sha256(h.encode() + (folder / "env.lock").read_bytes() + weights.encode()
                                       + json.dumps([f.schema for f in functions.values()], sort_keys=True).encode()
                                       ).hexdigest()[:16])

    # ---- catalog ----------------------------------------------------------------
    def _render(self) -> str:
        used = sorted({f.capability for m in self.models.values() for f in m.functions.values()})
        lines = ["Capability contracts (base inputs → outputs):"]
        w = max((len(c) for c in used), default=10)
        for c in used:
            lines.append(f"  {c:<{w}}  {self.contracts[c].get('catalog_line', '')}")
        lines += ["", "Models:"]
        for m in self.models.values():
            lines.append(f"  {m.name}   [{_tag_line(m.tags)}]")
            if m.description:
                lines.append(f"    {m.description}")
            fw = max(len(f) for f in m.functions)
            for f in m.functions.values():
                ex = f"   extras: {', '.join(f.extras)}" if f.extras else ""
                lines.append(f"    {f.name:<{fw}}  {f.summary}{ex}")
        return "\n".join(lines)

    def structured(self) -> dict:
        return {
            "contracts": {c: {"version": v.get("version"), "catalog_line": v.get("catalog_line"), "tool": v.get("tool"),
                              "inputs": list(v["input"].get("properties", {}))}
                          for c, v in self.contracts.items()},
            "models": {m.name: {"tags": m.tags, "description": m.description,
                                "functions": {f.name: {"capability": f.capability, "summary": f.summary,
                                                       "extras": f.extras} for f in m.functions.values()}}
                       for m in self.models.values()},
            "invalid": self.invalid,
        }


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _tag_line(tags: list[str]) -> str:
    def vals(prefix):
        return [t.split(":", 1)[1] for t in tags if t.startswith(prefix)]
    domains = sorted({c.split(".")[0] for c in vals("cap:")})
    parts = [" ".join(domains), " ".join(vals("lang:")), " ".join(vals("speed:")), " ".join(vals("quality:"))]
    return " · ".join(p for p in parts if p)
