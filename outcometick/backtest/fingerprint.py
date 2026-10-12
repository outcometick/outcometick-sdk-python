"""Compute a run's pre-registration SHA-256 fingerprint.

Covers strategy code, manifest with defaults, params, execution options, data
scope and seed. Publish it before viewing results, then compare with the report.

Keep in sync with api/lib/backtest-fingerprint.mjs: canonical JSON uses sorted
keys and JavaScript number formatting; file selection matches readSubmission.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re

from ._contract import CONTRACT
from ._js import js_string
from .datasets import engine_policy_for, normalize_intervals

FINGERPRINT_VERSION = 1
_MANIFEST = "outcometick.json"
_READABLE = re.compile(r"\.(py|mjs|js|json|csv)$")


def _js_key(s: str):
    """JavaScript compares strings by UTF-16 code unit."""
    return s.encode("utf-16-be")


def canonical_json(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return js_string(v) if math.isfinite(v) else "null"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(canonical_json(x) for x in v) + "]"
    if not isinstance(v, dict):
        # An in-process params value JSON cannot hold: its text, never a crash.
        return json.dumps(str(v), ensure_ascii=False)
    keys = sorted(v.keys(), key=_js_key)
    return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + canonical_json(v[k]) for k in keys) + "}"


def read_submission(path: str) -> list:
    """`ot`'s readSubmission: the top level and one directory down, skipping
    dot entries, node_modules and __pycache__; bytes decoded as UTF-8 with no
    newline translation."""
    out = []

    def walk(rel: str, depth: int):
        base = os.path.join(path, rel) if rel else path
        for e in sorted(os.scandir(base), key=lambda d: d.name):
            name = f"{rel}/{e.name}" if rel else e.name
            if e.name.startswith(".") or e.name in ("node_modules", "__pycache__"):
                continue
            if e.is_dir():
                if depth > 0:
                    walk(name, depth - 1)
                continue
            if not _READABLE.search(e.name):
                continue
            with open(e.path, "rb") as fh:
                out.append({"name": name, "content": fh.read().decode("utf-8", "replace")})
    walk("", 1)
    return out


def code_digest(files):
    # Every submitted file, outcometick.json included, byte for byte; see codeDigest.
    code = sorted(files or [], key=lambda f: _js_key(f["name"]))
    if not code:
        return None
    h = hashlib.sha256()
    for f in code:
        h.update(f["name"].encode("utf-8"))
        h.update(b"\x00")
        h.update((f.get("content") or "").encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest()


def inputs_digest(lines):
    """inputsDigest in backtest-fingerprint.mjs."""
    lst = sorted(set(lines or []), key=_js_key)
    return hashlib.sha256("\n".join(lst).encode("utf-8")).hexdigest() if lst else None


def local_inputs(root, paths):
    out = []
    for p in dict.fromkeys(paths or []):
        with open(os.path.join(root, p), "rb") as fh:
            out.append(f"{p}:{hashlib.sha256(fh.read()).hexdigest()}")
    return out


def expand_days(frm, to):
    """expandDays in backtest-fingerprint.mjs: every UTC day from..to inclusive."""
    import datetime as _dt
    if not frm or not to:
        return []
    out = []
    d = _dt.date.fromisoformat(frm)
    end = _dt.date.fromisoformat(to)
    while d <= end and len(out) <= 4000:
        out.append(d.isoformat())
        d += _dt.timedelta(days=1)
    return out


def dataset_digest(root, markets_file, files):
    """datasetDigest in backtest-fingerprint.mjs: your own data's bytes."""
    h = hashlib.sha256()
    names = ([markets_file] if markets_file else []) + sorted(files, key=_js_key)
    for name in names:
        with open(os.path.join(root, name), "rb") as fh:
            inner = hashlib.sha256(fh.read()).hexdigest()
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(inner.encode("ascii"))
        h.update(b"\x00")
    return h.hexdigest()


def memory_dataset_digest(dataset):
    """An in-memory Dataset (Python only): its entries, frames and tables, canonically."""
    def plain(v):
        if hasattr(v, "to_dict"):
            try:
                return plain(v.to_dict(orient="split"))
            except TypeError:
                return plain(v.to_dict())
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        return str(v)
    return hashlib.sha256(canonical_json(plain({
        "entries": dataset.entries, "frames": dataset.frames, "pmdata": dataset.pmdata,
        "root": None if dataset.root is None else dataset_digest_of_dir(dataset.root),
    })).encode("utf-8")).hexdigest()


def dataset_digest_of_dir(root):
    from .generic import dataset_files, markets_file_of
    return dataset_digest(root, markets_file_of(root), dataset_files(root))


def _sorted_unique(xs):
    return sorted({js_string(x) for x in (xs or [])}, key=_js_key)


def fingerprint_input(*, manifest, code=None, scope=None, seed=1, data="archive", params=None, has_params=False,
                      series_sha=None, data_sha256=None, replay=None, fee_bps_override=None, py_exact=None):
    m = manifest or {}
    scope = scope or {}
    entry = m.get("entry")
    if isinstance(entry, dict):
        entry = f"{entry.get('file')}:{entry.get('className')}"
    elif not isinstance(entry, str):
        entry = None
    series_sha = series_sha or {}
    series = [{"name": (s or {}).get("name"), "file": (s or {}).get("file"),
               "lag_ms": _coalesce((s or {}).get("lag_ms"), 0), "sha256": series_sha.get((s or {}).get("file"))}
              for s in (m.get("series") or [])]
    series.sort(key=lambda s: _js_key(js_string(s["name"])))
    out = {} if py_exact is None else {"py_exact": py_exact}
    return {
        **out,
        "v": FINGERPRINT_VERSION,
        "code": code,
        "manifest_doc": m,
        "language": m.get("language"),
        "entry": entry,
        "hooks": _sorted_unique(m.get("hooks")),
        "datasets": _sorted_unique(m.get("datasets")),
        "intervals": None if data == "generic" else _sorted_unique(normalize_intervals(m.get("intervals"))),
        "mode": _coalesce(m.get("mode"), "market"),
        "latency": _coalesce(m.get("latency"), 0),
        "cancel_latency": m.get("cancel_latency"),
        "fee_bps": m.get("fee_bps"),
        "fee": ({"mode": "bps", "bps": fee_bps_override} if fee_bps_override is not None
                else ({"mode": "bps", "bps": m["fee_bps"]} if m.get("fee_bps") is not None else {"mode": "venue"})),
        "implementation": {"image": CONTRACT["sandbox_image_tag"], "decode": CONTRACT["decode_semantics"]},
        "params": params if has_params else _coalesce(m.get("params"), {}),
        "reference": _sorted_unique(m.get("reference")),
        "series": series,
        "engine": engine_policy_for(m, _coalesce(scope.get("venue"), "polymarket"), data),
        "data": {"kind": data, "sha256": data_sha256 if data == "generic" else None},
        "replay": {
            "book_throttle": "off" if (replay or {}).get("book_throttle") is False else "on",
            "mark_max_age_ms": (replay or {}).get("mark_max_age_ms"),
        },
        "scope": {
            "venue": scope.get("venue"),
            "assets": _sorted_unique([js_string(a).upper() for a in (scope.get("assets") or [])]),
            "days": [js_string(d) for d in (scope.get("days") or [])],
        },
        "seed": _coalesce(seed, 1),
    }


def _coalesce(v, d):
    return d if v is None else v


def run_fingerprint(**kw) -> dict:
    inp = fingerprint_input(**kw)
    return {"version": FINGERPRINT_VERSION,
            "sha256": hashlib.sha256(canonical_json(inp).encode("utf-8")).hexdigest(),
            "code_sha256": inp["code"], "data_sha256": inp["data"]["sha256"]}


def fingerprint_of_files(files, *, scope=None, seed=1, data="archive", params=None, has_params=False,
                         fee_bps_override=None, data_sha256=None, replay=None, inputs=None):
    doc = next((f for f in files or [] if f["name"] == _MANIFEST), None)
    try:
        manifest = json.loads(doc["content"]) if doc else None
    except ValueError:
        manifest = None
    if not isinstance(manifest, dict):
        return None
    by_name = {f["name"]: f for f in files or []}
    series_sha = {}
    for s in manifest.get("series") if isinstance(manifest.get("series"), list) else []:
        f = by_name.get(s.get("file")) if isinstance(s, dict) and isinstance(s.get("file"), str) else None
        if f is not None:
            series_sha[s["file"]] = hashlib.sha256((f.get("content") or "").encode("utf-8", "surrogatepass")).hexdigest()
    fp = run_fingerprint(manifest=manifest, code=code_digest(files), scope=scope, seed=seed, data=data,
                         params=params, has_params=has_params, series_sha=series_sha, data_sha256=data_sha256,
                         replay=replay, fee_bps_override=fee_bps_override)
    return {**fp, "inputs_sha256": inputs_digest(inputs) if inputs is not None else None}
