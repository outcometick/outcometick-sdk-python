"""What a run reads from the archive.

A port of the parts of api/lib/backtest-datasets.mjs a local run needs:
which archive datasets a manifest's datasets expand to, which files belong to
a run, which settlement files the day's markets need, and the order files are
read in. The order matters: events stamped the same millisecond fall back to
read order, so a different order is a different report.
"""

from __future__ import annotations

import datetime as _dt

from ._contract import CONTRACT
from .taxonomy import classify_path

ARCHIVE_DATASETS: dict = CONTRACT["archive_datasets"]
ALWAYS_FED: tuple = tuple(CONTRACT["always_fed"])
CAPTURE_WINDOWS: dict = CONTRACT["capture_windows"]
DERIVED_DATASETS: dict = CONTRACT["derived_datasets"]
DEGRADING_DATASETS: tuple = tuple(CONTRACT["degrading_datasets"])
MARKET_INTERVALS: tuple = tuple(CONTRACT["market_intervals"])
DEFAULT_INTERVALS: tuple = tuple(CONTRACT["default_intervals"])

_SETTLEMENT_STREAMS = ("prices", "twap30s", "twap60s")


class ManifestError(ValueError):
    """The run asked for something the backtest cannot do (E_MANIFEST)."""


def resolve_settlement_stream(market) -> str | None:
    """Which captured stream a market settled on, from its own config, or None."""
    if not isinstance(market, dict):
        return None
    raw = market.get("raw")
    cfg = raw.get("cryptoMarketConfig") if isinstance(raw, dict) else None
    if cfg is None:
        cfg = market.get("cryptoMarketConfig")
    if not isinstance(cfg, dict):
        return None
    if "twapLookbackSeconds" not in cfg:
        return None
    lookback = cfg["twapLookbackSeconds"]
    # `=== 30` in the JS: a number, never a bool or a string.
    if lookback is None or (_is_number(lookback) and lookback == 0):
        return "prices"
    if _is_number(lookback) and lookback == 30:
        return "twap30s"
    if _is_number(lookback) and lookback == 60:
        return "twap60s"
    return None


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_captured(venue: str, dataset: str, day: str) -> bool:
    w = (CAPTURE_WINDOWS.get(venue) or {}).get(dataset)
    if not w:
        return False
    if day < w["from"]:
        return False
    if w.get("to") and day > w["to"]:
        return False
    return True


def _shift_day(day: str, delta: int) -> str:
    d = _dt.date.fromisoformat(day) + _dt.timedelta(days=delta)
    return d.isoformat()


def uncaptured_range(venue: str, dataset: str, frm: str, to: str):
    w = (CAPTURE_WINDOWS.get(venue) or {}).get(dataset)
    if not w:
        return {"from": frm, "to": to}
    bad_from = frm if frm < w["from"] else None
    bad_to = (to if to < w["from"] else _shift_day(w["from"], -1)) if bad_from else None
    if bad_from:
        return {"from": bad_from, "to": bad_to}
    if w.get("to") and to > w["to"]:
        return {"from": _shift_day(w["to"], 1), "to": to}
    return None


def archive_datasets_for(*, datasets, venue: str, frm: str, to: str) -> list:
    wanted: list = []

    def add(x):
        if x not in wanted:
            wanted.append(x)

    for ds in list(datasets or []) + list(ALWAYS_FED):
        if ds == "settlement":
            for s in _SETTLEMENT_STREAMS:
                if not uncaptured_range(venue, s, frm, to) or is_captured(venue, s, to):
                    add(s)
            continue
        derived = DERIVED_DATASETS.get(ds)
        add(derived["from"] if derived else ds)
    out: set = set()
    mapping = ARCHIVE_DATASETS.get(venue) or {}
    for w in wanted:
        for a in mapping.get(w) or []:
            out.add(a)
    return sorted(out)


def archive_datasets_for_day(*, datasets, venue: str, day: str, frm: str, to: str) -> list:
    usable = [d for d in (datasets or []) if d not in DEGRADING_DATASETS or is_captured(venue, d, day)]
    return archive_datasets_for(datasets=usable, venue=venue, frm=frm, to=to)


def settlement_paths_for(markets, paths, *, venue: str, assets, already=()) -> list:
    need = {m.get("stream") for m in markets if m and m.get("stream")}
    if not need:
        return []
    have = set(already)
    in_scope = {str(a).upper() for a in (assets or [])}
    out = []
    for p in paths:
        if p in have:
            continue
        c = classify_path(p)
        if c["venue"] != venue or c["dataset"] not in need:
            continue
        if c["asset"] and in_scope and str(c["asset"]).upper() not in in_scope:
            continue
        out.append(p)
    return _js_sorted(out)


def ordered_feed(paths) -> list:
    return _js_sorted(paths)


def _js_sorted(strings) -> list:
    """Array.prototype.sort() on strings: by UTF-16 code units, not code points."""
    return sorted(strings, key=lambda s: s.encode("utf-16-be"))


def file_matches_run(file_path: str, *, venue: str, assets, archive_datasets, intervals) -> bool:
    meta = classify_path(file_path)
    if meta["venue"] != venue:
        return False
    if meta["dataset"] not in archive_datasets:
        return False
    if meta["asset"] and assets and meta["asset"] not in assets:
        return False
    if meta["interval"] and intervals and meta["interval"] not in intervals:
        return False
    return True


def resting_policy_for(manifest, venue) -> dict:
    """restingPolicyFor in api/lib/backtest-datasets.mjs, word for word."""
    def refuse(refusal):
        return {"allowed": False, "refusal": refusal, "cancelLatencyMs": 0}
    if venue != "polymarket":
        return refuse("resting (gtc) orders need a trade stream to advance the queue; "
                      f"the {venue} archive has order-book snapshots only, so only ioc orders run there")
    datasets = manifest.get("datasets") or []
    missing = [d for d in ("book", "trades") if d not in datasets]
    if missing:
        names = " and ".join(f'"{d}"' for d in missing)
        plural = len(missing) > 1
        return refuse(f"resting (gtc) orders need the {names} dataset{'s' if plural else ''} — "
                      f"add {'them' if plural else 'it'} to datasets in the manifest")
    cancel = manifest.get("cancel_latency")
    if cancel is None:
        cancel = manifest.get("latency")
    return {"allowed": True, "refusal": None, "cancelLatencyMs": cancel if cancel is not None else 0}


def normalize_intervals(lst) -> list:
    if lst is None:
        return list(DEFAULT_INTERVALS)
    if not isinstance(lst, (list, tuple)) or len(lst) == 0:
        raise ManifestError(f"intervals must be a non-empty list; known: {', '.join(MARKET_INTERVALS)}")
    out = []
    for raw in lst:
        iv = str("" if raw is None else raw).strip()
        if iv not in MARKET_INTERVALS:
            raise ManifestError(f"unknown interval {iv!r}; known: {', '.join(MARKET_INTERVALS)}")
        if iv not in out:
            out.append(iv)
    return _js_sorted(out)
