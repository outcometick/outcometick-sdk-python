"""Select archive files, settlement streams and read order for a run.

Keep in sync with api/lib/backtest-datasets.mjs. Equal-timestamp events use
read order as a tie-breaker, so file ordering affects results.
"""

from __future__ import annotations

import datetime as _dt
import math

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
    """restingPolicyFor in api/lib/backtest-datasets.mjs, with matching behavior."""
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


# Execution options. Keep validation and messages in sync with
# api/lib/backtest-datasets.mjs.

ASSUMED_ORDER_RULES = {"tick": 0.001, "min_size": 5, "size_step": 0.01}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float("inf"), float("-inf"))


def _json(v) -> str:
    import json
    try:
        return json.dumps(v, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(v)


def normalize_fok(value) -> bool:
    if value is None or value is False:
        return False
    if value is True:
        return True
    raise ManifestError(f"fok must be true or false, got {_json(value)}")


def normalize_order_rules(value):
    if value is None:
        return None
    if value in ("venue", "assumed"):
        return value
    raise ManifestError(f'order_rules must be "venue" or "assumed", got {_json(value)}')


def normalize_maker_rebate(value):
    if value is None:
        return None
    if not _num(value) or value < 0 or value > 1:
        raise ManifestError(f"maker_rebate must be a number between 0 and 1, got {_json(value)}")
    return value


def normalize_quote_guard(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ManifestError("quote_guard must be an object")
    for k in value:
        if k not in ("max_age_ms", "ask_sum", "include_exits"):
            raise ManifestError(f"quote_guard has an unknown key {_json(k)}")
    out = {}
    if value.get("max_age_ms") is not None:
        ms = value["max_age_ms"]
        if not _num(ms) or float(ms) != int(ms) or ms < 0 or ms > 86_400_000:
            raise ManifestError("quote_guard max_age_ms must be a whole number of milliseconds between 0 and 86400000")
        out["max_age_ms"] = ms
    if value.get("ask_sum") is not None:
        r = value["ask_sum"]
        if (not isinstance(r, list) or len(r) != 2 or not _num(r[0]) or not _num(r[1])
                or r[0] < 0 or r[0] > r[1] or r[1] > 2):
            raise ManifestError("quote_guard ask_sum must be [lo, hi] with 0 <= lo <= hi <= 2")
        out["ask_sum"] = [r[0], r[1]]
    if not out:
        raise ManifestError("quote_guard needs max_age_ms or ask_sum")
    if value.get("include_exits") is not None:
        if not isinstance(value["include_exits"], bool):
            raise ManifestError("quote_guard include_exits must be true or false")
        if value["include_exits"]:
            out["include_exits"] = True
    return out


def normalize_mirror_depth(value):
    if value is None:
        return None
    if value == "shared":
        return value
    raise ManifestError(f'mirror_depth must be "shared", got {_json(value)}')


def normalize_feed_latency(value):
    """normalizeFeedLatency in api/lib/backtest-datasets.mjs."""
    if value is None or (_num(value) and value == 0):
        return None
    max_ms = CONTRACT["max_latency_ms"]
    if not _num(value) or float(value) != int(value) or value < 0 or value > max_ms:
        raise ManifestError(f"feed_latency must be a whole number of milliseconds between 0 and {max_ms}, got {_json(value)}")
    return int(value)


RECORD_MS_MIN = 1000
RECORD_MS_MAX = 3_600_000


def normalize_record_ms(value):
    """normalizeRecordMs in api/lib/backtest-datasets.mjs."""
    if value is None:
        return None
    if not _num(value) or not math.isfinite(value) or value != int(value) or not RECORD_MS_MIN <= value <= RECORD_MS_MAX:
        raise ManifestError(f"record_ms must be a whole number of milliseconds between {RECORD_MS_MIN} and "
                            f"{RECORD_MS_MAX}, got {_json(value)}")
    return int(value)


QUEUE_PROB_FUNCS = ("power", "power2", "power3", "log", "log2")


def normalize_queue_model(value):
    """normalizeQueueModel in api/lib/backtest-datasets.mjs."""
    if value is None:
        return None
    bad = ManifestError('queue_model must be {"prob": "power"|"power2"|"power3", "n": 1..10} or '
                        '{"prob": "log"|"log2"}, got ' + _json(value))
    if not isinstance(value, dict):
        raise bad
    prob = value.get("prob")
    if prob not in QUEUE_PROB_FUNCS:
        raise bad
    if prob.startswith("power"):
        if any(k not in ("prob", "n") for k in value):
            raise bad
        n = value.get("n")
        # A whole number by value, as JS's Number.isInteger: 2.0 from a JSON file is 2.
        if not _num(n) or not math.isfinite(n) or n != int(n) or not 1 <= n <= 10:
            raise bad
        return {"prob": prob, "n": int(n)}
    if any(k != "prob" for k in value):
        raise bad
    return {"prob": prob, "n": None}


def engine_policy_for(manifest, venue, data="archive"):
    """enginePolicyFor in api/lib/backtest-datasets.mjs."""
    fok = normalize_fok(manifest.get("fok"))
    rules = normalize_order_rules(manifest.get("order_rules"))
    rebate = normalize_maker_rebate(manifest.get("maker_rebate"))
    guard = normalize_quote_guard(manifest.get("quote_guard"))
    mirror = normalize_mirror_depth(manifest.get("mirror_depth"))
    feed_latency = normalize_feed_latency(manifest.get("feed_latency"))
    queue = normalize_queue_model(manifest.get("queue_model"))
    record_ms = normalize_record_ms(manifest.get("record_ms"))
    timers = "on_timer" in (manifest.get("hooks") or [])
    is_global = timers or feed_latency is not None
    if not fok and rules is None and rebate is None and guard is None and mirror is None and not is_global \
            and queue is None and record_ms is None:
        return None
    order_rules = None
    if rules == "venue":
        if data != "generic":
            raise ManifestError('order_rules "venue" needs each market\'s tick and minimum size, which archive markets '
                                'do not carry; use "assumed" (tick 0.001, minimum 5 shares — measured on the archive, '
                                'not read per market)')
        order_rules = {"mode": "venue"}
    elif rules == "assumed":
        if data == "generic":
            raise ManifestError('order_rules "assumed" is for archive markets; your own data carries its tick and '
                                'minimum — use "venue"')
        if venue != "polymarket":
            raise ManifestError(f'order_rules "assumed" is Polymarket\'s; the {venue} archive publishes no order rules')
        order_rules = {"mode": "assumed", **ASSUMED_ORDER_RULES}
    if rebate is not None and manifest.get("fee_bps") is not None:
        raise ManifestError("maker_rebate is a share of the venue's taker fee; it cannot be combined with a flat fee_bps")
    if mirror is not None and data == "archive" and venue == "polymarket":
        raise ManifestError('mirror_depth "shared" is for single-book markets (Predict.fun); Polymarket publishes two '
                            'separate books')
    if record_ms is not None and (manifest.get("mode") or "market") != "market":
        raise ManifestError('record_ms needs mode "market": in a session one portfolio is shared by every market, and '
                            'this recorder samples one market at a time')
    if queue is not None and data == "archive" and venue != "polymarket":
        raise ManifestError(f"queue_model is for resting (gtc) orders, which need a trade stream; the {venue} archive "
                            "has none")
    return {
        "fok": fok,
        "order_rules": order_rules,
        "maker_rebate": rebate,
        "quote_guard": guard,
        "mirror_depth": mirror,
        "mirror_markets": None if mirror is None else ("all" if data == "archive" else "single_mirrored"),
        **({"scheduler": "global", "timers": timers, "feed_latency": feed_latency or 0} if is_global else {}),
        **({"queue_prob": queue} if queue is not None else {}),
        **({"record_ms": record_ms} if record_ms is not None else {}),
    }
