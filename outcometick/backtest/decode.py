"""Archived rows -> replayable events.

A port of runner/events.mjs (the decode half). The hosted queue and `ot run`
decode the archive with that module; this is the same decoder for a run that
has no Node. Every rule below is the JS rule, in the JS order, and the reasons
for each are written down there -- read events.mjs before changing anything
here, and change both or neither. client/backtest/equivalence.test.mjs decodes
the public samples with both and compares every event.
"""

from __future__ import annotations

import math

from ._contract import CONTRACT
from ._js import js_string, js_to_fixed, num
from .datasets import resolve_settlement_stream
from .taxonomy import classify_path

OUTCOME_TIE = "TIE"
OUTCOMES = ("UP", "DOWN", OUTCOME_TIE)

_BOOK_CAPTURE = CONTRACT["book_capture"]


def _coalesce(*vals):
    """`a ?? b ?? c`: the first value that is not None."""
    for v in vals:
        if v is not None:
            return v
    return None


def _get(obj, key):
    """`obj?.key` for a JSON value: None unless obj is an object with the key."""
    return obj.get(key) if isinstance(obj, dict) else None


def _interval_label(seconds):
    n = num(seconds)
    if n is None or n <= 0:
        return None
    for unit, suffix in ((86400, "d"), (3600, "h"), (60, "m")):
        if n % unit == 0:
            return f"{js_string(n / unit)}{suffix}"
    return f"{js_string(n)}s"


def _side_of_token(market, asset_id):
    ids = _get(market, "token_ids")
    if not isinstance(ids, list) or asset_id is None:
        return None
    key = js_string(asset_id)
    try:
        i = ids.index(key)
    except ValueError:
        return None
    if i == 0:
        return "UP"
    if i == 1:
        return "DOWN"
    return None


def _polymarket_outcome(row):
    if _get(row, "resolved") is not True:
        return None
    p = row.get("outcome_prices")
    if not isinstance(p, list) or len(p) != 2:
        return None
    up, down = (js_string(x) for x in p)
    if up == "1" and down == "0":
        return "UP"
    if up == "0" and down == "1":
        return "DOWN"
    return None


FEE_MODEL_PM = "polymarket-taker-v1"


def _fee_of(row):
    """`feeOf` in runner/events.mjs: the normalised taker schedule, {model:'none'}
    for a known zero, or None when unknown."""
    raw = _get(row, "raw")
    if not isinstance(raw, dict):
        return None
    if raw.get("feesEnabled") is False:
        return {"model": "none"}
    if raw.get("feesEnabled") is not True:
        return None
    sched = raw.get("feeSchedule")
    if not isinstance(sched, dict):
        return None
    rate = sched.get("rate")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isfinite(rate) or rate < 0 or rate > 1:
        return None
    exponent = sched.get("exponent")
    # `s.exponent !== 1`: the number 1, never a bool or a string.
    if not isinstance(exponent, (int, float)) or isinstance(exponent, bool) or exponent != 1:
        return None
    return {"model": FEE_MODEL_PM, "rate": rate}


def _polymarket_record(row):
    start, end = num(row.get("start_sec")), num(row.get("end_sec"))
    strike_raw = num(row.get("strike_value"))
    token_ids = row.get("token_ids")
    return {
        "market_id": js_string(_coalesce(row.get("condition_id"), row.get("slug"), "")),
        "slug": _coalesce(row.get("slug"), None),
        "asset": js_string(row["asset"]).upper() if row.get("asset") else None,
        "interval": _interval_label(row.get("interval_sec")),
        "strike": None if strike_raw is None else strike_raw / 1e18,
        "outcome": _polymarket_outcome(row),
        "open_ts_ms": None if start is None else start * 1000,
        "close_ts_ms": None if end is None else end * 1000,
        "stream": resolve_settlement_stream(row),
        "token_ids": [js_string(t) for t in token_ids] if isinstance(token_ids, list) else [],
        "fee": _fee_of(row),
        "raw": row,
    }


def _predict_record(row):
    start_s, end_s = num(row.get("start_sec")), num(row.get("end_sec"))
    start, end = num(row.get("start_price")), num(row.get("end_price"))
    outcome = None
    if start is not None and end is not None:
        outcome = "UP" if end > start else "DOWN" if end < start else OUTCOME_TIE
    feed = num(row.get("price_feed_id"))
    label = row.get("interval_label")
    return {
        "market_id": js_string(_coalesce(row.get("market_id"), row.get("condition_id"), "")),
        "slug": _coalesce(row.get("category_slug"), None),
        "asset": js_string(row["asset"]).upper() if row.get("asset") else None,
        "interval": js_string(label) if label else None,
        "strike": start,
        "outcome": outcome,
        "open_ts_ms": None if start_s is None else start_s * 1000,
        "close_ts_ms": None if end_s is None else end_s * 1000,
        "stream": None if feed is None else "prices",
        "price_feed_id": feed,
        "token_ids": [],
        "raw": row,
    }


def index_markets(rows, *, venue="polymarket"):
    by_id = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        rec = _predict_record(row) if venue == "predict" else _polymarket_record(row)
        if not rec["market_id"]:
            continue
        by_id[rec["market_id"]] = rec
    return by_id


def build_slug_index(markets):
    by_slug = {}
    for m in markets.values():
        if m.get("slug"):
            by_slug[js_string(m["slug"])] = m
    return by_slug


def _market_for_row(row, markets, by_slug):
    direct = _coalesce(row.get("market_id"), row.get("marketId"))
    if direct is not None and js_string(direct) in markets:
        return markets[js_string(direct)]
    slug = _coalesce(row.get("slug"), row.get("category_slug"))
    if slug is not None and js_string(slug) in by_slug:
        return by_slug[js_string(slug)]
    return None


def _ladder(levels):
    if not isinstance(levels, list):
        return []
    out = []
    for lv in levels:
        if isinstance(lv, list):
            px = num(lv[0] if len(lv) > 0 else None)
            size = num(lv[1] if len(lv) > 1 else None)
        else:
            px, size = num(_get(lv, "price")), num(_get(lv, "size"))
        if px is None or size is None:
            continue
        if px < 0 or px > 1:
            continue
        if size <= 0:
            continue
        out.append([px, size])
    return out


def _mirror(levels):
    out = []
    for px, size in levels:
        m = float(js_to_fixed(1 - px, 10))
        if not (0 <= m <= 1):
            continue
        out.append([m + 0.0, size])
    return out


# --- book cadence ---------------------------------------------------------

def book_throttle_ms(*, venue, asset, frm, to):
    """`bookThrottleMs` in api/lib/backtest-contract.mjs."""
    history = _BOOK_CAPTURE.get(venue)
    if not history:
        return 0
    a = js_string("" if asset is None else asset).upper()
    coarsest = 0
    for i, entry in enumerate(history):
        nxt = history[i + 1] if i + 1 < len(history) else None
        if to < entry["from"]:
            continue
        if nxt and frm >= nxt["from"]:
            continue
        per = entry["perAsset"]
        if a == "*":
            ms = max([entry["defaultMs"], *per.values()])
        else:
            ms = per.get(a, entry["defaultMs"])
        if ms > coarsest:
            coarsest = ms
    return coarsest


_THROTTLED = {"polymarket": "price_change", "predict": "orderbook"}


class BookThrottle:
    """`makeBookThrottle`: first row per market per window, per-asset window."""

    def __init__(self, *, venue, assets=(), frm, to):
        self.dataset = _THROTTLED.get(venue)
        self.per_asset = {js_string(a).upper(): book_throttle_ms(venue=venue, asset=a, frm=frm, to=to) for a in assets}
        self.fallback = book_throttle_ms(venue=venue, asset="*", frm=frm, to=to)
        self._last = {}

    def keep(self, ds, market_id, ts, asset):
        if ds != self.dataset or self.fallback == 0:
            return True
        ms = self.per_asset.get(js_string("" if asset is None else asset).upper(), self.fallback)
        if ms == 0:
            return True
        key = js_string(market_id)
        prev = self._last.get(key)
        if prev is not None and ts - prev < ms:
            return False
        self._last[key] = ts
        return True


# --- rows -> events -------------------------------------------------------

def events_from_row(file_path, row, markets, by_slug=None, throttle=None, meta=None):
    meta = meta or classify_path(file_path)
    slug_index = by_slug if by_slug is not None else build_slug_index(markets)
    ds = meta["dataset"]

    if ds in ("prices", "twap30s", "twap60s"):
        pub_sec = num(row.get("publish_time"))
        ts = num(_coalesce(row.get("feed_ts_ms"), row.get("ts_ms"), row.get("timestamp_ms"), row.get("event_ts_ms")))
        if ts is None:
            ts = None if pub_sec is None else pub_sec * 1000
        if ts is None:
            return []
        value = num(_coalesce(row.get("value"), row.get("price"), row.get("answer")))
        if value is None:
            return []
        server_sec = num(row.get("server_ts"))
        server_ms = num(row.get("server_ts_ms"))
        if server_ms is None:
            server_ms = None if server_sec is None else server_sec * 1000
        recv_ms = num(_coalesce(row.get("recv_ts_ms"), row.get("recv_ms")))
        out = []
        for mid, m in markets.items():
            if m["stream"] != ds:
                continue
            if m["asset"] and meta["asset"] and js_string(m["asset"]) != js_string(meta["asset"]):
                continue
            if m.get("price_feed_id") is not None:
                row_feed = num(row.get("price_feed_id"))
                if row_feed is None or row_feed != m["price_feed_id"]:
                    continue
            if m["open_ts_ms"] is not None and ts < m["open_ts_ms"]:
                continue
            if m["close_ts_ms"] is not None and ts > m["close_ts_ms"]:
                continue
            out.append((mid, {
                "kind": "tick",
                "ts_ms": ts,
                "market_id": mid,
                "value": value,
                "source": ds,
                "server_ts_ms": server_ms,
                "recv_ts_ms": recv_ms,
            }))
        return out

    market = _market_for_row(row, markets, slug_index)
    if not market:
        return []
    mid = market["market_id"]
    ts = num(_coalesce(row.get("event_ts_ms"), row.get("update_ts_ms"), row.get("ts_ms"), row.get("timestamp_ms")))
    if ts is None:
        return []
    if throttle is not None and not throttle.keep(ds, mid, ts, market["asset"]):
        return []
    payload = _coalesce(row.get("payload"), row)

    if ds == "orderbook":
        asks, bids = _ladder(_get(payload, "asks")), _ladder(_get(payload, "bids"))
        return [(mid, {
            "kind": "book", "ts_ms": ts, "snapshot": True,
            "levels": {"UP": {"asks": asks, "bids": bids},
                       "DOWN": {"asks": _mirror(bids), "bids": _mirror(asks)}},
        })]

    if ds == "book":
        side = _side_of_token(market, _coalesce(row.get("asset_id"), _get(payload, "asset_id")))
        if not side:
            return []
        return [(mid, {
            "kind": "book", "ts_ms": ts, "snapshot": True, "side": side,
            "levels": {side: {"asks": _ladder(_get(payload, "asks")), "bids": _ladder(_get(payload, "bids"))}},
        })]

    if ds == "best_bid_ask":
        side = _side_of_token(market, _coalesce(row.get("asset_id"), _get(payload, "asset_id")))
        if not side:
            return []
        bid, ask = num(_get(payload, "best_bid")), num(_get(payload, "best_ask"))
        if bid is None or ask is None:
            return []
        if bid < 0 or bid > 1 or ask < 0 or ask > 1:
            return []
        if bid > ask:
            return []
        return [(mid, {"kind": "book", "ts_ms": ts, "snapshot": False, "bbo": True,
                       "side": side, "bid": bid, "ask": ask})]

    if ds == "price_change":
        changes = _get(payload, "price_changes")
        if not isinstance(changes, list):
            changes = []
        out = []
        for ch in changes:
            if ch is None:
                # `ch.asset_id` on null throws in the JS, ending the run.
                raise TypeError("Cannot read properties of null (reading 'asset_id')")
            side = _side_of_token(market, _get(ch, "asset_id"))
            if not side:
                continue
            px, size = num(_get(ch, "price")), num(_get(ch, "size"))
            if px is None or size is None:
                continue
            if px < 0 or px > 1 or size < 0:
                continue
            out.append((mid, {
                "kind": "book", "ts_ms": ts, "snapshot": False, "side": side,
                "ladder": "asks" if js_string(_get(ch, "side")).upper() == "SELL" else "bids",
                "px": px, "size": size,
            }))
        return out

    if ds == "last_trade_price":
        side = _side_of_token(market, _coalesce(row.get("asset_id"), _get(payload, "asset_id")))
        if not side:
            return []
        px, size = num(_get(payload, "price")), num(_get(payload, "size"))
        if px is None or px < 0 or px > 1:
            return []
        if size is None or size <= 0:
            return []
        return [(mid, {
            "kind": "trade", "ts_ms": ts, "market_id": mid, "px": px, "size": size, "side": side,
            "taker": "SELL" if js_string(_get(payload, "side")).upper() == "SELL" else "BUY",
        })]

    return []


# --- per market -----------------------------------------------------------

_RANK = {"book": 0, "trade": 1, "tick": 2, "ref": 3, "ext": 4}


def market_unusable(market, in_window):
    if not market:
        return "no market metadata"
    if not market.get("asset"):
        return "market has no asset"
    if market.get("stream") is None:
        return "settlement stream could not be resolved"
    if market.get("outcome") not in OUTCOMES:
        return "outcome could not be read"
    if not in_window or not any(e.get("kind") != "trade" for e in in_window):
        return "no events inside the market window"
    if not any(e.get("kind") == "tick" for e in in_window):
        return f"no settlement ticks on {market['stream']}"
    return None


def finalise_market(events, market, book_factory):
    """Order one market's events, cut them to its window, find the opening quotes.

    `book_factory` builds the engine's own Book (otengine.Book), so the opening
    quote is computed by the same book the strategy trades on.
    """
    events.sort(key=lambda e: (
        e["ts_ms"],
        _RANK.get(e.get("kind"), 9),
        1 if e.get("kind") == "book" and e.get("snapshot") is not True else 0,
    ))
    open_ms, close_ms = market.get("open_ts_ms"), market.get("close_ts_ms")
    in_window = [e for e in events if (open_ms is None or e["ts_ms"] >= open_ms)
                 and (close_ms is None or e["ts_ms"] <= close_ms)]
    book = book_factory()
    up_px = down_px = None
    for ev in in_window:
        if ev.get("kind") != "book":
            continue
        if ev.get("bbo"):
            book.bbo(ev["ts_ms"], ev.get("side"), ev.get("bid"), ev.get("ask"))
        elif ev.get("snapshot"):
            book.snapshot(ev["ts_ms"], ev.get("levels"))
        elif ev.get("side") and ev.get("ladder"):
            book.delta(ev["ts_ms"], ev["side"], ev["ladder"], ev.get("px"), ev.get("size"))
        up, down = book.best("UP"), book.best("DOWN")
        if up is not None and down is not None:
            up_px, down_px = up, down
            break
    return {"events": in_window, "up_px": up_px, "down_px": down_px}


def sort_markets_for_replay(markets, *, mode="market"):
    def utf16(s):
        return js_string(s).encode("utf-16-be")

    def key_id(m):
        return utf16(_coalesce((m.get("market") or {}).get("market_id"), ""))

    def key_time(m):
        v = (m.get("market") or {}).get("open_ts_ms")
        return 0 if v is None else v

    if mode == "session":
        markets.sort(key=lambda m: (key_time(m), key_id(m)))
        return markets
    markets.sort(key=lambda m: (utf16(_coalesce((m.get("market") or {}).get("asset"), "")), key_time(m), key_id(m)))
    return markets


def _mkey(m, field, default):
    return js_string(_coalesce((m.get("market") or {}).get(field), default))


def count_market_days(markets):
    return len({f"{_mkey(m, 'asset', 'unknown')}|{js_string(m.get('day'))}|{_mkey(m, 'interval', 'none')}"
                for m in markets})


def count_streams(items):
    out = {}
    for m in items:
        s = _coalesce((m or {}).get("stream"), "unknown")
        out[s] = out.get(s, 0) + 1
    return out


def build_coverage(*, market_days_requested=None, market_days_scanned, markets_reported_by_runner=None,
                   missing=(), reference_declared=(), reference_missing=(), streams=None, dropped_rows=0,
                   unreconciled_rows=0, bbo_declared=False, bbo_days=(), bbo_missing_days=(),
                   bbo_partial_days=(), local=False, source=None, extra=None):
    out = {
        "market_days_requested": market_days_requested,
        "market_days_scanned": market_days_scanned,
        "markets_reported_by_runner": markets_reported_by_runner,
        "missing": list(missing),
        "reference_declared": list(reference_declared),
        "reference_missing": list(reference_missing),
        "streams": streams or {},
        "dropped_rows": dropped_rows,
        "unreconciled_rows": unreconciled_rows,
        "bbo_declared": bbo_declared,
        "bbo_days": list(bbo_days),
        "bbo_missing_days": list(bbo_missing_days),
        "bbo_partial_days": list(bbo_partial_days),
    }
    if extra:
        out.update(extra)
    if local:
        out["local"] = True
        out["source"] = source
    return out


def bbo_coverage(*, venue, datasets, markets, applied=None):
    if "bbo" not in (datasets or []):
        return {"bbo_declared": False, "bbo_days": [], "bbo_missing_days": [], "bbo_partial_days": []}
    if applied is None:
        raise ValueError("bbo_coverage: `applied` is required when the manifest declares bbo")
    by_day = {}
    for m in markets or []:
        day = (m or {}).get("day")
        if not day:
            continue
        by_day.setdefault(day, [])
        key = f"{_mkey(m, 'asset', 'unknown')}|{_mkey(m, 'interval', 'none')}"
        if key not in by_day[day]:
            by_day[day].append(key)
    full, none, partial = [], [], []
    for day in sorted(by_day):
        scanned = by_day[day]
        got = applied.get(day, set())
        without = sorted(k for k in scanned if k not in got)
        if not without:
            full.append(day)
        elif len(without) == len(scanned):
            none.append(day)
        else:
            partial.append({"day": day, "without": without})
    return {"bbo_declared": True, "bbo_days": full, "bbo_missing_days": none, "bbo_partial_days": partial}


def parse_row(line, *, is_csv, header, loads):
    if not is_csv:
        try:
            return loads(line)
        except ValueError:
            return None
    if not header:
        return None
    cells = line.split(",")
    return {h: (cells[i] if i < len(cells) else None) for i, h in enumerate(header)}


__all__ = [
    "OUTCOMES", "OUTCOME_TIE", "BookThrottle", "book_throttle_ms", "build_coverage", "bbo_coverage",
    "build_slug_index", "count_market_days", "count_streams", "events_from_row", "finalise_market",
    "index_markets", "market_unusable", "parse_row", "sort_markets_for_replay",
]
