"""The backtest report.

A port of runner/engine/report.mjs and of the row validation in
runner/harness/protocol.mjs (parseTrade / parseFill / parseResult). The hosted
queue and `ot run` build the report with the JS; a pure-Python run builds it
with this, and client/backtest/equivalence.test.mjs requires the two reports to
be equal over the same archive. The reasons behind each number are written in
report.mjs -- read them there before changing anything here.

Float arithmetic is done in the JS order (left-to-right sums, the same sorts)
so the results agree to the bit, and rounding uses JS `toFixed` semantics.
"""

from __future__ import annotations

import datetime as _dt
import math

from ._js import js_math_round, js_round_fixed, js_to_fixed

OUTCOMES = ("UP", "DOWN", "TIE")

CALIBRATION_BUCKETS = (
    (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
    (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0),
)
EQUITY_MAX_POINTS = 2000
TRADES_HEAD = 20


def contract_value(side, outcome):
    if outcome == "TIE":
        return 0.5
    return 1 if outcome == side else 0


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _sum(xs) -> float:
    acc = 0
    for x in xs:
        acc = acc + x
    return acc


def _mean(xs) -> float:
    return _sum(xs) / len(xs) if xs else 0


def _stdev(xs) -> float:
    if len(xs) < 2:
        return 0
    m = _mean(xs)
    return math.sqrt(_sum([(x - m) ** 2 for x in xs]) / (len(xs) - 1))


def r2(x):
    return js_round_fixed(x, 2)


def r4(x):
    return js_round_fixed(x, 4)


def _div(a, b):
    """`a / b` with IEEE results instead of ZeroDivisionError."""
    if b == 0:
        if a == 0 or (isinstance(a, float) and math.isnan(a)):
            return math.nan
        return math.inf if a > 0 else -math.inf
    return a / b


def _collateral(t) -> float:
    entry = t.get("entry_px")
    size = t.get("size")
    return (0 if entry is None else entry) * (0 if size is None else size)


def _peak_capital(trades) -> float:
    events = []
    for t in trades:
        if t.get("opened_ms") is None or t.get("closed_ms") is None:
            continue
        amt = _collateral(t)
        if not amt > 0:
            continue
        events.append((t["opened_ms"], amt))
        events.append((t["closed_ms"], -amt))
    events.sort(key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


def _holding_ratio(trades):
    with_times = [t for t in trades if t.get("opened_ms") is not None and t.get("closed_ms") is not None]
    if not with_times:
        return None
    first = min(t["opened_ms"] for t in with_times)
    last = max(t["closed_ms"] for t in with_times)
    span = last - first
    if not span > 0:
        return None
    spans = sorted(((t["opened_ms"], t["closed_ms"]) for t in with_times), key=lambda s: s[0])
    held = 0
    s, e = spans[0]
    for a, b in spans[1:]:
        if a > e:
            held += e - s
            s, e = a, b
        elif b > e:
            e = b
    held += e - s
    return held / span


def _utc_day(ms) -> str:
    # new Date(ms).toISOString().slice(0, 10): the time value truncates towards zero.
    days = math.floor(math.trunc(ms) / 86_400_000)
    return (_dt.date(1970, 1, 1) + _dt.timedelta(days=days)).isoformat()


def _closed_key(t):
    v = t.get("closed_ms")
    return 0 if v is None else v


def equity_curve(trades):
    acc = 0
    out = []
    for t in sorted(trades, key=_closed_key):
        acc += t["pnl"]
        out.append({"ts_ms": t.get("closed_ms"), "equity": acc})
    return out


def max_drawdown(series):
    peak = 0
    worst_abs = 0
    worst_pct = None
    for v in series:
        if v > peak:
            peak = v
        decline = peak - v
        if decline > worst_abs:
            worst_abs = decline
            worst_pct = -(decline / peak) if peak > 0 else None
    return {"abs": worst_abs, "pct": worst_pct}


def worst_losing_run(pnls) -> int:
    worst = current = 0
    for p in pnls:
        if p < 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def _settled(trades):
    return [t for t in trades if t.get("how") == "settled" and t.get("entry_px") is not None and t.get("outcome")]


def edge_per_contract(trades) -> float:
    settled = _settled(trades)
    if not settled:
        return 0
    contracts = edge = 0
    for t in settled:
        edge += (contract_value(t["side"], t["outcome"]) - t["entry_px"]) * t["size"]
        contracts += t["size"]
    return edge / contracts if contracts > 0 else 0


def brier(trades):
    settled = _settled(trades)
    if not settled:
        return None
    return _mean([(t["entry_px"] - contract_value(t["side"], t["outcome"])) ** 2 for t in settled])


def metrics(trades, *, fees_paid=0, days=1):
    closed = [t for t in trades if _finite(t.get("pnl"))]
    pnls = [t["pnl"] for t in sorted(closed, key=_closed_key)]
    net = _sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    collateral = _sum([_collateral(t) for t in closed])
    peak = _peak_capital(closed)
    hold = _holding_ratio(closed)
    dd = max_drawdown([p["equity"] for p in equity_curve(closed)])

    by_day: dict = {}
    for t in closed:
        day = _utc_day(t["closed_ms"]) if t.get("closed_ms") else "unknown"
        by_day[day] = by_day.get(day, 0) + t["pnl"]
    daily = list(by_day.values())
    sd = _stdev(daily)
    sharpe = None if sd == 0 else (_mean(daily) / sd) * math.sqrt(365)
    holds = [t["closed_ms"] - t["opened_ms"] for t in closed
             if t.get("opened_ms") is not None and t.get("closed_ms") is not None]

    return {
        "net_pnl": r2(net),
        "win_rate": r4(len(wins) / len(closed)) if closed else None,
        "profit_factor": r2(_div(_sum(wins), abs(_sum(losses)))) if losses else None,
        "max_drawdown": None if dd["pct"] is None else r4(dd["pct"]),
        "max_drawdown_abs": r2(dd["abs"]),
        "sharpe": None if sharpe is None else r2(sharpe),
        "trades": len(closed),
        "markets_traded": len({t.get("market_id") for t in closed}),
        "return_on_collateral": r4(net / collateral) if collateral > 0 else None,
        "peak_capital": r2(peak),
        "return_on_peak": r4(net / peak) if peak > 0 else None,
        "holding_ratio": None if hold is None else r4(hold),
        "edge_per_contract": r4(edge_per_contract(closed)) if closed else None,
        "brier_score": r4(brier(closed)),
        "fees": r2(-abs(fees_paid)),
        "avg_hold_ms": int(js_math_round(_mean(holds))) if holds else None,
        "worst_losing_run": worst_losing_run(pnls),
        "collateral_deployed": r2(collateral),
        "market_days": days,
    }


def downsample_equity(points, mx=EQUITY_MAX_POINTS):
    if len(points) <= mx:
        return points
    buckets = (mx - 2) // 2
    inner = len(points) - 2
    keep = {0, len(points) - 1}
    for b in range(buckets):
        lo = 1 + (b * inner) // buckets
        hi = 1 + ((b + 1) * inner) // buckets
        i_min = i_max = lo
        for i in range(lo, hi):
            if points[i]["equity"] < points[i_min]["equity"]:
                i_min = i
            if points[i]["equity"] > points[i_max]["equity"]:
                i_max = i
        if hi > lo:
            keep.add(i_min)
            keep.add(i_max)
    return [points[i] for i in sorted(keep)]


def calibration(trades):
    settled = _settled(trades)
    out = []
    for lo, hi in CALIBRATION_BUCKETS:
        in_bucket = [t for t in settled if lo <= t["entry_px"] < hi]
        if not in_bucket:
            continue
        implied = _mean([t["entry_px"] for t in in_bucket])
        realized = _mean([contract_value(t["side"], t["outcome"]) for t in in_bucket])
        out.append({
            "bucket": f"{js_to_fixed(lo, 2)}-{js_to_fixed(hi, 2)}",
            "lo": lo,
            "hi": hi,
            "implied": r4(implied),
            "realized": r4(realized),
            "edge_cents": r2((realized - implied) * 100),
            "trades": len(in_bucket),
        })
    return out


def baselines(summaries, *, size=1):
    up = down = fav = 0
    for m in summaries:
        if not m.get("outcome") or m.get("up_px") is None or m.get("down_px") is None:
            continue
        up += (contract_value("UP", m["outcome"]) - m["up_px"]) * size
        down += (contract_value("DOWN", m["outcome"]) - m["down_px"]) * size
        fav_side = "UP" if m["up_px"] >= m["down_px"] else "DOWN"
        fav += (contract_value(fav_side, m["outcome"]) - max(m["up_px"], m["down_px"])) * size
    return {"always_up": r2(up), "always_down": r2(down), "always_favourite": r2(fav)}


def slippage(fills):
    orders = at_quote = walked = nothing = 0
    cost = unfilled_size = requested_size = 0
    slips = []
    for f in fills:
        if f.get("action") != "open":
            continue
        orders += 1
        if f["filled"] > 0 and f["levels_walked"] == 1:
            at_quote += 1
        if f["filled"] > 0 and f["levels_walked"] > 1:
            walked += 1
        if f["filled"] == 0:
            nothing += 1
        if f["filled"] > 0 and f.get("quoted_px") is not None and f.get("avg_px") is not None:
            slips.append((f["avg_px"] - f["quoted_px"]) * 100)
            cost += (f["avg_px"] - f["quoted_px"]) * f["filled"]
        unfilled_size += f["unfilled"]
        requested_size += f["requested"]
    if not orders:
        return {
            "fills_at_quote": None, "partial_fills": None, "unfilled": None,
            "median_slippage_cents": None, "worst_1pct_slippage_cents": None, "pnl_lost_to_slippage": None,
            "orders": 0,
        }
    slips.sort()
    n = len(slips)

    def at(q):
        return slips[min(n - 1, math.floor(q * n))] if n else None

    return {
        "orders": orders,
        "fills_at_quote": r4(at_quote / orders),
        "partial_fills": r4(walked / orders),
        "unfilled": r4(nothing / orders),
        "median_slippage_cents": r2(at(0.5)),
        "worst_1pct_slippage_cents": r2(at(0.99)),
        "pnl_lost_to_slippage": r2(-cost),
        "unfilled_size_ratio": r4(_div(unfilled_size, requested_size)),
    }


def fill_stats(fills):
    return {"count": len(fills), "slippage": slippage(fills)}


def split_by_market(trades, market_meta):
    groups: dict = {}
    for t in trades:
        meta = market_meta.get(t.get("market_id")) or {}
        asset = meta.get("asset")
        interval = meta.get("interval")
        key = f"{'unknown' if asset is None else asset} {'' if interval is None else interval}".strip()
        g = groups.setdefault(key, {"name": key, "pnl": 0, "trades": 0})
        g["pnl"] += t["pnl"]
        g["trades"] += 1
    rows = [{**g, "pnl": r2(g["pnl"])} for g in groups.values()]
    rows.sort(key=lambda r: -abs(r["pnl"]))
    return rows


FEE_MODEL_PM = "polymarket-taker-v1"


def fee_model_report(*, policy, markets):
    """`feeModelReport` in report.mjs: which fees, and how many markets each way."""
    venue = (policy or {}).get("mode") != "bps"
    counts = {"charged": 0, "known_zero": 0, "unknown": 0}
    for m in markets or []:
        fee = (m or {}).get("fee")
        model = fee.get("model") if isinstance(fee, dict) else None
        if model == FEE_MODEL_PM:
            counts["charged"] += 1
        elif model == "none":
            counts["known_zero"] += 1
        else:
            counts["unknown"] += 1
    bps = (policy or {}).get("bps")
    return {
        "mode": "venue" if venue else "bps",
        "bps": None if venue else (bps if _finite(bps) else 0),
        "model": FEE_MODEL_PM if venue else None,
        "rounding": "per book level taken, 5 decimals, ties up" if venue else None,
        "estimate": venue,
        "markets": counts if venue else None,
    }


def build_report(*, run_id, submitted_at, manifest, scope, source_sha256=None, trades, fill_summary,
                 market_summaries, market_meta, fees_paid=0, fill_delay_ms=0, fee_model=None, sweep=None,
                 coverage=None, crosschecks=(), budget=None, seed=None, scanned=None):
    closed = [t for t in trades if _finite(t.get("pnl"))]
    equity = equity_curve(closed)
    matched = sum(1 for c in crosschecks if c.get("match"))
    scope = scope or {}
    manifest = manifest or {}
    return {
        "run_id": run_id,
        "source_sha256": source_sha256,
        "generated_ms": submitted_at,
        "sdk_schema": manifest.get("schema"),
        "language": manifest.get("language"),
        "mode": _coalesce(manifest.get("mode"), "market"),
        "seed": seed,
        "scope": {
            "venue": scope.get("venue"),
            "assets": scope.get("assets") or [],
            "from": scope.get("from"),
            "to": scope.get("to"),
            "market_days": scope.get("market_days"),
        },
        "scanned": scanned or {},
        "metrics": metrics(closed, fees_paid=fees_paid, days=_coalesce(scope.get("archived_day_count"), 1)),
        "equity": [{"ts_ms": p["ts_ms"], "equity": r2(p["equity"])} for p in downsample_equity(equity)],
        "crosscheck": {
            "markets_touched": len({t.get("market_id") for t in closed}),
            "recompute_checks": len(crosschecks),
            "recompute_matches": matched,
            "mismatches": len(crosschecks) - matched,
        },
        "trades_head": closed[:TRADES_HEAD],
        "rows": {"trades": len(closed), "fills": fill_summary["count"], "equity_points": len(equity)},
        "calibration": calibration(closed),
        "baselines": baselines(market_summaries or [], size=(
            _sum([t.get("size") or 0 for t in closed]) / len(closed) if closed else 1)),
        "split": split_by_market(closed, market_meta or {}),
        "slippage": fill_summary["slippage"],
        "fill_delay_ms": fill_delay_ms,
        "fee_model": fee_model,
        "sweep": sweep,
        "coverage": coverage,
        "budget": budget,
    }


# --- harness rows (runner/harness/protocol.mjs) ---------------------------

def parse_trade(raw):
    if not isinstance(raw, dict):
        return None
    mid = raw.get("market_id")
    if not isinstance(mid, str) or not mid:
        return None
    if raw.get("side") not in ("UP", "DOWN"):
        return None
    if not _finite(raw.get("size")) or raw["size"] < 0:
        return None
    if not _finite(raw.get("pnl")):
        return None

    bad = object()

    def px(v):
        if v is None:
            return None
        return v if _finite(v) and 0 <= v <= 1 else bad

    entry, exit_ = px(raw.get("entry_px")), px(raw.get("exit_px"))
    if entry is bad or exit_ is bad:
        return None
    row = {
        "market_id": mid,
        "side": raw["side"],
        "size": raw["size"],
        "entry_px": entry,
        "exit_px": exit_,
        "pnl": raw["pnl"],
        "fees": raw["fees"] if _finite(raw.get("fees")) else 0,
        "opened_ms": raw["opened_ms"] if _finite(raw.get("opened_ms")) else None,
        "closed_ms": raw["closed_ms"] if _finite(raw.get("closed_ms")) else None,
        "how": raw["how"] if isinstance(raw.get("how"), str) else "exit",
    }
    # `outcome: undefined` in the JS: the key is absent once serialised.
    if raw.get("outcome") in OUTCOMES:
        row["outcome"] = raw["outcome"]
    return row


def parse_fill(raw):
    if not isinstance(raw, dict):
        return None
    mid = raw.get("market_id")
    if not isinstance(mid, str) or not mid:
        return None
    if raw.get("side") not in ("UP", "DOWN"):
        return None
    if not _finite(raw.get("requested")) or not _finite(raw.get("filled")):
        return None

    def px(v):
        return v if _finite(v) else None

    tag = raw.get("tag")
    return {
        "ts_ms": raw["ts_ms"] if _finite(raw.get("ts_ms")) else None,
        "market_id": mid,
        "side": raw["side"],
        "action": "reduce" if raw.get("action") == "reduce" else "open",
        "requested": raw["requested"],
        "filled": raw["filled"],
        "unfilled": raw["unfilled"] if _finite(raw.get("unfilled")) else max(0, raw["requested"] - raw["filled"]),
        "avg_px": px(raw.get("avg_px")),
        "worst_px": px(raw.get("worst_px")),
        "quoted_px": px(raw.get("quoted_px")),
        "levels_walked": raw["levels_walked"] if _finite(raw.get("levels_walked")) else 0,
        "fee": raw["fee"] if _finite(raw.get("fee")) else 0,
        "realised": raw["realised"] if _finite(raw.get("realised")) else 0,
        "tag": _js_slice(tag, 64) if isinstance(tag, str) else None,
    }


def _js_slice(s: str, n: int) -> str:
    """`s.slice(0, n)`: n UTF-16 code units, not n code points (a pair can be split)."""
    units = s.encode("utf-16-le", errors="surrogatepass")
    if len(units) <= 2 * n:
        return s
    return units[: 2 * n].decode("utf-16-le", errors="surrogatepass")


def _coalesce(v, default):
    return default if v is None else v
