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


def r6(x):
    return js_round_fixed(x, 6)


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


def edge_per_trade(trades):
    """edgePerTrade in runner/engine/report.mjs."""
    settled = _settled(trades)
    if not settled:
        return None
    edge = 0
    for t in settled:
        edge += contract_value(t["side"], t["outcome"]) - t["entry_px"]
    return r4(edge / len(settled))


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
    down_sq = 0
    winning_days = 0
    best_day = worst_day = None
    for d in daily:
        if d < 0:
            down_sq += d * d
        if d > 0:
            winning_days += 1
        if best_day is None or d > best_day:
            best_day = d
        if worst_day is None or d < worst_day:
            worst_day = d
    downside = math.sqrt(down_sq / len(daily)) if daily else 0
    sortino = None if len(daily) < 2 or downside == 0 else (_mean(daily) / downside) * math.sqrt(365)
    holds = [t["closed_ms"] - t["opened_ms"] for t in closed
             if t.get("opened_ms") is not None and t.get("closed_ms") is not None]

    return {
        "net_pnl": r2(net),
        "win_rate": r4(len(wins) / len(closed)) if closed else None,
        "profit_factor": r2(_div(_sum(wins), abs(_sum(losses)))) if losses else None,
        "max_drawdown": None if dd["pct"] is None else r4(dd["pct"]),
        "max_drawdown_abs": r2(dd["abs"]),
        "sharpe": None if sharpe is None else r2(sharpe),
        "sortino": None if sortino is None else r2(sortino),
        "return_over_max_drawdown": r2(net / dd["abs"]) if dd["abs"] > 0 else None,
        "trading_days": len(daily),
        "winning_days": winning_days,
        "best_day": None if best_day is None else r2(best_day),
        "worst_day": None if worst_day is None else r2(worst_day),
        "trades": len(closed),
        "markets_traded": len({t.get("market_id") for t in closed}),
        "return_on_collateral": r4(net / collateral) if collateral > 0 else None,
        "peak_capital": r2(peak),
        "return_on_peak": r4(net / peak) if peak > 0 else None,
        "holding_ratio": None if hold is None else r4(hold),
        "edge_per_contract": r4(edge_per_contract(closed)) if closed else None,
        "edge_per_trade": edge_per_trade(closed),
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
        if f.get("action") != "open" or f.get("liquidity") == "maker":
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


MARKOUT_KEYS = (("1s", "markout_1s"), ("10s", "markout_10s"), ("60s", "markout_60s"),
                ("settle", "markout_settle"))


class MakerFillAccumulator:
    """makerFillAccumulator in runner/engine/report.mjs."""

    def __init__(self):
        self.join, self.at_fill, self.in_queue = [], [], []
        self.fills = 0
        self.size = 0
        self.mk = {k: {"sum": 0, "size": 0, "n": 0} for k, _ in MARKOUT_KEYS}

    def add(self, f):
        if f.get("liquidity") != "maker" or not f["filled"] > 0:
            return
        self.fills += 1
        self.size += f["filled"]
        if _finite(f.get("queue_ahead_at_join")):
            self.join.append(f["queue_ahead_at_join"])
        if _finite(f.get("queue_ahead_at_fill")):
            self.at_fill.append(f["queue_ahead_at_fill"])
        if _finite(f.get("time_in_queue_ms")):
            self.in_queue.append(f["time_in_queue_ms"])
        for k, field in MARKOUT_KEYS:
            v = f.get(field)
            if not _finite(v):
                continue
            self.mk[k]["sum"] += v * f["filled"]
            self.mk[k]["size"] += f["filled"]
            self.mk[k]["n"] += 1

    def result(self):
        def median(xs):
            if not xs:
                return None
            s = sorted(xs)
            return s[min(len(s) - 1, math.floor(0.5 * len(s)))]
        return {
            "fills": self.fills,
            "size": self.size,
            "median_ahead_at_join": median(self.join),
            "median_ahead_at_fill": median(self.at_fill),
            "median_time_in_queue_ms": median(self.in_queue),
            "markout": {k: {
                "mean": self.mk[k]["sum"] / self.mk[k]["size"] if self.mk[k]["size"] > 0 else None,
                "fills": self.mk[k]["n"],
                "size": self.mk[k]["size"],
            } for k, _ in MARKOUT_KEYS},
        }


class ActivityAccumulator:
    """activityAccumulator in runner/engine/report.mjs."""

    def __init__(self):
        self.contracts = self.value = self.taker = self.maker = 0
        self.fills = 0

    def add(self, f):
        filled = f.get("filled")
        px = f.get("avg_px")
        if not (_finite(filled) and filled > 0) or not _finite(px):
            return
        self.fills += 1
        self.contracts += filled
        self.value += filled * px
        if f.get("liquidity") == "maker":
            self.maker += filled
        else:
            self.taker += filled

    def result(self):
        return {"fills": self.fills, "contracts": r2(self.contracts), "value": r2(self.value),
                "taker_contracts": r2(self.taker), "maker_contracts": r2(self.maker)}


def fill_stats(fills):
    maker = MakerFillAccumulator()
    activity = ActivityAccumulator()
    for f in fills:
        maker.add(f)
        activity.add(f)
    return {"count": len(fills), "slippage": slippage(fills), "makerFills": maker.result(),
            "activity": activity.result()}


RECORD_VALUATION = "liquidation at the best bid; a leg with no bid counts 0; quotes are not checked for freshness"


def account_series(samples, record_ms, *, max_rows=math.inf):
    """accountSeries in runner/engine/report.mjs."""
    n = len(samples)
    if not n or not record_ms > 0:
        return {"truncated": False, "points": []}
    first = math.inf
    last = -math.inf
    for x in samples:
        if x["ts"] < first:
            first = x["ts"]
        if x["final"] and x["ts"] > last:
            last = x["ts"]
    if last == -math.inf:
        for x in samples:
            if x["ts"] > last:
                last = x["ts"]
    g0 = math.ceil(first / record_ms) * record_ms
    grid_count = max(0, math.floor(last / record_ms) - math.ceil(first / record_ms) + 1)
    total = grid_count + (0 if last % record_ms == 0 else 1)
    if n + total > max_rows:
        return {"truncated": True, "points": []}
    order = sorted(range(n), key=lambda i: (samples[i]["ts"], i))
    latest = {}
    st = {"value": 0, "position": 0, "open": 0, "i": 0}
    points = []

    def advance_to(t):
        while st["i"] < n and samples[order[st["i"]]]["ts"] <= t:
            x = samples[order[st["i"]]]
            v = (x["cash"] + x["marked"]) + x["position_value"]
            prev = latest.get(x["key"])
            if prev is not None:
                st["value"] += v - prev[0]
                st["position"] += x["position_value"] - prev[1]
                if prev[1] > 0:
                    st["open"] -= 1
            else:
                st["value"] += v
                st["position"] += x["position_value"]
            if x["position_value"] > 0:
                st["open"] += 1
            latest[x["key"]] = (v, x["position_value"])
            st["i"] += 1
        points.append({"ts": t, "value": st["value"], "position_value": st["position"], "open_positions": st["open"]})
    for k in range(grid_count):
        advance_to(g0 + k * record_ms)
    if last % record_ms != 0:
        advance_to(last)
    return {"truncated": False, "points": points}


def record_report(*, samples, record_ms, harness, max_rows=math.inf):
    """recordReport in runner/engine/report.mjs."""
    if not record_ms:
        return {"block": None, "points": [], "samples": []}
    if not harness or harness.get("truncated"):
        return {"block": {"interval_ms": record_ms, "truncated": True}, "points": [], "samples": []}
    series = account_series(samples, record_ms, max_rows=max_rows)
    if series["truncated"]:
        return {"block": {"interval_ms": record_ms, "truncated": True}, "points": [], "samples": []}
    pts = series["points"]
    peak = max_dd = 0
    dd_at = None
    worst = None
    max_pos = sum_pos = 0
    for p in pts:
        if p["value"] > peak:
            peak = p["value"]
        if peak - p["value"] > max_dd:
            max_dd = peak - p["value"]
            dd_at = p["ts"]
        if worst is None or p["value"] < worst[1]:
            worst = (p["ts"], p["value"])
        if p["position_value"] > max_pos:
            max_pos = p["position_value"]
        sum_pos += p["position_value"]
    return {
        "block": {
            "interval_ms": record_ms,
            "truncated": False,
            "valuation": RECORD_VALUATION,
            "points": len(pts),
            "instances": len({x["key"] for x in samples}),
            "final_value": r2(pts[-1]["value"]) if pts else None,
            "max_drawdown_abs": r2(max_dd),
            "max_drawdown_at": dd_at,
            "worst_point": {"ts": worst[0], "value": r2(worst[1])} if worst else None,
            "max_position_value": r2(max_pos),
            "mean_position_value": r2(sum_pos / len(pts)) if pts else None,
        },
        # Rounded where they are written out -- see recordReport.
        "points": [{**p, "value": r6(p["value"]), "position_value": r6(p["position_value"])} for p in pts],
        "samples": [{**x, "cash": r6(x["cash"]), "marked": r6(x["marked"]), "position_value": r6(x["position_value"])}
                    for x in samples],
    }


MARKET_COLUMNS = (
    "market_id", "asset", "interval", "settlement", "open_ts_ms", "close_ts_ms", "strike", "outcome",
    "up_px", "down_px", "entered", "trades", "contracts", "net_pnl", "fees", "first_entry_ms",
    "edge_per_contract", "edge_per_trade",
)


def market_rows(trades, market_meta):
    """marketRows in runner/engine/report.mjs: every market replayed, entered or not."""
    by_market: dict = {}
    for t in trades:
        if not _finite(t.get("pnl")):
            continue
        by_market.setdefault(t.get("market_id"), []).append(t)
    rows = []
    for mid, m in (market_meta or {}).items():
        ts = by_market.get(mid, [])
        pnl = fees = contracts = 0
        first = None
        for t in ts:
            pnl += t["pnl"]
            fees += _coalesce(t.get("fees"), 0)
            contracts += t.get("size") or 0
            o = t.get("opened_ms")
            if o is not None and (first is None or o < first):
                first = o
        settled = _settled(ts)
        rows.append({
            "market_id": mid,
            "asset": m.get("asset"),
            "interval": m.get("interval"),
            "settlement": m.get("stream"),
            "open_ts_ms": m.get("open_ts_ms"),
            "close_ts_ms": m.get("close_ts_ms"),
            "strike": m.get("strike"),
            "outcome": m.get("outcome"),
            "up_px": m.get("up_px"),
            "down_px": m.get("down_px"),
            "entered": len(ts) > 0,
            "trades": len(ts),
            "contracts": r4(contracts),
            "net_pnl": r4(pnl),
            "fees": r4(fees),
            "first_entry_ms": first,
            "edge_per_contract": r4(edge_per_contract(settled)) if settled else None,
            "edge_per_trade": edge_per_trade(settled),
        })
    return rows


def by_settlement(rows, trades):
    """bySettlement in runner/engine/report.mjs."""
    rule = {r["market_id"]: (r["settlement"] if r["settlement"] is not None else "unknown") for r in rows}
    groups: dict = {}

    def group(k):
        if k not in groups:
            groups[k] = {"settlement": k, "markets": 0, "markets_entered": 0, "trades": []}
        return groups[k]
    for r in rows:
        g = group(r["settlement"] if r["settlement"] is not None else "unknown")
        g["markets"] += 1
        if r["entered"]:
            g["markets_entered"] += 1
    for t in trades:
        if not _finite(t.get("pnl")) or t.get("market_id") not in rule:
            continue
        group(rule[t.get("market_id")])["trades"].append(t)
    out = []
    for k in sorted(groups):
        g = groups[k]
        pnl = 0
        for t in g["trades"]:
            pnl += t["pnl"]
        settled = _settled(g["trades"])
        out.append({
            "settlement": k,
            "markets": g["markets"],
            "markets_entered": g["markets_entered"],
            "trades": len(g["trades"]),
            "net_pnl": r2(pnl),
            "edge_per_contract": r4(edge_per_contract(settled)) if settled else None,
            "edge_per_trade": edge_per_trade(settled),
        })
    return out


def maker_report(*, stats, fills, queue_model, print_lag_ms, queue_prob=None):
    """makerReport in runner/engine/report.mjs."""
    if not stats or not stats.get("submitted", 0) > 0:
        return None

    def rate(a, b):
        return r4(a / b) if b > 0 else None
    f = fills if fills is not None else MakerFillAccumulator().result()
    s = stats
    return {
        "queue_model": queue_model,
        **({"queue_prob": {"prob": queue_prob["prob"], "n": queue_prob.get("n")}} if queue_prob else {}),
        "print_lag_ms": print_lag_ms,
        "orders": {
            "submitted": s["submitted"],
            "rested": s["rested"],
            "with_maker_fill": s["orders_with_maker_fill"],
            "fully_filled": s["fully_filled"],
            "cancelled": s["cancelled"],
            "cancelled_before_entry": s["cancelled_before_entry"],
            "cancelled_no_position": s["cancelled_no_position"],
            "expired": s["expired"],
            "rejected_invalid": s["rejected_invalid"],
            "rejected_post_only": s["rejected_post_only"],
            "rejected_self_cross": s["rejected_self_cross"],
            "rejected_duplicate_id": s["rejected_duplicate_id"],
        },
        "size": {
            "rested": r2(s["rested_size"]),
            "maker_filled": r2(s["maker_filled_size"]),
            "taker_on_arrival": r2(s["taker_on_arrival_size"]),
        },
        "fill_rate_size": rate(s["maker_filled_size"], s["rested_size"]),
        "fill_rate_orders": rate(s["orders_with_maker_fill"], s["rested"]),
        "maker_fills": f["fills"],
        "median_ahead_at_join": r2(f["median_ahead_at_join"]),
        "median_ahead_at_fill": r2(f["median_ahead_at_fill"]),
        "median_time_in_queue_ms": f["median_time_in_queue_ms"],
        "markout": {k: {"mean": r4(m["mean"]), "fills": m["fills"], "size": r2(m["size"])}
                    for k, m in f["markout"].items()},
        "lag_suppressed_size": r2(s["lag_suppressed_size"]),
        "crossed_observations": s["crossed_observations"],
    }


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


def orders_report(*, policy, stats):
    """MUST MATCH ordersReport in runner/engine/report.mjs."""
    if not policy or (not policy.get("fok") and not policy.get("order_rules") and policy.get("maker_rebate") is None
                      and not policy.get("quote_guard") and not policy.get("mirror_depth")):
        return None
    rules = policy.get("order_rules")
    if rules is None:
        rules_out = None
    elif rules.get("mode") == "assumed":
        rules_out = {
            "mode": "assumed", "assumed_tick": rules["tick"], "assumed_min_size": rules["min_size"],
            "size_step": rules["size_step"],
            "note": "archive markets do not carry their tick or minimum size: these are values measured on the archive, "
                    "not read per market \u2014 a market quoting in 0.01 would refuse a 0.001 limit accepted here, "
                    "one quoting finer would accept limits refused here",
        }
    else:
        rules_out = {"mode": "venue", "size_step": 0.01, "note": "tick and minimum size read from each market"}
    rebate = policy.get("maker_rebate")
    return {
        "fok": bool(policy.get("fok")),
        "order_rules": rules_out,
        "quote_guard": None if policy.get("quote_guard") is None else {
            **policy["quote_guard"],
            "applies_to": ("opening and exit taker orders" if policy["quote_guard"].get("include_exits")
                           else "opening taker orders (exits are never blocked)"),
        },
        "mirror_depth": policy.get("mirror_depth"),
        "maker_rebate": None if rebate is None else {
            "rate": rebate,
            "total": r2((stats or {}).get("rebates", 0)),
            "note": "an estimate: the venue pays maker rebates from a daily pool, pro rata, not per fill",
        },
        "rejected": dict((stats or {}).get("rejected") or {}),
    }


def build_report(*, run_id, submitted_at, manifest, scope, source_sha256=None, trades, fill_summary,
                 market_summaries, market_meta, fees_paid=0, fill_delay_ms=0, fee_model=None, sweep=None,
                 coverage=None, crosschecks=(), budget=None, seed=None, scanned=None, maker=None, orders=None,
                 timers=None, fingerprint=None, record=None, spot=None):
    closed = [t for t in trades if _finite(t.get("pnl"))]
    equity = equity_curve(closed)
    matched = sum(1 for c in crosschecks if c.get("match"))
    scope = scope or {}
    manifest = manifest or {}
    return {
        "run_id": run_id,
        "source_sha256": source_sha256,
        "fingerprint": fingerprint,
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
        "by_settlement": by_settlement(market_rows(closed, market_meta or {}), closed),
        "activity": fill_summary.get("activity"),
        "slippage": fill_summary["slippage"],
        "fill_delay_ms": fill_delay_ms,
        "fee_model": fee_model,
        "maker": maker,
        **({"orders": orders} if orders else {}),
        **({"timers": timers} if timers else {}),
        **({"record": record} if record else {}),
        **({"spot": spot} if spot else {}),
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
    if isinstance(raw.get("label"), str):
        row.update(_generic_trade_fields(raw))
    return row


def _num_or_none(v):
    return v if _finite(v) else None


def _generic_trade_fields(raw):
    """A generic market's extra trade fields. MUST MATCH genericTradeFields in protocol.mjs."""
    mp = raw.get("mark_px")
    return {
        "label": _js_slice(raw["label"], 200),
        "mark_px": None if mp is None else (mp if _finite(mp) and 0 <= mp <= 1 else None),
        "marked_size": _num_or_none(raw.get("marked_size")),
        "marked_pnl": _num_or_none(raw.get("marked_pnl")),
        "mark_source": raw.get("mark_source") if raw.get("mark_source") in ("up_mid", "down_mid", "last_trade") else None,
        "mark_ts_ms": _num_or_none(raw.get("mark_ts_ms")),
        "mark_age_ms": _num_or_none(raw.get("mark_age_ms")),
        "mark_missing": raw.get("mark_missing") if raw.get("mark_missing") in ("no_quote", "stale") else None,
    }


def parse_record(raw):
    """parseRecord in runner/harness/protocol.mjs."""
    if not isinstance(raw, dict) or not isinstance(raw.get("key"), str):
        return None
    for f in ("ts", "cash", "marked", "position_value"):
        if not _finite(raw.get(f)):
            return None
    if not isinstance(raw.get("final"), bool):
        return None
    return {"key": raw["key"], "ts": raw["ts"], "cash": raw["cash"], "marked": raw["marked"],
            "position_value": raw["position_value"], "final": raw["final"]}


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
        "liquidity": "maker" if raw.get("liquidity") == "maker" else "taker",
        "order_id": _js_slice(raw["order_id"], 16) if isinstance(raw.get("order_id"), str) else None,
        "queue_ahead_at_join": px(raw.get("queue_ahead_at_join")),
        "queue_ahead_at_fill": px(raw.get("queue_ahead_at_fill")),
        "time_in_queue_ms": px(raw.get("time_in_queue_ms")),
        "markout_1s": px(raw.get("markout_1s")),
        "markout_10s": px(raw.get("markout_10s")),
        "markout_60s": px(raw.get("markout_60s")),
        "markout_settle": px(raw.get("markout_settle")),
        **({"label": _js_slice(raw["label"], 200)} if isinstance(raw.get("label"), str) else {}),
        **({"reject_reason": raw["reject_reason"] if raw["reject_reason"] in REJECT_REASONS else None}
           if "reject_reason" in raw else {}),
        **({"rebate": raw["rebate"] if _finite(raw.get("rebate")) else 0} if "rebate" in raw else {}),
    }


# MUST MATCH REJECT_REASONS in runner/harness/protocol.mjs.
REJECT_REASONS = ("fok_unfilled", "quote_guard", "below_min_size", "size_step", "off_tick")


def _js_slice(s: str, n: int) -> str:
    """`s.slice(0, n)`: n UTF-16 code units, not n code points (a pair can be split)."""
    units = s.encode("utf-16-le", errors="surrogatepass")
    if len(units) <= 2 * n:
        return s
    return units[: 2 * n].decode("utf-16-le", errors="surrogatepass")


def _coalesce(v, default):
    return default if v is None else v
