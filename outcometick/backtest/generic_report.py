"""What a report says about a generic-data run. MUST MATCH runner/generic-report.mjs."""

from __future__ import annotations

import math

from ._js import js_round_fixed

LISTED = 50


def _r2(x):
    return js_round_fixed(x, 2)


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def generic_report_extras(*, trades, markets, unusable, counts):
    marked = [t for t in trades if t.get("how") == "marked"]
    marked_pnl = 0
    for t in marked:
        marked_pnl += t["marked_pnl"] if _fin(t.get("marked_pnl")) else 0
    total = 0
    for t in trades:
        total += t["pnl"] if _fin(t.get("pnl")) else 0
    zero = [t for t in marked if t.get("mark_px") is None]
    ages = [t["mark_age_ms"] for t in marked if _fin(t.get("mark_age_ms"))]
    resolved = {m["market"]["market_id"] for m in markets if m["market"].get("outcome")}
    on_resolved = [t for t in trades if t["market_id"] in resolved]
    opens = [m["market"]["open_ts_ms"] for m in markets if _fin(m["market"].get("open_ts_ms"))]
    closes = [m["market"]["close_ts_ms"] for m in markets if _fin(m["market"].get("close_ts_ms"))]
    res_pnl = 0
    for t in on_resolved:
        res_pnl += t["pnl"] if _fin(t.get("pnl")) else 0
    return {
        "data_kind": "generic",
        "scope": {
            "venue": "polymarket",
            "unit": "market",
            "markets": len(markets),
            "resolved_markets": len(resolved),
            "marked_markets": len(markets) - len(resolved),
            "from_ms": min(opens) if opens else None,
            "to_ms": max(closes) if closes else None,
        },
        "valuation": {
            "net_pnl_includes_marks": len(marked) > 0,
            "realised_pnl": _r2(total - marked_pnl),
            "marked_pnl": _r2(marked_pnl),
            "marked_legs": len(marked),
            "marked_markets": len({t["market_id"] for t in marked}),
            "zero_legs": len(zero),
            "zero_legs_listed": [{
                "market_id": t["market_id"], "side": t["side"], "label": t.get("label"),
                "size": t.get("marked_size"), "reason": t.get("mark_missing"),
            } for t in zero[:LISTED]],
            "max_age_ms": max(ages) if ages else None,
            "complete": len(zero) == 0,
            "note": "marked legs are valued at the external mid (or last public trade) when trading ended "
                    "— not an executable exit",
        },
        "strategy_on_resolved": {
            "markets": len(resolved),
            "net_pnl": _r2(res_pnl),
            "trades": len(on_resolved),
        },
        "coverage_generic": {
            "markets_replayed": len(markets),
            "unusable": len(unusable),
            "unusable_listed": list(unusable[:LISTED]),
            "rows": counts,
        },
    }


def apply_generic_extras(report, extras):
    """MUST MATCH applyGenericExtras: key order is part of the bytes."""
    out = {}
    for k, v in report.items():
        if k == "scope":
            out["scope"] = extras["scope"]
        elif k == "coverage":
            out["coverage"] = v if v is None else {**v, "generic": extras["coverage_generic"]}
        else:
            out[k] = v
        if k == "run_id":
            out["data_kind"] = extras["data_kind"]
    out["valuation"] = extras["valuation"]
    out["strategy_on_resolved"] = extras["strategy_on_resolved"]
    return out
