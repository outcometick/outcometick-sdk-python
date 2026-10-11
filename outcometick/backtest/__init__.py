"""Backtest a Polymarket / Predict.fun strategy on your own machine, in Python.

    from outcometick.backtest import run, load_sample

    class Favourite:
        def on_tick(self, ctx, tick):
            ...

    result = run(Favourite, load_sample())
    print(result.summary())

The same engine, decoder, settlement, fees and report as a backtest queued on
outcometick.com and as `ot run` -- not a lookalike. client/backtest/
equivalence.test.mjs replays the public samples through this package and
through `ot run` and requires every trade, fill and report field to match.

Limits, stated plainly:
  - Market orders that take liquidity (IOC) only. No resting orders: a fill for
    a resting order needs a queue-position model, and a guessed one inflates
    market-making returns.
  - Depth is the archived book, reconstructed and thinned to the capture
    cadence; the top-of-book stream can only delete levels, never add them.
  - Polymarket fees are the venue's taker schedule per market, computed per
    book level (an estimate: the venue rounds per maker order matched). The
    archive has no Predict.fun fee schedule, so those runs are fee-free unless
    you pass `fee_bps`.
  - Reference feeds (Binance) live on our runner, not in the archive, so a run
    that needs them has to be submitted rather than run here. Custom CSV series
    (`series`) are read by `ot run` and the hosted runner, not yet by this one.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

from ._contract import CONTRACT
from .datasets import (
    ManifestError, engine_policy_for, normalize_feed_latency, normalize_fok, normalize_intervals, normalize_maker_rebate,
    normalize_mirror_depth, normalize_order_rules, normalize_queue_model, normalize_quote_guard, resting_policy_for,
)
from .decode import BookThrottle, bbo_coverage, build_coverage, count_market_days, count_streams, sort_markets_for_replay
from .feed import _dumps as _feed_dumps, market_lines, session_lines
from ._parallel import SharedState, default_workers, fork_context, replay_markets
from .local import archive_venues, default_venue, load_local_day, load_local_day_cached, local_days, looks_like_archive
from .report import (
    REJECT_REASONS, build_report, fee_model_report, fill_stats, maker_report, orders_report, parse_fill, parse_trade,
)
from .generic import Dataset, DatasetError, looks_like_generic_dataset
from .generic_report import apply_generic_extras, generic_report_extras
from .sample import load_sample
from .sweep import SweepResult, sweep

from ._engine import load as _load_engine

_ENGINE = _load_engine()
_Book = _ENGINE["otengine"].Book
RunAbort = _ENGINE["otengine"].RunAbort
_H = _ENGINE["otharness"]
CHANNEL_FILL, CHANNEL_LOG, CHANNEL_PROGRESS = _H.CHANNEL_FILL, _H.CHANNEL_LOG, _H.CHANNEL_PROGRESS
CHANNEL_RESULT, CHANNEL_TRADE, EXIT_OK, run_job = _H.CHANNEL_RESULT, _H.CHANNEL_TRADE, _H.EXIT_OK, _H.run_job

__all__ = ["run", "run_template", "sweep", "load_sample", "Result", "SweepResult", "BacktestError", "Dataset"]

HOOKS = ("on_market_open", "on_tick", "on_book", "on_trade", "on_settle", "on_timer")
_EMITTING = ("on_tick", "on_book", "on_trade", "on_timer")
# The assets `ot run` replays when none are named.
_DEFAULT_ASSETS = ("BTC", "ETH", "SOL", "XRP")


class BacktestError(RuntimeError):
    """The run was refused or could not produce a report. `code` is the same
    rejection code a queued run would report (E_MANIFEST, E_RUNTIME, ...)."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class Result:
    """A finished backtest: the report plus every trade, fill and log line."""

    def __init__(self, report: dict, trades: list, fills: list, logs: str):
        self.report = report
        self.trades = trades
        self.fills = fills
        self.logs = logs

    @property
    def metrics(self) -> dict:
        return self.report["metrics"]

    def trades_df(self):
        """Trades as a pandas DataFrame (needs pandas)."""
        import pandas as pd
        return pd.DataFrame(self.trades)

    def fills_df(self):
        """Fills as a pandas DataFrame (needs pandas)."""
        import pandas as pd
        return pd.DataFrame(self.fills)

    def summary(self) -> str:
        m = self.metrics
        fm = self.report.get("fee_model") or {}

        def money(v):
            if v is None:
                return "—"
            if v == 0:
                return "$0.00"
            return f"{'-' if v < 0 else '+'}${abs(v):,.2f}"

        def pct(v):
            return "—" if v is None else f"{v * 100:.1f}%"

        fees = money(m.get("fees"))
        if fm.get("mode") == "venue":
            c = fm.get("markets") or {}
            if not c.get("charged") and not c.get("known_zero"):
                fees += "  (no fee schedule in this archive; pass fee_bps to set one)"
            elif c.get("unknown"):
                fees += f"  (venue schedule, estimated; {c['unknown']} market(s) had none and paid nothing)"
            else:
                fees += "  (venue schedule, estimated per book level)"
        elif fm.get("mode") == "bps":
            fees += f"  (flat {fm.get('bps')} bps)"
        rows = [
            ("market-days", self.report["scanned"].get("market_days")),
            ("net pnl", money(m.get("net_pnl"))),
            ("trades", m.get("trades")),
            ("win rate", pct(m.get("win_rate"))),
            ("brier", m.get("brier_score") if m.get("brier_score") is not None else "—"),
            ("edge/contract", "—" if m.get("edge_per_contract") is None else f"{m['edge_per_contract'] * 100:.1f}¢"),
            ("max drawdown", money(None if m.get("max_drawdown_abs") is None else -m["max_drawdown_abs"])),
            ("fees", fees),
            ("lost to slippage", money(self.report["slippage"].get("pnl_lost_to_slippage"))),
        ]
        b = self.report.get("baselines") or {}
        rows.append(("vs always-favourite", money(b.get("always_favourite"))))
        return "\n".join(f"  {k:<20} {v}" for k, v in rows)

    def plot(self, ax=None):
        """Realised PnL curve, one point per closed trade (needs matplotlib)."""
        import matplotlib.pyplot as plt
        from datetime import datetime, timezone

        pts = [p for p in self.report["equity"] if p.get("ts_ms") is not None]
        ax = ax or plt.subplots(figsize=(9, 3.5))[1]
        ax.plot([datetime.fromtimestamp(p["ts_ms"] / 1000, timezone.utc) for p in pts],
                [p["equity"] for p in pts], lw=1.4)
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_ylabel("realised PnL ($)")
        ax.set_title(f"{self.report['scope']['venue']} · {', '.join(self.report['scope']['assets'])}"
                     f" · {self.report['scope']['from']}..{self.report['scope']['to']}")
        return ax

    def __repr__(self):
        m = self.metrics
        return f"<Result net_pnl={m.get('net_pnl')} trades={m.get('trades')} market_days={m.get('market_days')}>"


def _hooks_of(klass) -> list:
    return [h for h in HOOKS if callable(getattr(klass, h, None))]


def run(strategy, data, *, venue=None, assets=None, days=None, datasets=None, intervals=None,
        params=None, mode="market", latency_ms=0, fee_bps=None, seed=1, hooks=None,
        cancel_latency_ms=None, cache=True, workers=None, mark_max_age_ms=None, fok=False, order_rules=None,
        maker_rebate=None, quote_guard=None, mirror_depth=None, book_throttle=None, feed_latency=None,
        queue_model=None) -> Result:
    """Backtest `strategy` (a class) over an unpacked archive at `data` -- or
    over your own data for ANY Polymarket binary market: a `Dataset`, or a
    directory with markets.json / markets.jsonl and CLOB frame files.

    Generic data: every market in it is replayed (no assets / days / intervals);
    datasets are what the frames hold. A resolved market pays at its resolution
    time; an unresolved one is valued at the external mid (else last public
    trade) when trading ends -- `mark_max_age_ms` refuses older marks (a leg
    with no usable mark is valued at 0, and the report says so).

    Execution options (all off by default; same meaning as the manifest keys):
    fok           True: orders may use tif="fok" (all within the limit, or none)
    order_rules   "assumed" (archive Polymarket: tick 0.001, min 5 shares, sizes
                  in 0.01) or "venue" (your own data: each market's own tick/min)
    maker_rebate  share of the taker fee credited on resting fills (an estimate)
    quote_guard   {"max_age_ms": n, "ask_sum": [lo, hi]}: opening taker orders
                  only execute against fresh, consistent quotes
    mirror_depth  "shared": on single-book markets one fill consumes both tokens
    feed_latency  ms between an event on the venue and the strategy seeing it
                  (hooks fire late and ctx.book() lags; orders still match the
                  venue's book). on_timer + ctx.schedule() wake the strategy
                  at chosen times.
    queue_model   resting orders: where unexplained book decreases came from in
                  the queue -- {"prob": "power", "n": 2} (power/power2/power3,
                  n 1..10) or {"prob": "log"} (log/log2). None: the conservative
                  default (cancellations behind the order).
    book_throttle False: replay the archive's book unthinned (local only); True
                  or False also writes coverage.book_throttle

    assets    default BTC, ETH, SOL, XRP (as `ot run`); name others to include them.
    datasets  default: settlement and book (so orders can fill), plus trades
              when the class has on_trade. Add "bbo" for the top-of-book bound.
    intervals default ["5m"]; Polymarket has 5m and 15m.
    fee_bps   None = each market's venue schedule; a number = flat override.
    cancel_latency_ms  how long ctx.cancel() takes to reach the venue (resting
              orders); None = the same as latency_ms. Resting (gtc) orders need
              datasets to include "book" and "trades", and Polymarket.
    cache     keep decoded days under ~/.cache/outcometick/decoded (or a path),
              so the next run over the same days starts at once. False: never.
    workers   None (default): decode on every CPU core, replay in this process —
              always the single-process result. A number: decode AND replay
              markets on that many processes (market mode). Parallel replay
              relies on market mode's contract: a strategy keeps its state on
              the instance. State found on its class (nested classes included)
              or in params makes the run replay in one process instead.
    """
    if not isinstance(strategy, type):
        raise TypeError("pass the strategy CLASS, not an instance")
    declared = list(hooks) if hooks is not None else _hooks_of(strategy)
    if not any(h in _EMITTING for h in declared):
        raise BacktestError("E_MANIFEST", "the strategy defines none of on_tick / on_book / on_trade, so it can never trade")
    if datasets is None:
        datasets = ["settlement", "book"]
        if "on_trade" in declared:
            datasets.append("trades")
    manifest = {
        "schema": 1,
        "language": "python@3.14",
        "entry": {"file": "<in-process>", "className": strategy.__name__},
        "hooks": declared,
        "datasets": list(datasets),
        "intervals": normalize_intervals(intervals),
        "latency": latency_ms or None,
        "cancel_latency": cancel_latency_ms,
        "fee_bps": fee_bps,
        "mode": mode,
        "params": dict(params or getattr(strategy, "params", None) or {}),
        **({"fok": True} if fok else {}),
        **({"order_rules": order_rules} if order_rules is not None else {}),
        **({"maker_rebate": maker_rebate} if maker_rebate is not None else {}),
        **({"quote_guard": quote_guard} if quote_guard is not None else {}),
        **({"mirror_depth": mirror_depth} if mirror_depth is not None else {}),
        **({"feed_latency": feed_latency} if feed_latency is not None else {}),
        **({"queue_model": queue_model} if queue_model is not None else {}),
    }
    return _run(manifest, lambda: strategy, data=data, venue=venue, assets=assets, days=days, seed=seed, cache=cache,
                workers=workers, mark_max_age_ms=mark_max_age_ms, book_throttle=book_throttle)


def run_template(path, data, *, venue=None, assets=None, days=None, seed=1, fee_bps=None, cache=True,
                 workers=None) -> Result:
    """Run a strategy directory (outcometick.json + source), like `ot run`."""
    import importlib.util
    import json

    with open(os.path.join(path, "outcometick.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    if not str(doc.get("language", "")).startswith("python"):
        raise BacktestError("E_MANIFEST", "this runs Python strategies; use `ot run` for JavaScript")
    if doc.get("reference"):
        raise BacktestError("E_MANIFEST", "reference feeds live on the runner, not in the archive — submit this one instead")
    if doc.get("series"):
        # Refused rather than dropped: replaying without them would hand ctx.ext()
        # nothing and report a run that `ot run` and the queue would not produce.
        raise BacktestError("E_MANIFEST", "custom series (manifest \"series\") are not read by the Python runner yet — "
                                          "run this one with `ot run` or submit it")
    file_name, _, class_name = str(doc.get("entry", "")).rpartition(":")
    module_name = os.path.splitext(os.path.basename(file_name))[0]
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(path, file_name))
    if spec is None or spec.loader is None:
        raise BacktestError("E_ENTRY", f"could not load {file_name}")
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs, as the sandbox harness does: dataclasses and
    # typing resolve a class's module through sys.modules, so a strategy using
    # @dataclass with postponed annotations fails to import otherwise.
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as err:  # noqa: BLE001
        if previous is not None:
            sys.modules[module_name] = previous
        else:
            sys.modules.pop(module_name, None)
        raise BacktestError("E_ENTRY", f"could not load {file_name}: {err}") from err
    klass = getattr(module, class_name, None)
    if not isinstance(klass, type):
        raise BacktestError("E_ENTRY", f"{file_name} does not define a class named {class_name}")
    manifest = {
        "schema": doc.get("schema", 1),
        "language": doc.get("language"),
        "entry": {"file": file_name, "className": class_name},
        "hooks": list(doc.get("hooks") or []),
        "datasets": list(doc.get("datasets") or []),
        "intervals": normalize_intervals(doc.get("intervals")),
        "latency": doc.get("latency"),
        "cancel_latency": doc.get("cancel_latency"),
        "fee_bps": fee_bps if fee_bps is not None else doc.get("fee_bps"),
        "mode": doc.get("mode") or "market",
        "params": dict(doc.get("params") or {}),
        **{k: doc[k] for k in ("fok", "order_rules", "maker_rebate", "quote_guard", "mirror_depth", "feed_latency",
                                    "queue_model")
           if k in doc},
    }
    return _run(manifest, lambda: klass, data=data, venue=venue, assets=assets, days=days, seed=seed, cache=cache,
                workers=workers)


def _validate(manifest) -> None:
    """The manifest rules a queued run enforces (api/lib/backtest-manifest.mjs)
    that decide what a local run would replay. Refused here rather than
    replayed differently: a local run that quietly read nothing for a misspelt
    dataset would look like a strategy that never trades."""
    unknown = [h for h in manifest["hooks"] if h not in CONTRACT["known_hooks"]]
    if unknown:
        raise BacktestError("E_MANIFEST", f"unknown hook {unknown[0]!r}; known: {', '.join(CONTRACT['known_hooks'])}")
    known = set(CONTRACT["known_datasets"]) | set(CONTRACT["derived_datasets"])
    if not manifest["datasets"]:
        raise BacktestError("E_MANIFEST", "datasets must be a non-empty list")
    for d in manifest["datasets"]:
        if d not in known:
            raise BacktestError("E_MANIFEST", f"unknown dataset {d!r}; known: {', '.join(sorted(known))}")
    latency = manifest.get("latency")
    if latency is not None and (not isinstance(latency, int) or isinstance(latency, bool)
                                or not 0 < latency <= CONTRACT["max_latency_ms"]):
        raise BacktestError("E_MANIFEST", f"latency must be a whole number of milliseconds between 1 and "
                                          f"{CONTRACT['max_latency_ms']}, got {latency!r}")
    cancel = manifest.get("cancel_latency")
    if cancel is not None and (not isinstance(cancel, int) or isinstance(cancel, bool)
                               or not 0 <= cancel <= CONTRACT["max_latency_ms"]):
        raise BacktestError("E_MANIFEST", f"cancel_latency must be a whole number of milliseconds between 0 and "
                                          f"{CONTRACT['max_latency_ms']}, got {cancel!r}")
    fee_bps = manifest.get("fee_bps")
    if fee_bps is not None and (not isinstance(fee_bps, (int, float)) or isinstance(fee_bps, bool)
                                or not 0 <= fee_bps <= CONTRACT["max_fee_bps"]):
        raise BacktestError("E_MANIFEST", f"fee_bps must be between 0 and {CONTRACT['max_fee_bps']}, got {fee_bps!r}")
    if manifest["mode"] not in ("market", "session"):
        raise BacktestError("E_MANIFEST", f"unknown mode {manifest['mode']!r}; known: market, session")
    try:
        normalize_fok(manifest.get("fok"))
        normalize_order_rules(manifest.get("order_rules"))
        normalize_maker_rebate(manifest.get("maker_rebate"))
        normalize_quote_guard(manifest.get("quote_guard"))
        normalize_mirror_depth(manifest.get("mirror_depth"))
        normalize_feed_latency(manifest.get("feed_latency"))
        normalize_queue_model(manifest.get("queue_model"))
    except ManifestError as err:
        raise BacktestError("E_MANIFEST", str(err)) from None


def _maker_stats(raw):
    """parseMakerStats in runner/harness/protocol.mjs."""
    if not isinstance(raw, dict):
        return None
    keys = _ENGINE["otmaker"].new_maker_stats()
    out = {}
    for k in keys:
        v = raw.get(k)
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float("inf"),) and v >= 0
        out[k] = v if ok else 0
    return out


def _fee_policy(manifest) -> dict:
    """feePolicyFor in api/lib/backtest-datasets.mjs (no operator override here)."""
    if manifest.get("fee_bps") is not None:
        return {"mode": "bps", "bps": manifest["fee_bps"]}
    return {"mode": "venue"}


def _cache_root(cache):
    if cache is False or cache is None:
        return None
    if isinstance(cache, (str, os.PathLike)):
        return os.fspath(cache)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "outcometick", "decoded")


def _run(manifest, get_class, *, data, venue, assets, days, seed, cache=True, workers=None,
         mark_max_age_ms=None, book_throttle=None) -> Result:
    if book_throttle is not None and not isinstance(book_throttle, bool):
        raise BacktestError("E_MANIFEST", f"book_throttle must be True or False, got {book_throttle!r}")
    if mark_max_age_ms is not None and (not isinstance(mark_max_age_ms, int) or isinstance(mark_max_age_ms, bool)
                                        or mark_max_age_ms < 0):
        raise BacktestError("E_MANIFEST", f"mark_max_age_ms must be a whole number of milliseconds, got {mark_max_age_ms!r}")
    if isinstance(data, Dataset) or looks_like_generic_dataset(os.fspath(data)):
        if book_throttle is not None:
            raise BacktestError("E_MANIFEST", "book_throttle applies to the archive; your own data is never thinned")
        return _run_generic(manifest, get_class, data=data, venue=venue, days=days, seed=seed, workers=workers,
                            mark_max_age_ms=mark_max_age_ms)
    if mark_max_age_ms is not None:
        raise BacktestError("E_MANIFEST", "mark_max_age_ms applies to generic data only: archive markets settle "
                                          "on their official outcome")
    root = os.fspath(data)
    if not looks_like_archive(root):
        raise BacktestError("E_DATA", f"{os.path.abspath(root)} does not look like an archive — "
                            "no recognisable data files under it. load_sample() fetches the free sample.")
    _validate(manifest)

    available = local_days(root)
    if not available:
        raise BacktestError("E_DATA", f"no dated files under {os.path.abspath(root)}")
    want_days = list(days) if days else available
    unknown = [d for d in want_days if d not in available]
    if unknown:
        raise BacktestError("E_DATA", f"{', '.join(unknown)} not in {os.path.abspath(root)} — "
                            f"it holds {available[0]}..{available[-1]}")
    venue = venue or default_venue(archive_venues(root))
    asset_list = [str(a).strip().upper() for a in (assets or []) if str(a).strip()]
    scope_assets = asset_list or list(_DEFAULT_ASSETS)

    cache_root = _cache_root(cache)
    n_workers = default_workers() if workers is None else max(1, int(workers))
    events_dir = tempfile.mkdtemp(prefix="ot-events-")
    try:
        markets, missing, bbo_applied = [], [], {}
        throttle = (None if book_throttle is False
                    else BookThrottle(venue=venue, assets=scope_assets, frm=want_days[0], to=want_days[-1]))
        try:
            policy = engine_policy_for(manifest, venue, "archive")
        except ManifestError as err:
            raise BacktestError("E_MANIFEST", str(err)) from None
        for day in want_days:
            kw = dict(root=root, day=day, venue=venue, assets=scope_assets, datasets=manifest["datasets"],
                      intervals=manifest["intervals"], throttle=throttle, book_factory=lambda: _Book(None),
                      workers=n_workers, ctx=fork_context())
            try:
                if cache_root:
                    loaded = load_local_day_cached(cache_root=cache_root, engine_file=_ENGINE["otengine"].__file__, **kw)
                else:
                    loaded = load_local_day(events_dir=events_dir, **kw)
            except ManifestError as err:
                raise BacktestError("E_MANIFEST", str(err)) from None
            if loaded.get("bbo_keys"):
                bbo_applied.setdefault(day, set()).update(loaded["bbo_keys"])
            unusable = loaded.get("unusable") or []
            if not loaded["markets"]:
                entry = {"day": day, "reason": loaded.get("reason") or "no markets"}
                if unusable:
                    entry.update({"partial": False, "markets": len(unusable), "dropped": unusable[:10],
                                  "reasons": list(dict.fromkeys(u["why"] for u in unusable))})
                missing.append(entry)
                continue
            if unusable:
                why = f"{len(unusable)} market(s) dropped: {unusable[0]['why']}"
                entry = {"day": day, "partial": True, "markets": len(unusable), "reason": why,
                         "dropped": unusable[:10]}
                if len(unusable) > 10:
                    entry["dropped_truncated"] = len(unusable) - 10
                entry["reasons"] = list(dict.fromkeys(u["why"] for u in unusable))
                missing.append(entry)
            markets.extend(loaded["markets"])
        if not markets:
            raise BacktestError("E_DATA", "no market-days could be read from that archive")
        if manifest["mode"] == "session":
            sort_markets_for_replay(markets, mode="session")
        market_days = count_market_days(markets)

        fees = _fee_policy(manifest)
        job = {
            "entry": manifest["entry"],
            "hooks": {h: h for h in manifest["hooks"]},
            "arities": {"on_market_open": 3, "on_tick": 3, "on_book": 3, "on_trade": 3, "on_settle": 4, "on_timer": 3},
            "params": manifest["params"],
            "mode": manifest["mode"],
            "seed": int(seed),
            "fees": fees,
            "resting": resting_policy_for(manifest, venue),
            **({"engine": policy} if policy else {}),
            "limits": CONTRACT["limits"],
            "fillDelayMs": manifest.get("latency") or 0,
        }

        def lines_of(m):
            with open(m["events_file"], encoding="utf-8") as fh:
                return [line.strip() for line in fh if line.strip()]

        def header_of(m):
            return {"market": m["market"], "stream": m["stream"]}

        if manifest["mode"] == "session":
            stream = session_lines(markets, lines_of=lines_of, header_of=header_of,
                                   timed=bool(policy and policy.get("scheduler") == "global"))
        else:
            stream = market_lines(markets, lines_of=lines_of, header_of=header_of)
        it = iter(stream)

        def next_line():
            return next(it, None)

        out_lines = []

        def emit(channel, payload):
            out_lines.append((channel, payload))

        def load_class():
            klass = get_class()
            for h in manifest["hooks"]:
                if not callable(getattr(klass, h, None)):
                    raise RunAbort("E_HOOK_SIG", f"{h} was declared but {h}() is not defined on the class")
            return klass

        # Replay is parallel only when asked for (see `workers` in run()):
        # decoding never runs strategy code, replaying does.
        parallel = (workers is not None and manifest["mode"] == "market" and n_workers > 1
                    and len(markets) > 1 and fork_context() is not None)
        if parallel:
            try:
                code, out_lines = replay_markets(
                    workers=min(n_workers, len(markets)), markets=markets, job=job, load_class=load_class,
                    get_class=get_class, lines_of=lines_of, header_of=header_of, run_job=run_job,
                    dumps=_feed_dumps, engine=_ENGINE,
                    channels={"log": CHANNEL_LOG, "result": CHANNEL_RESULT, "progress": CHANNEL_PROGRESS})
            except SharedState:
                # The strategy keeps state on its class or in params, which a
                # sequential run carries across markets: replay it that way.
                parallel = False
        if not parallel:
            code = run_job(job, load_class, next_line, emit)
        trades, fills, logs, result, malformed = _demux(out_lines)
        if code != EXIT_OK:
            rej = result.get("rejection") or {"code": "E_RUNTIME", "detail": f"exited {code}"}
            raise BacktestError(rej.get("code", "E_RUNTIME"), rej.get("detail", ""))
        if result["markets_run"] < len(markets):
            raise BacktestError("E_RUNTIME", f"only {result['markets_run']} of {len(markets)} market(s) were replayed")

        meta = {}
        for m in markets:
            mk = m["market"]
            meta[mk["market_id"]] = {
                "market_id": mk["market_id"], "asset": mk.get("asset"), "interval": mk.get("interval"),
                "outcome": mk.get("outcome"), "up_px": m.get("up_px"), "down_px": m.get("down_px"),
                "stream": m.get("stream"),
            }
        delay = manifest.get("latency") or 0
        summary = fill_stats(fills)
        report = build_report(
            run_id=f"local_{want_days[0]}",
            submitted_at=0,
            manifest={"schema": manifest["schema"], "language": manifest["language"], "mode": manifest["mode"]},
            source_sha256=None,
            scope={
                "venue": venue,
                "assets": asset_list or list(dict.fromkeys(m["market"]["asset"] for m in markets if m["market"].get("asset"))),
                "from": want_days[0],
                "to": want_days[-1],
                "market_days": market_days,
                "archived_day_count": market_days,
            },
            scanned={"markets": result["markets_run"], "market_days": market_days, "events": result["events_seen"]},
            trades=trades,
            fill_summary=summary,
            maker=maker_report(stats=result.get("maker"), fills=summary["makerFills"],
                               queue_model=_ENGINE["otmaker"].queue_model_name((policy or {}).get("queue_prob")),
                               queue_prob=(policy or {}).get("queue_prob"),
                               print_lag_ms=_ENGINE["otmaker"].PRINT_LAG_MS),
            orders=orders_report(policy=policy, stats=result.get("orders")),
            timers=result.get("timers") if policy and policy.get("timers") else None,
            market_summaries=list(meta.values()),
            market_meta=meta,
            fees_paid=result["fees_paid"],
            fill_delay_ms=delay,
            fee_model=fee_model_report(policy=fees, markets=[m["market"] for m in markets]),
            sweep=None,
            crosschecks=result["crosschecks"],
            seed=int(seed),
            coverage=build_coverage(
                market_days_requested=None,
                market_days_scanned=market_days,
                markets_reported_by_runner=result["markets_run"],
                missing=missing,
                reference_declared=[],
                streams=count_streams(list(meta.values())),
                dropped_rows=malformed,
                **{k: v for k, v in bbo_coverage(venue=venue, datasets=manifest["datasets"], markets=markets,
                                                 applied=bbo_applied).items()},
                local=True,
                source=os.path.abspath(root),
            ),
            budget=result["budget"],
        )
        if book_throttle is not None and report.get("coverage") is not None:
            gate = BookThrottle(venue=venue, assets=scope_assets, frm=want_days[0], to=want_days[-1])
            report["coverage"]["book_throttle"] = ("off" if book_throttle is False
                                                  else {a: gate.window_for(a) for a in scope_assets})
        return Result(report, trades, fills, logs)
    finally:
        shutil.rmtree(events_dir, ignore_errors=True)


def _run_generic(manifest, get_class, *, data, venue, days, seed, workers, mark_max_age_ms) -> Result:
    """Generic data (generic.py): the same replay and report as an archive run,
    with the generic report fields. MUST MATCH the generic branch of
    cli/commands/run.mjs."""
    _validate(manifest)
    if days:
        raise BacktestError("E_DATA", "days= selects archive days; generic data has no days -- every market is replayed")
    if venue not in (None, "polymarket"):
        raise BacktestError("E_DATA", "generic data is Polymarket CLOB data; venue must be polymarket or None")
    dataset = data if isinstance(data, Dataset) else Dataset.from_dir(os.fspath(data))
    try:
        policy = engine_policy_for(manifest, "polymarket", "generic")
    except ManifestError as err:
        raise BacktestError("E_MANIFEST", str(err)) from None
    events_dir = tempfile.mkdtemp(prefix="ot-events-")
    try:
        try:
            decoded = dataset.decode(events_dir=events_dir, book_factory=lambda: _Book(None),
                                     mark_max_age_ms=mark_max_age_ms,
                                     order_rules=bool(policy and (policy.get("order_rules") or {}).get("mode") == "venue"))
        except DatasetError as err:
            raise BacktestError("E_DATA", str(err)) from None
        markets = decoded["markets"]
        if not markets:
            first = decoded["unusable"][0] if decoded["unusable"] else None
            raise BacktestError("E_DATA", "no market of that dataset could be replayed"
                                + (f" ({first['market_id']}: {first['why']})" if first else ""))
        if manifest["mode"] == "session":
            sort_markets_for_replay(markets, mode="session")
        fees = _fee_policy(manifest)
        job = {
            "entry": manifest["entry"],
            "hooks": {h: h for h in manifest["hooks"]},
            "arities": {"on_market_open": 3, "on_tick": 3, "on_book": 3, "on_trade": 3, "on_settle": 4, "on_timer": 3},
            "params": manifest["params"],
            "mode": manifest["mode"],
            "seed": int(seed),
            "fees": fees,
            "resting": resting_policy_for(manifest, "polymarket"),
            **({"engine": policy} if policy else {}),
            "limits": CONTRACT["limits"],
            "fillDelayMs": manifest.get("latency") or 0,
        }

        def lines_of(m):
            raise AssertionError("generic markets are streamed")

        def cursor_of(m):
            from .generic import file_lines
            return file_lines(m["events_file"])

        def header_of(m):
            return {"market": m["market"], "stream": m["stream"], "terminal": m.get("terminal")}

        if manifest["mode"] == "session":
            stream = session_lines(markets, lines_of=lines_of, header_of=header_of, cursor_of=cursor_of,
                                   timed=bool(policy and policy.get("scheduler") == "global"))
        else:
            stream = market_lines(markets, lines_of=lines_of, header_of=header_of, cursor_of=cursor_of)
        it = iter(stream)

        def next_line():
            return next(it, None)

        out_lines = []

        def emit(channel, payload):
            out_lines.append((channel, payload))

        def load_class():
            klass = get_class()
            for h in manifest["hooks"]:
                if not callable(getattr(klass, h, None)):
                    raise RunAbort("E_HOOK_SIG", f"{h} was declared but {h}() is not defined on the class")
            return klass

        code = run_job(job, load_class, next_line, emit)
        trades, fills, logs, result, malformed = _demux(out_lines)
        if code != EXIT_OK:
            rej = result.get("rejection") or {"code": "E_RUNTIME", "detail": f"exited {code}"}
            raise BacktestError(rej.get("code", "E_RUNTIME"), rej.get("detail", ""))
        if result["markets_run"] < len(markets):
            raise BacktestError("E_RUNTIME", f"only {result['markets_run']} of {len(markets)} market(s) were replayed")

        meta = {}
        for m in markets:
            mk = m["market"]
            meta[mk["market_id"]] = {
                "market_id": mk["market_id"], "asset": mk.get("asset"), "interval": mk.get("interval"),
                "outcome": mk.get("outcome"), "up_px": m.get("up_px"), "down_px": m.get("down_px"),
                "stream": m.get("stream"),
            }
        delay = manifest.get("latency") or 0
        summary = fill_stats(fills)
        n = len(markets)
        report = build_report(
            run_id="local_generic",
            submitted_at=0,
            manifest={"schema": manifest["schema"], "language": manifest["language"], "mode": manifest["mode"]},
            source_sha256=None,
            scope={"venue": "polymarket", "assets": [], "from": None, "to": None,
                   "market_days": n, "archived_day_count": n},
            scanned={"markets": result["markets_run"], "market_days": n, "events": result["events_seen"]},
            trades=trades,
            fill_summary=summary,
            maker=maker_report(stats=result.get("maker"), fills=summary["makerFills"],
                               queue_model=_ENGINE["otmaker"].queue_model_name((policy or {}).get("queue_prob")),
                               queue_prob=(policy or {}).get("queue_prob"),
                               print_lag_ms=_ENGINE["otmaker"].PRINT_LAG_MS),
            orders=orders_report(policy=policy, stats=result.get("orders")),
            timers=result.get("timers") if policy and policy.get("timers") else None,
            market_summaries=list(meta.values()),
            market_meta=meta,
            fees_paid=result["fees_paid"],
            fill_delay_ms=delay,
            fee_model=fee_model_report(policy=fees, markets=[m["market"] for m in markets]),
            sweep=None,
            crosschecks=result["crosschecks"],
            seed=int(seed),
            coverage=build_coverage(
                market_days_requested=None,
                market_days_scanned=n,
                markets_reported_by_runner=result["markets_run"],
                missing=[],
                reference_declared=[],
                streams=count_streams(list(meta.values())),
                dropped_rows=malformed,
                **{k: v for k, v in bbo_coverage(venue="polymarket", datasets=manifest["datasets"], markets=markets,
                                                 applied={}).items()},
                local=True,
                source=os.path.abspath(dataset.root) if dataset.root else None,
            ),
            budget=result["budget"],
        )
        report = apply_generic_extras(report, generic_report_extras(
            trades=trades, markets=markets, unusable=decoded["unusable"], counts=decoded["counts"]))
        return Result(report, trades, fills, logs)
    finally:
        shutil.rmtree(events_dir, ignore_errors=True)


def _demux(lines):
    """Split the harness output by channel, validating rows like `ot run` does."""
    import json

    trades, fills, logs = [], [], []
    malformed = 0
    result = None
    for channel, payload in lines:
        if channel == CHANNEL_LOG:
            logs.append(payload)
            continue
        if channel == CHANNEL_PROGRESS:
            continue
        if channel == CHANNEL_RESULT:
            try:
                result = json.loads(payload)
            except ValueError:
                malformed += 1
            continue
        try:
            raw = json.loads(payload)
        except ValueError:
            malformed += 1
            continue
        row = parse_trade(raw) if channel == CHANNEL_TRADE else parse_fill(raw) if channel == CHANNEL_FILL else None
        if row is None:
            malformed += 1
            continue
        (trades if channel == CHANNEL_TRADE else fills).append(row)
    return trades, fills, "\n".join(logs), _parse_result(result or {}), malformed


def _order_stats(raw, fin):
    """parseOrderStats in runner/harness/protocol.mjs."""
    if not isinstance(raw, dict):
        return None
    rj = raw.get("rejected") if isinstance(raw.get("rejected"), dict) else {}
    rejected = {k: rj[k] for k in REJECT_REASONS if fin(rj.get(k)) and rj[k] > 0}
    return {"rejected": rejected, "rebates": raw["rebates"] if fin(raw.get("rebates")) else 0}


def _parse_result(r: dict) -> dict:
    def fin(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")

    cc = [c for c in (r.get("crosschecks") or [])[:100_000] if isinstance(c, dict)]
    return {
        "markets_run": r["markets_run"] if fin(r.get("markets_run")) else 0,
        "events_seen": r["events_seen"] if fin(r.get("events_seen")) else 0,
        "fees_paid": r["fees_paid"] if fin(r.get("fees_paid")) else 0,
        "budget": r["budget"] if isinstance(r.get("budget"), dict) else None,
        "maker": _maker_stats(r.get("maker")),
        "orders": _order_stats(r.get("orders"), fin),
        "timers": ({k: (r["timers"].get(k) if fin(r["timers"].get(k)) else 0) for k in ("created", "fired", "dropped")}
                   if isinstance(r.get("timers"), dict) else None),
        "crosschecks": [{
            "market_id": c["market_id"] if isinstance(c.get("market_id"), str) else None,
            "claimed": c.get("claimed"),
            "official": c.get("official"),
            "match": bool(c.get("match")),
        } for c in cc],
        "rejection": ({"code": str(r["rejection"].get("code", "E_RUNTIME")),
                       "detail": str(r["rejection"].get("detail", ""))[:4000]}
                      if isinstance(r.get("rejection"), dict) else None),
    }
