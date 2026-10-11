"""Hand-computed answers for outcometick.backtest.

client/backtest/equivalence.test.mjs proves this package agrees with the queue
and `ot run`. Agreement alone would pass two implementations that are wrong the
same way, so these cases are worked out by hand from the published rules:
fills walk the resting depth, the venue's taker fee is C x rate x p x (1 - p)
rounded to 5 decimals per level taken, a market settles at $1 / $0 off its own
settlement stream, and a hook never sees what happens later.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
import unittest

from outcometick import Order, Strategy
from outcometick.backtest import BacktestError, run, sweep

T0 = 1788825600000  # 2026-09-08T00:00:00Z
UP_TOKEN, DOWN_TOKEN = "1" * 70, "2" * 70


def _write(root, rel, rows, csv_header=None, gz=True):
    if not gz:
        rel = rel[:-3] if rel.endswith(".gz") else rel
    path = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with (gzip.open(path, "wt", encoding="utf-8") if gz else open(path, "w", encoding="utf-8")) as fh:
        if csv_header:
            fh.write(",".join(csv_header) + "\n")
            for r in rows:
                fh.write(",".join(str(x) for x in r) + "\n")
        else:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


def build_archive(root, *, outcome="UP", fee=True, asks=((0.50, 60), (0.52, 40)), gz=True):
    """One BTC 5-minute market, 00:00-00:05, settling on the 60s TWAP."""
    day = "2026-09-08"
    raw = {"cryptoMarketConfig": {"twapLookbackSeconds": 60}}
    if fee:
        raw.update(feesEnabled=True, feeSchedule={"rate": 0.07, "exponent": 1, "takerOnly": True, "rebateRate": 0.2})
    else:
        raw.update(feesEnabled=False)
    _write(root, f"data/polymarket/daily/markets/BTC-5m/BTC-5m-markets-{day}.jsonl.gz", [{
        "slug": "btc-updown-5m-t", "asset": "btc", "interval_sec": 300, "condition_id": "0x" + "ab" * 32,
        "token_ids": [UP_TOKEN, DOWN_TOKEN], "start_sec": T0 // 1000, "end_sec": T0 // 1000 + 300,
        "resolved": True, "outcome_prices": ["1", "0"] if outcome == "UP" else ["0", "1"],
        "strike_value": "79104869058460346482688", "raw": raw,
    }], gz=gz)
    book = lambda ts, token, a, b: {  # noqa: E731
        "slug": "btc-updown-5m-t", "asset_id": token, "event_type": "book", "event_ts_ms": ts, "recv_ms": ts,
        "payload": {"asks": [{"price": str(p), "size": str(s)} for p, s in a],
                    "bids": [{"price": str(p), "size": str(s)} for p, s in b]}}
    _write(root, f"data/polymarket/daily/book/BTC-5m/BTC-5m-book-{day}.jsonl.gz", [
        book(T0 + 1000, UP_TOKEN, asks, [(0.48, 100)]),
        book(T0 + 1000, DOWN_TOKEN, [(0.53, 100)], [(0.47, 100)]),
        # A later, much dearer book: only a strategy that waited could pay this.
        book(T0 + 200_000, UP_TOKEN, [(0.90, 500)], [(0.88, 500)]),
    ], gz=gz)
    _write(root, f"data/chainlink-twap-60s/daily/prices/BTCUSD/BTCUSD-twap60s-prices-{day}.csv.gz",
           # Ticks from 5 s in: the first one lands after the opening book.
           [[T0 + s * 1000, 79100 + s, "0", T0 + s * 1000, T0 + s * 1000] for s in range(5, 301, 10)],
           csv_header=["feed_ts_ms", "value", "full_accuracy_value", "server_ts_ms", "recv_ms"], gz=gz)
    return root


class BuyOnce(Strategy):
    def on_market_open(self, ctx, market):
        self.done = False

    def on_tick(self, ctx, tick):
        if self.done:
            return None
        self.done = True
        return Order(side="UP", size=100, limit=0.99)


class Cases(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="ot-bt-")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_fill_fee_and_settlement_by_hand(self):
        r = run(BuyOnce, build_archive(self.root), assets=["BTC"])
        self.assertEqual(len(r.fills), 1)
        fill = r.fills[0]
        # 60 at 0.50 and 40 at 0.52 -- the depth resting at the first tick.
        self.assertEqual(fill["filled"], 100)
        self.assertAlmostEqual(fill["avg_px"], 0.508, places=12)
        self.assertEqual(fill["levels_walked"], 2)
        # 60 x 0.07 x 0.50 x 0.50 = 1.05 ; 40 x 0.07 x 0.52 x 0.48 = 0.69888
        self.assertAlmostEqual(fill["fee"], 1.74888, places=12)
        (trade,) = r.trades
        self.assertEqual(trade["how"], "settled")
        self.assertEqual(trade["outcome"], "UP")
        # 100 contracts pay $1 each: 100 - 50.8 - 1.74888
        self.assertAlmostEqual(trade["pnl"], 47.45112, places=9)
        self.assertEqual(r.metrics["net_pnl"], 47.45)
        self.assertEqual(r.metrics["fees"], -1.75)
        self.assertEqual(r.report["fee_model"]["mode"], "venue")
        self.assertEqual(r.report["fee_model"]["markets"], {"charged": 1, "known_zero": 0, "unknown": 0})

    def test_losing_side_settles_at_zero(self):
        r = run(BuyOnce, build_archive(self.root, outcome="DOWN"), assets=["BTC"])
        self.assertAlmostEqual(r.trades[0]["pnl"], -50.8 - 1.74888, places=9)

    def test_known_zero_fee_and_flat_override(self):
        r = run(BuyOnce, build_archive(self.root, fee=False), assets=["BTC"])
        self.assertEqual(r.fills[0]["fee"], 0)
        self.assertEqual(r.report["fee_model"]["markets"]["known_zero"], 1)
        shutil.rmtree(self.root)
        os.makedirs(self.root)
        r = run(BuyOnce, build_archive(self.root), assets=["BTC"], fee_bps=0)
        self.assertEqual(r.fills[0]["fee"], 0)
        self.assertAlmostEqual(r.trades[0]["pnl"], 49.2, places=9)
        self.assertEqual(r.report["fee_model"], {"mode": "bps", "bps": 0, "model": None, "rounding": None,
                                                 "estimate": False, "markets": None})

    def test_a_delayed_order_pays_the_book_that_exists_when_it_lands(self):
        # 250 ms of latency still lands long before the dearer 200 s book.
        r = run(BuyOnce, build_archive(self.root), assets=["BTC"], latency_ms=250)
        self.assertAlmostEqual(r.fills[0]["avg_px"], 0.508, places=12)
        self.assertEqual(r.report["fill_delay_ms"], 250)

    def test_the_outcome_is_hidden_until_settlement(self):
        seen = {}

        class Peek(Strategy):
            def on_market_open(self, ctx, market):
                seen["open"] = "outcome" in market

            def on_tick(self, ctx, tick):
                return None

            def on_settle(self, ctx, market, outcome):
                seen["settle"] = outcome

        run(Peek, build_archive(self.root), assets=["BTC"])
        self.assertEqual(seen, {"open": False, "settle": "UP"})

    def test_a_strategy_that_cannot_trade_is_refused(self):
        class Silent(Strategy):
            def on_settle(self, ctx, market, outcome):
                pass

        with self.assertRaises(BacktestError) as cm:
            run(Silent, build_archive(self.root), assets=["BTC"])
        self.assertEqual(cm.exception.code, "E_MANIFEST")

    def test_what_a_queued_run_would_refuse_is_refused_here(self):
        archive = build_archive(self.root)
        for kw in ({"datasets": ["settlement", "bookz"]}, {"latency_ms": -5}, {"latency_ms": 10**6},
                   {"fee_bps": 5000}, {"mode": "portfolio"}, {"hooks": ["on_tick", "on_quote"]}):
            with self.assertRaises(BacktestError, msg=str(kw)) as cm:
                run(BuyOnce, archive, assets=["BTC"], **kw)
            self.assertEqual(cm.exception.code, "E_MANIFEST", kw)

    def test_a_cached_decode_replays_identically_and_follows_the_files(self):
        cache = os.path.join(self.root, "cache")
        archive = build_archive(os.path.join(self.root, "a"))
        cold = run(BuyOnce, archive, assets=["BTC"], cache=cache)
        self.assertEqual(len(os.listdir(cache)), 1)
        warm = run(BuyOnce, archive, assets=["BTC"], cache=cache)
        self.assertEqual(len(os.listdir(cache)), 1, "the second run decoded again")
        bare = run(BuyOnce, archive, assets=["BTC"], cache=False)
        for r in (warm, bare):
            self.assertEqual(r.trades, cold.trades)
            self.assertEqual(r.fills, cold.fills)
            self.assertEqual({k: v for k, v in r.report.items() if k != "budget"},
                             {k: v for k, v in cold.report.items() if k != "budget"})
        # A changed archive file is a different decode: the dearer book is now first.
        build_archive(os.path.join(self.root, "a"), asks=((0.60, 100),))
        changed = run(BuyOnce, archive, assets=["BTC"], cache=cache)
        self.assertEqual(len(os.listdir(cache)), 2)
        self.assertAlmostEqual(changed.fills[0]["avg_px"], 0.60, places=12)

    def test_a_template_with_a_dataclass_loads_as_it_does_in_the_sandbox(self):
        from outcometick.backtest import run_template

        tpl = os.path.join(self.root, "tpl")
        os.makedirs(tpl)
        with open(os.path.join(tpl, "outcometick.json"), "w") as fh:
            json.dump({"schema": 1, "language": "python@3.14", "entry": "dc_strategy.py:S",
                       "hooks": ["on_market_open", "on_tick"], "datasets": ["settlement", "book"]}, fh)
        with open(os.path.join(tpl, "dc_strategy.py"), "w") as fh:
            fh.write("from __future__ import annotations\n"
                     "from dataclasses import dataclass\n"
                     "from outcometick import Strategy, Order\n\n\n"
                     "@dataclass\nclass Sizing:\n    size: int = 100\n\n\n"
                     "class S(Strategy):\n"
                     "    def on_market_open(self, ctx, market):\n        self.done = False\n\n"
                     "    def on_tick(self, ctx, tick):\n"
                     "        if self.done:\n            return None\n"
                     "        self.done = True\n"
                     "        return Order(side='UP', size=Sizing().size, limit=0.99)\n")
        r = run_template(tpl, build_archive(os.path.join(self.root, "a")), assets=["BTC"], cache=False)
        self.assertEqual(r.fills[0]["filled"], 100)

    def test_a_template_with_custom_series_is_refused_not_run_without_them(self):
        from outcometick.backtest import run_template

        tpl = os.path.join(self.root, "tpl-series")
        os.makedirs(tpl)
        with open(os.path.join(tpl, "outcometick.json"), "w") as fh:
            json.dump({"schema": 1, "language": "python@3.14", "entry": "s.py:S", "hooks": ["on_tick"],
                       "datasets": ["settlement", "book"], "series": [{"name": "sig", "file": "sig.csv"}]}, fh)
        with open(os.path.join(tpl, "s.py"), "w") as fh:
            fh.write("class S:\n    def on_tick(self, ctx, tick):\n        return None\n")
        with self.assertRaises(BacktestError) as cm:
            run_template(tpl, build_archive(os.path.join(self.root, "a")), assets=["BTC"], cache=False)
        self.assertEqual(cm.exception.code, "E_MANIFEST")

    def test_two_archives_with_identical_metadata_do_not_share_a_cache_entry(self):
        cache = os.path.join(self.root, "cache")
        # Uncompressed, so changing one digit (0.50 -> 0.60) leaves every file
        # the same size; the timestamps are copied over below. Only bytes differ.
        a = build_archive(os.path.join(self.root, "a"), gz=False)
        b = build_archive(os.path.join(self.root, "b"), asks=((0.60, 60), (0.52, 40)), gz=False)
        same_size = True
        for dirpath, _, names in os.walk(a):
            for n in names:
                pa = os.path.join(dirpath, n)
                pb = os.path.join(b, os.path.relpath(pa, a))
                same_size &= os.path.getsize(pa) == os.path.getsize(pb)
                st = os.stat(pa)
                os.utime(pb, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertTrue(same_size, "fixture drift: the two archives must have equal file sizes to test this")
        ra = run(BuyOnce, a, assets=["BTC"], cache=cache)
        rb = run(BuyOnce, b, assets=["BTC"], cache=cache)
        self.assertAlmostEqual(ra.fills[0]["avg_px"], 0.508, places=12)
        self.assertAlmostEqual(rb.fills[0]["avg_px"], (0.60 * 60 + 0.52 * 40) / 100, places=12)

    def test_files_are_found_in_byte_order_like_node_readdir(self):
        # The order files are found in decides a day's market order and the
        # dropped-market list; Node's readdir is sorted, os.scandir is not.
        from outcometick.backtest.local import walk

        d = os.path.join(self.root, "w")
        for name in ("b.jsonl", "a.jsonl", "C.jsonl", "_x.jsonl", "c.jsonl"):
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, name), "w").close()
        self.assertEqual(walk(d), ["C.jsonl", "_x.jsonl", "a.jsonl", "b.jsonl", "c.jsonl"])

    def test_print_still_works_in_a_notebook(self):
        # The sandbox harness discards print(); the local runner must not.
        import io
        from contextlib import redirect_stdout

        class Loud(BuyOnce):
            def on_settle(self, ctx, market, outcome):
                print("settled", outcome)

        buf = io.StringIO()
        with redirect_stdout(buf):
            run(Loud, build_archive(self.root), assets=["BTC"])
        self.assertIn("settled UP", buf.getvalue())


if __name__ == "__main__":
    unittest.main()


def build_multi_archive(root, markets=6, day="2026-09-08", t0=T0):
    """`markets` consecutive BTC 5-minute markets, each with a book and its ticks."""
    T0 = t0  # noqa: N806 -- the module constant, shifted for another day
    raw = {"cryptoMarketConfig": {"twapLookbackSeconds": 60}, "feesEnabled": True,
           "feeSchedule": {"rate": 0.07, "exponent": 1, "takerOnly": True, "rebateRate": 0.2}}
    rows, books = [], []
    for i in range(markets):
        start = T0 + i * 300_000
        slug = f"btc-updown-5m-m{i}"
        up, down = str(i + 1) * 70, str(i + 1) * 69 + "9"
        rows.append({"slug": slug, "asset": "btc", "interval_sec": 300, "condition_id": "0x" + f"{i:02x}" * 32,
                     "token_ids": [up, down], "start_sec": start // 1000, "end_sec": start // 1000 + 300,
                     "resolved": True, "outcome_prices": ["1", "0"] if i % 2 == 0 else ["0", "1"],
                     "strike_value": "79104869058460346482688", "raw": raw})
        for token, a, b in ((up, [(0.50, 60)], [(0.48, 100)]), (down, [(0.53, 100)], [(0.47, 100)])):
            books.append({"slug": slug, "asset_id": token, "event_type": "book", "event_ts_ms": start + 1000,
                          "recv_ms": start + 1000,
                          "payload": {"asks": [{"price": str(p), "size": str(s)} for p, s in a],
                                      "bids": [{"price": str(p), "size": str(s)} for p, s in b]}})
    _write(root, f"data/polymarket/daily/markets/BTC-5m/BTC-5m-markets-{day}.jsonl.gz", rows)
    _write(root, f"data/polymarket/daily/book/BTC-5m/BTC-5m-book-{day}.jsonl.gz", books)
    _write(root, f"data/chainlink-twap-60s/daily/prices/BTCUSD/BTCUSD-twap60s-prices-{day}.csv.gz",
           [[T0 + s * 1000, 79100 + s % 50, "0", T0 + s * 1000, T0 + s * 1000] for s in range(5, markets * 300, 7)],
           csv_header=["feed_ts_ms", "value", "full_accuracy_value", "server_ts_ms", "recv_ms"])
    return root


class Chatty(Strategy):
    """Logs every tick — with trailing whitespace, which the harness strips from
    what it prints but still charges to the log budget — and trades once a market."""

    def on_market_open(self, ctx, market):
        self.done = False

    def on_tick(self, ctx, tick):
        ctx.log(f"tick {tick.value} {'x' * (tick.ts_ms % 40)}   ")
        if self.done:
            return None
        self.done = True
        return Order(side="UP" if tick.value % 2 else "DOWN", size=10, limit=0.99)


class Parallel(unittest.TestCase):
    """workers > 1 replays markets on separate processes; the result must be the
    sequential one, byte for byte — rows, totals and the run-wide log budget."""

    def setUp(self):
        self.root = build_multi_archive(tempfile.mkdtemp(prefix="ot-bt-par-"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def both(self, klass, **kw):
        seq = run(klass, self.root, assets=["BTC"], workers=1, cache=False, **kw)
        par = run(klass, self.root, assets=["BTC"], workers=4, cache=False, **kw)
        return seq, par

    def same(self, seq, par):
        strip = lambda r: {k: v for k, v in r.report.items() if k != "budget"}  # noqa: E731
        self.assertEqual(strip(seq), strip(par))
        self.assertEqual(seq.trades, par.trades)
        self.assertEqual(seq.fills, par.fills)
        self.assertEqual(seq.logs, par.logs)

    def test_rows_and_totals_match_the_sequential_run(self):
        seq, par = self.both(Chatty)
        self.assertEqual(seq.report["scanned"]["markets"], 6)
        self.assertGreater(len(seq.trades), 3)
        self.same(seq, par)

    def test_the_run_wide_log_budget_is_cut_at_the_same_line(self):
        from outcometick.backtest._contract import CONTRACT
        limits = CONTRACT["limits"]
        saved = limits["logBytesPerRun"]
        # Small enough to run out in the third market or so.
        limits["logBytesPerRun"] = 6000
        try:
            seq, par = self.both(Chatty)
        finally:
            limits["logBytesPerRun"] = saved
        self.assertIn("[runner] log budget spent", seq.logs)
        self.same(seq, par)


class SizedSide(Strategy):
    """One order per market, side and size from params."""

    def on_market_open(self, ctx, market):
        self.done = False

    def on_tick(self, ctx, tick):
        if self.done:
            return None
        self.done = True
        return Order(side=ctx.p.side, size=ctx.p.size, limit=0.99)


class Sweeps(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="ot-bt-sweep-")
        build_multi_archive(self.root, markets=4, day="2026-09-08", t0=T0)
        build_multi_archive(self.root, markets=4, day="2026-09-09", t0=T0 + 86_400_000)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_cells_are_plain_runs_ranked_with_a_held_out_replay(self):
        grid = {"side": ["UP", "DOWN"], "size": [5, 10]}
        s = sweep(SizedSide, self.root, grid, assets=["BTC"], holdout=0.5, top=2, workers=2)
        self.assertEqual(s.in_days, ["2026-09-08"])
        self.assertEqual(s.out_days, ["2026-09-09"])
        self.assertEqual(len(s.cells), 4)
        for c in s.cells:
            direct = run(SizedSide, self.root, assets=["BTC"], days=["2026-09-08"], params=c["params"], workers=1)
            self.assertEqual(c["metrics"], direct.metrics, c["params"])
        pnls = [c["metrics"]["net_pnl"] for c in s.ranked]
        self.assertEqual(pnls, sorted(pnls, reverse=True))
        held = [c for c in s.ranked if c["holdout"] is not None]
        self.assertEqual([c["params"] for c in held], [c["params"] for c in s.ranked[:2]])
        for c in held:
            direct = run(SizedSide, self.root, assets=["BTC"], days=["2026-09-09"], params=c["params"], workers=1)
            self.assertEqual(c["holdout"], direct.metrics)
        self.assertIn("held out 2026-09-09", s.summary())

    def test_smaller_is_better_metrics_rank_ascending(self):
        s = sweep(SizedSide, self.root, {"side": ["UP", "DOWN"], "size": [5, 10]}, assets=["BTC"],
                  metric="max_drawdown_abs")
        vals = [c["metrics"]["max_drawdown_abs"] for c in s.ranked]
        self.assertEqual(vals, sorted(vals))
        self.assertIn("No days were held out", s.summary())

    def test_bad_sweeps_are_refused(self):
        with self.assertRaises(BacktestError):
            sweep(SizedSide, self.root, {}, assets=["BTC"])
        with self.assertRaises(BacktestError):
            sweep(SizedSide, self.root, {"size": []}, assets=["BTC"])
        with self.assertRaises(BacktestError):
            sweep(SizedSide, self.root, {"size": [5]}, assets=["BTC"], params={"side": "UP"}, holdout=0.99)
        with self.assertRaises(BacktestError):
            sweep(SizedSide, self.root, {"size": [5]}, assets=["BTC"], params={"side": "UP"}, metric="nope")


class CountsMarkets(Strategy):
    """Keeps a counter on the CLASS — outside the market-mode contract, but a
    sequential run carries it across markets, so a parallel one must too."""
    seen = 0

    def on_market_open(self, ctx, market):
        type(self).seen += 1
        self.done = False

    def on_tick(self, ctx, tick):
        if self.done:
            return None
        self.done = True
        return Order(side="UP" if type(self).seen % 3 else "DOWN", size=type(self).seen, limit=0.99)


class Once:
    """A log message that may be formatted exactly once, as Ctx.log formats it."""

    def __init__(self, n):
        self.n = n
        self.formatted = False

    def __str__(self):
        if self.formatted:
            raise RuntimeError("formatted twice")
        self.formatted = True
        return f"once#{self.n}  "


class LogsObjects(Strategy):
    def on_market_open(self, ctx, market):
        self.k = 0

    def on_tick(self, ctx, tick):
        self.k += 1
        ctx.log(Once(self.k))
        return None


class MutatesParams(Strategy):
    def on_market_open(self, ctx, market):
        ctx.p.seen.append(1)
        self.done = False

    def on_tick(self, ctx, tick):
        if self.done:
            return None
        self.done = True
        return Order(side="UP", size=len(ctx.p.seen), limit=0.99)


class ParallelContract(unittest.TestCase):
    def setUp(self):
        self.root = build_multi_archive(tempfile.mkdtemp(prefix="ot-bt-par2-"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def same(self, seq, par):
        strip = lambda r: {k: v for k, v in r.report.items() if k != "budget"}  # noqa: E731
        self.assertEqual(strip(seq), strip(par))
        self.assertEqual(seq.trades, par.trades)
        self.assertEqual(seq.fills, par.fills)
        self.assertEqual(seq.logs, par.logs)

    def test_state_on_the_class_falls_back_to_a_sequential_replay(self):
        CountsMarkets.seen = 0
        seq = run(CountsMarkets, self.root, assets=["BTC"], workers=1, cache=False)
        CountsMarkets.seen = 0
        par = run(CountsMarkets, self.root, assets=["BTC"], workers=4, cache=False)
        self.assertGreater(len({t["size"] for t in seq.trades}), 3, "the counter did not carry across markets")
        self.same(seq, par)

    def test_mutable_params_fall_back_to_a_sequential_replay(self):
        seq = run(MutatesParams, self.root, assets=["BTC"], workers=1, cache=False, params={"seen": []})
        par = run(MutatesParams, self.root, assets=["BTC"], workers=4, cache=False, params={"seen": []})
        self.same(seq, par)

    def test_a_log_message_is_formatted_once_and_the_parent_is_left_alone(self):
        from outcometick.backtest import _ENGINE
        before = _ENGINE["otreplay"].Ctx.log
        seq = run(LogsObjects, self.root, assets=["BTC"], workers=1, cache=False)
        par = run(LogsObjects, self.root, assets=["BTC"], workers=4, cache=False)
        # The cost of a line is read off the line Ctx.log stored: a second
        # str(msg) would raise in Once and fail the run.
        self.assertIs(_ENGINE["otreplay"].Ctx.log, before)
        self.assertIn("once#1", seq.logs)
        self.same(seq, par)


class Flag:
    """Formatting this message flips the strategy's own side — a side effect
    Ctx.log stops having once the run's log budget is spent."""

    def __init__(self, owner):
        self.owner = owner

    def __str__(self):
        self.owner.side = "DOWN" if self.owner.side == "UP" else "UP"
        return "x" * 200


class TradesOnItsLog(Strategy):
    def on_market_open(self, ctx, market):
        self.side, self.n = "UP", 0

    def on_tick(self, ctx, tick):
        self.n += 1
        ctx.log(Flag(self))
        if self.n == 5:
            return Order(side=self.side, size=10, limit=0.99)
        return None


class ParallelBudget(unittest.TestCase):
    def setUp(self):
        self.root = build_multi_archive(tempfile.mkdtemp(prefix="ot-bt-par3-"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_a_spent_log_budget_changes_behaviour_and_the_run_falls_back(self):
        from outcometick.backtest._contract import CONTRACT
        limits = CONTRACT["limits"]
        saved = limits["logBytesPerRun"]
        limits["logBytesPerRun"] = 4000  # spent during the second market
        try:
            seq = run(TradesOnItsLog, self.root, assets=["BTC"], workers=1, cache=False)
            par = run(TradesOnItsLog, self.root, assets=["BTC"], workers=4, cache=False)
        finally:
            limits["logBytesPerRun"] = saved
        # Once the budget is gone the side stops flipping, so later markets trade
        # a different side than the first one did.
        self.assertGreater(len({t["side"] for t in seq.trades}), 1)
        self.assertEqual(seq.trades, par.trades)
        self.assertEqual(seq.fills, par.fills)
        self.assertEqual(seq.logs, par.logs)


class HoldsALambda(Strategy):
    """Class state that cannot be pickled, mutated in place across markets."""
    state = {"seen": 0, "callback": lambda: None}

    def on_market_open(self, ctx, market):
        type(self).state["seen"] += 1
        self.done = False

    def on_tick(self, ctx, tick):
        if self.done:
            return None
        self.done = True
        return Order(side="UP", size=type(self).state["seen"], limit=0.99)


def add_price_changes(root, rows, day="2026-09-08"):
    _write(root, f"data/polymarket/daily/price_change/BTC-5m/BTC-5m-price_change-{day}.jsonl.gz", rows)


def price_change(ts, changes, slug="btc-updown-5m-m0"):
    return {"slug": slug, "asset_id": None, "event_type": "price_change", "event_ts_ms": ts, "recv_ms": ts,
            "payload": {"price_changes": changes}}


class ParallelEdges(unittest.TestCase):
    def setUp(self):
        self.root = build_multi_archive(tempfile.mkdtemp(prefix="ot-bt-par4-"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_unpicklable_class_state_runs_sequentially(self):
        HoldsALambda.state["seen"] = 0
        seq = run(HoldsALambda, self.root, assets=["BTC"], workers=1, cache=False)
        HoldsALambda.state["seen"] = 0
        par = run(HoldsALambda, self.root, assets=["BTC"], workers=4, cache=False)
        self.assertEqual(seq.trades, par.trades)
        self.assertGreater(len({t["size"] for t in seq.trades}), 3)

    def test_a_bad_row_the_throttle_drops_is_dropped_in_parallel_too(self):
        good = {"asset_id": "1" * 70, "side": "BUY", "price": "0.49", "size": "10", "best_bid": "0.49", "best_ask": "0.50"}
        # BTC's throttle on this day is 20 ms: the second row is dropped by it
        # before its payload is read — sequentially it never fails.
        add_price_changes(self.root, [price_change(T0 + 2000, [good]), price_change(T0 + 2005, [good, None]),
                                      price_change(T0 + 3000, [good])])
        seq = run(BuyOnce, self.root, assets=["BTC"], workers=1, cache=False)
        par = run(BuyOnce, self.root, assets=["BTC"], workers=4, cache=False)
        self.assertEqual(seq.fills, par.fills)

    def test_a_bad_row_the_throttle_keeps_fails_both_ways(self):
        good = {"asset_id": "1" * 70, "side": "BUY", "price": "0.49", "size": "10", "best_bid": "0.49", "best_ask": "0.50"}
        add_price_changes(self.root, [price_change(T0 + 2000, [good]), price_change(T0 + 2500, [good, None])])
        errors = []
        for w in (1, 4):
            with self.assertRaises(Exception) as cm:
                run(BuyOnce, self.root, assets=["BTC"], workers=w, cache=False)
            errors.append(type(cm.exception))
        self.assertEqual(errors[0], errors[1])


class NestedState(Strategy):
    """State on a class defined inside the strategy."""

    class State:
        seen = 0

    def on_market_open(self, ctx, market):
        type(self).State.seen += 1
        self.done = False

    def on_tick(self, ctx, tick):
        if self.done:
            return None
        self.done = True
        return Order(side="UP", size=type(self).State.seen, limit=0.99)


class ParallelNested(unittest.TestCase):
    def setUp(self):
        self.root = build_multi_archive(tempfile.mkdtemp(prefix="ot-bt-par5-"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_state_on_a_nested_class_falls_back_too(self):
        NestedState.State.seen = 0
        seq = run(NestedState, self.root, assets=["BTC"], workers=1, cache=False)
        NestedState.State.seen = 0
        par = run(NestedState, self.root, assets=["BTC"], workers=4, cache=False)
        self.assertGreater(len({t["size"] for t in seq.trades}), 3)
        self.assertEqual(seq.trades, par.trades)

    def test_by_default_only_decoding_is_parallel(self):
        CountsMarkets.seen = 0
        seq = run(CountsMarkets, self.root, assets=["BTC"], workers=1, cache=False)
        CountsMarkets.seen = 0
        default = run(CountsMarkets, self.root, assets=["BTC"], cache=False)
        self.assertEqual(seq.trades, default.trades)


class ConcurrentRuns(unittest.TestCase):
    """Two run() calls at once in one process, each decoding in parallel, must
    not see each other's archive."""

    def test_two_threads_decoding_different_archives(self):
        import threading
        a = build_multi_archive(tempfile.mkdtemp(prefix="ot-bt-ta-"), markets=5)
        b = build_multi_archive(tempfile.mkdtemp(prefix="ot-bt-tb-"), markets=3)
        try:
            want = {k: run(Chatty, root, assets=["BTC"], workers=1, cache=False).trades for k, root in (("a", a), ("b", b))}
            got, errors = {}, []

            def go(k, root):
                try:
                    got[k] = run(Chatty, root, assets=["BTC"], cache=False).trades
                except Exception as err:  # noqa: BLE001
                    errors.append(err)
            threads = [threading.Thread(target=go, args=x) for x in (("a", a), ("b", b))] * 1
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            self.assertEqual(errors, [])
            self.assertEqual(got, want)
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)
