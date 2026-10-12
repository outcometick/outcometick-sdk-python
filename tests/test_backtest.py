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
import sys
import tempfile
import unittest

from outcometick import Order, Strategy
from outcometick.backtest import BacktestError, run, run_template, sweep

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

    def test_fok_takes_all_or_nothing(self):
        class Fok(Strategy):
            def on_market_open(self, ctx, market):
                self.n = 0

            def on_tick(self, ctx, tick):
                self.n += 1
                if self.n == 1:
                    return Order(side="UP", size=101, limit=0.99, tif="fok")   # 100 rests: none of it
                if self.n == 2:
                    return Order(side="UP", size=100, limit=0.99, tif="fok")   # exactly what rests
                return None
        with self.assertRaises(BacktestError):
            run(Fok, build_archive(self.root), assets=["BTC"])                # not declared
        r = run(Fok, self.root, assets=["BTC"], fok=True)
        self.assertEqual([(f["filled"], f.get("reject_reason")) for f in r.fills], [(0, "fok_unfilled"), (100, None)])
        self.assertEqual(r.report["orders"]["rejected"], {"fok_unfilled": 1})

    def test_assumed_order_rules_and_the_report_saying_so(self):
        class Small(Strategy):
            def on_market_open(self, ctx, market):
                self.n = 0

            def on_tick(self, ctx, tick):
                self.n += 1
                if self.n == 1:
                    return Order(side="UP", size=4, limit=0.99)
                if self.n == 2:
                    # A plain order: the runner converts its notional, to 0.01
                    # shares under the rules (an SDK Order converts when built).
                    return {"side": "UP", "notional": 3, "limit": 0.52}     # 5.76 shares, not 5
                return None
        r = run(Small, build_archive(self.root), assets=["BTC"], order_rules="assumed")
        self.assertEqual(r.fills[0]["reject_reason"], "below_min_size")
        self.assertAlmostEqual(r.fills[1]["filled"], 5.76, places=12)
        rules = r.report["orders"]["order_rules"]
        self.assertEqual((rules["mode"], rules["assumed_tick"], rules["assumed_min_size"]), ("assumed", 0.001, 5))
        self.assertNotIn("orders", run(BuyOnce, self.root, assets=["BTC"]).report, "off: no block, old bytes")
        with self.assertRaises(BacktestError):
            run(Small, self.root, assets=["BTC"], order_rules="venue")        # archive markets carry no tick

    def test_timers_wake_the_strategy_and_a_feed_latency_delays_what_it_sees(self):
        class Timed(Strategy):
            def on_market_open(self, ctx, market):
                self.log = []
                ctx.schedule(ctx.now + 30_000, "check")

            def on_tick(self, ctx, tick):
                self.log.append(ctx.now - tick.ts_ms)
                return None

            def on_timer(self, ctx, t):
                ctx.log(f"timer {t.tag} {t.at_ms} {ctx.now}")
                return Order(side="UP", size=10, limit=0.99)
        r = run(Timed, build_archive(self.root), assets=["BTC"])
        self.assertIn(f"timer check {T0 + 30_000} {T0 + 30_000}", r.logs)
        self.assertEqual(r.report["timers"], {"created": 1, "fired": 1, "dropped": 0})
        # The timer's order matched the book resting at its own time: 10 at 0.50.
        self.assertEqual((r.fills[0]["ts_ms"], r.fills[0]["avg_px"]), (T0 + 30_000, 0.5))
        lagged = run(Timed, self.root, assets=["BTC"], feed_latency=250)
        self.assertEqual(lagged.report["timers"]["fired"], 1)
        with self.assertRaises(BacktestError):
            run(Timed, self.root, assets=["BTC"], feed_latency=2.5)

    def test_fingerprint_is_taken_before_the_strategy_runs(self):
        class Wiper(Strategy):
            def on_market_open(self, ctx, market):
                self.done = False

            def on_tick(self, ctx, tick):
                if self.done:
                    return None
                self.done = True
                cfg = ctx.p["cfg"]
                size = 10 if cfg.get("threshold", 0) > 0.5 else 5
                cfg.clear()                       # rewrite its own params mid-run
                return Order(side="UP", size=size, limit=0.99)
        root = build_archive(self.root)
        a = run(Wiper, root, assets=["BTC"], params={"cfg": {"threshold": 0.1}})
        b = run(Wiper, root, assets=["BTC"], params={"cfg": {"threshold": 0.9}})
        self.assertNotEqual(a.fills[0]["filled"], b.fills[0]["filled"])
        self.assertNotEqual(a.report["fingerprint"]["sha256"], b.report["fingerprint"]["sha256"])

    def test_a_strategy_that_rewrites_its_own_files_is_refused(self):
        import json as _json
        d = tempfile.mkdtemp(prefix="ot-self-rewrite-")
        try:
            with open(os.path.join(d, "outcometick.json"), "w") as fh:
                _json.dump({"schema": 1, "language": "python@3.14", "entry": "strategy.py:S",
                            "hooks": ["on_tick"], "datasets": ["settlement", "book"]}, fh)
            with open(os.path.join(d, "strategy.py"), "w") as fh:
                fh.write("import os\n"
                         "from outcometick import Strategy\n"
                         "open(os.path.join(os.path.dirname(__file__), 'outcometick.json'), 'a').write(' ')\n"
                         "class S(Strategy):\n"
                         "    def on_tick(self, ctx, tick):\n"
                         "        return None\n")
            with self.assertRaises(BacktestError) as cm:
                run_template(d, build_archive(self.root))
            self.assertIn("changed while it loaded", str(cm.exception))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_a_stale_bytecode_cache_never_runs_instead_of_the_source(self):
        import json as _json
        import py_compile
        d = tempfile.mkdtemp(prefix="ot-stale-pyc-")
        try:
            with open(os.path.join(d, "outcometick.json"), "w") as fh:
                _json.dump({"schema": 1, "language": "python@3.14", "entry": "strategy.py:S",
                            "hooks": ["on_tick"], "datasets": ["settlement", "book"]}, fh)
            src = ("from outcometick import Order, Strategy\n"
                   "SIZE = {}\n"
                   "class S(Strategy):\n"
                   "    def on_tick(self, ctx, tick):\n"
                   "        if getattr(self, 'done', False):\n"
                   "            return None\n"
                   "        self.done = True\n"
                   "        return Order(side='UP', size=SIZE, limit=0.99)\n")
            path = os.path.join(d, "strategy.py")
            with open(path, "w") as fh:
                fh.write(src.format(5))
            py_compile.compile(path)                       # a cache of the SIZE = 5 version
            st = os.stat(path)
            with open(path, "w") as fh:
                fh.write(src.format(7))                    # same length, same second
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
            r = run_template(d, build_archive(self.root))
            self.assertEqual(r.fills[0]["requested"], 7)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_a_helper_module_edited_between_two_runs_in_one_process_is_reloaded(self):
        import json as _json
        d = tempfile.mkdtemp(prefix="ot-helper-")
        sys.path.insert(0, d)
        try:
            with open(os.path.join(d, "outcometick.json"), "w") as fh:
                _json.dump({"schema": 1, "language": "python@3.14", "entry": "strategy.py:S",
                            "hooks": ["on_tick"], "datasets": ["settlement", "book"]}, fh)
            with open(os.path.join(d, "strategy.py"), "w") as fh:
                fh.write("from outcometick import Order, Strategy\n"
                         "import ot_helper_mod\n"
                         "class S(Strategy):\n"
                         "    def on_tick(self, ctx, tick):\n"
                         "        if getattr(self, 'done', False):\n"
                         "            return None\n"
                         "        self.done = True\n"
                         "        return Order(side='UP', size=ot_helper_mod.SIZE, limit=0.99)\n")
            root = build_archive(self.root)
            sizes, prints = [], []
            for size in (5, 7):
                with open(os.path.join(d, "ot_helper_mod.py"), "w") as fh:
                    fh.write(f"SIZE = {size}\n")
                r = run_template(d, root)
                sizes.append(r.fills[0]["requested"])
                prints.append(r.report["fingerprint"]["sha256"])
            self.assertEqual(sizes, [5, 7])
            self.assertNotEqual(prints[0], prints[1])
        finally:
            sys.path.remove(d)
            sys.modules.pop("ot_helper_mod", None)
            shutil.rmtree(d, ignore_errors=True)

    def test_two_strategy_directories_with_a_same_named_helper_each_run_their_own(self):
        import json as _json
        dirs = [tempfile.mkdtemp(prefix=f"ot-helper-{i}-") for i in range(2)]
        root = build_archive(self.root)
        try:
            sizes = []
            for d, size in zip(dirs, (5, 7)):
                with open(os.path.join(d, "outcometick.json"), "w") as fh:
                    _json.dump({"schema": 1, "language": "python@3.14", "entry": "strategy.py:S",
                                "hooks": ["on_tick"], "datasets": ["settlement", "book"]}, fh)
                with open(os.path.join(d, "ot_shared_helper.py"), "w") as fh:
                    fh.write(f"SIZE = {size}\n")
                with open(os.path.join(d, "strategy.py"), "w") as fh:
                    fh.write("from outcometick import Order, Strategy\n"
                             "import ot_shared_helper\n"
                             "class S(Strategy):\n"
                             "    def on_tick(self, ctx, tick):\n"
                             "        if getattr(self, 'done', False):\n"
                             "            return None\n"
                             "        self.done = True\n"
                             "        return Order(side='UP', size=ot_shared_helper.SIZE, limit=0.99)\n")
                sys.path.insert(0, d)
                sizes.append(run_template(d, root).fills[0]["requested"])
            self.assertEqual(sizes, [5, 7])
        finally:
            for d in dirs:
                if d in sys.path:
                    sys.path.remove(d)
                shutil.rmtree(d, ignore_errors=True)
            sys.modules.pop("ot_shared_helper", None)

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

    def test_execution_option_counters_add_up_the_same_in_parallel(self):
        class Dusty(Strategy):
            def on_market_open(self, ctx, market):
                self.n = 0

            def on_tick(self, ctx, tick):
                self.n += 1
                if self.n in (1, 3):
                    return Order(side="UP", size=4, limit=0.99)     # below the minimum
                if self.n == 2:
                    return Order(side="UP", size=6, limit=0.99, tif="fok")
                return None
        seq, par = self.both(Dusty, fok=True, order_rules="assumed")
        self.assertGreaterEqual(seq.report["orders"]["rejected"]["below_min_size"], 6)
        self.same(seq, par)

    def test_the_record_is_the_same_in_parallel(self):
        class Holder(Strategy):
            def on_market_open(self, ctx, market):
                self.done = False

            def on_tick(self, ctx, tick):
                if self.done:
                    return None
                self.done = True
                return Order(side="UP", size=5, limit=0.99)
        seq, par = self.both(Holder, record_ms=1000)
        self.assertFalse(seq.report["record"]["truncated"])
        self.assertGreater(len(seq.record), 3)
        self.assertEqual(seq.record, par.record)
        self.assertEqual(seq.record_markets, par.record_markets)
        self.same(seq, par)

    def test_timer_counters_add_up_the_same_in_parallel(self):
        class Timed(Strategy):
            def on_market_open(self, ctx, market):
                ctx.schedule(ctx.now + 30_000, "a")
                self.later = ctx.schedule(ctx.now + 60_000, "b")

            def on_tick(self, ctx, tick):
                if self.later:
                    ctx.unschedule(self.later)
                    self.later = None
                return None

            def on_timer(self, ctx, t):
                return Order(side="UP", size=5, limit=0.99)
        seq, par = self.both(Timed)
        self.assertEqual(seq.report["timers"], {"created": 12, "fired": 6, "dropped": 0})
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


class PidSide(SizedSide):
    """SizedSide that leaves a file named after the process each market ran on."""

    def on_market_open(self, ctx, market):
        super().on_market_open(ctx, market)
        open(os.path.join(ctx.p.pids, str(os.getpid())), "w").close()


class CountsCells(SizedSide):
    """Counts its runs on the CLASS and sizes by the count: a sequential sweep
    carries the count from cell to cell, so a parallel one has to as well."""
    calls = 0

    def on_market_open(self, ctx, market):
        super().on_market_open(ctx, market)
        if not getattr(self, "counted", False):
            type(self).calls += 1
            self.counted = True

    def on_tick(self, ctx, tick):
        if self.done:
            return None
        self.done = True
        return Order(side=ctx.p.side, size=ctx.p.size + type(self).calls, limit=0.99)


class FailsOn13(SizedSide):
    def on_tick(self, ctx, tick):
        if ctx.p.size == 13:
            raise ValueError("thirteen")
        return super().on_tick(ctx, tick)


class HangsOn99(FailsOn13):
    def on_tick(self, ctx, tick):
        if ctx.p.size == 99:
            import time
            time.sleep(3600)
        return super().on_tick(ctx, tick)


class DiesOn77(SizedSide):
    """On a forked cell with size 77: start a sleeper, then die without a word."""

    def on_tick(self, ctx, tick):
        if ctx.p.size == 77 and os.getpid() != ctx.p.parent:
            if os.fork() == 0:
                open(os.path.join(ctx.p.pids, str(os.getpid())), "w").close()
                import time
                time.sleep(3600)
                os._exit(0)
            import time
            time.sleep(0.2)  # the sleeper has written its pid
            os._exit(9)
        return super().on_tick(ctx, tick)


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

    def test_parallel_cells_return_the_sequential_sweep(self):
        grid = {"side": ["UP", "DOWN"], "size": [5, 10, 15]}
        pids = tempfile.mkdtemp(prefix="ot-bt-pids-")
        try:
            seq = sweep(PidSide, self.root, grid, assets=["BTC"], holdout=0.5, top=3, params={"pids": pids})
            parent = {str(os.getpid())}
            self.assertEqual(set(os.listdir(pids)), parent)
            par = sweep(PidSide, self.root, grid, assets=["BTC"], holdout=0.5, top=3, params={"pids": pids},
                        cell_workers=3)
            self.assertTrue(set(os.listdir(pids)) - parent, "no cell ran on another process")
        finally:
            shutil.rmtree(pids, ignore_errors=True)
        self.assertEqual(seq.cells, par.cells)
        self.assertEqual([c["params"] for c in seq.ranked], [c["params"] for c in par.ranked])

    def test_state_left_on_the_class_makes_the_rest_of_the_sweep_sequential(self):
        grid = {"size": [5, 6, 7, 8, 9]}
        CountsCells.calls = 0
        seq = sweep(CountsCells, self.root, grid, assets=["BTC"], params={"side": "UP"})
        self.assertGreater(len({c["metrics"]["net_pnl"] for c in seq.cells}), 1)
        seq_calls, CountsCells.calls = CountsCells.calls, 0
        par = sweep(CountsCells, self.root, grid, assets=["BTC"], params={"side": "UP"}, cell_workers=4)
        self.assertEqual(seq.cells, par.cells)
        self.assertEqual(CountsCells.calls, seq_calls, "the parent did not carry the class state on")

    def test_a_cell_that_fails_fails_the_parallel_sweep_the_same_way(self):
        grid = {"size": [5, 6, 7, 13, 9]}
        with self.assertRaises(BacktestError) as seq:
            sweep(FailsOn13, self.root, grid, assets=["BTC"], params={"side": "UP"})
        with self.assertRaises(BacktestError) as par:
            sweep(FailsOn13, self.root, grid, assets=["BTC"], params={"side": "UP"}, cell_workers=3)
        self.assertEqual((seq.exception.code, seq.exception.detail), (par.exception.code, par.exception.detail))

    def test_a_failed_cell_stops_the_cells_still_running(self):
        # Sequentially the sweep raises at 13 and never reaches 99, which would
        # block forever; in parallel 99 is already running and must be killed.
        import multiprocessing
        import signal

        def hung(*_):
            for p in multiprocessing.active_children():  # or this test process would hang at exit too
                p.kill()
            # Not TimeoutError: that is an OSError, and multiprocessing's waitpid swallows those.
            raise RuntimeError("sweep hung")
        signal.signal(signal.SIGALRM, hung)
        signal.alarm(30)
        try:
            with self.assertRaises(BacktestError) as cm:
                sweep(HangsOn99, self.root, {"size": [5, 13, 99, 98]}, assets=["BTC"], params={"side": "UP"},
                      cell_workers=3)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
        self.assertIn("thirteen", cm.exception.detail)

    def test_an_interrupt_while_waiting_kills_every_cell(self):
        import multiprocessing
        import signal
        calls = []

        def alarm(*_):
            calls.append(1)
            if len(calls) == 1:
                signal.alarm(20)
                raise KeyboardInterrupt  # Ctrl-C, while the parent waits on the sleeping cell
            for p in multiprocessing.active_children():
                p.kill()
            raise RuntimeError("the interrupt did not end the sweep")
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(2)
        try:
            with self.assertRaises(KeyboardInterrupt):
                sweep(HangsOn99, self.root, {"size": [5, 99, 98]}, assets=["BTC"], params={"side": "UP"},
                      cell_workers=2)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
        self.assertEqual(multiprocessing.active_children(), [])

    def test_a_cell_that_dies_leaving_a_process_behind_is_rerun_here(self):
        # The cell forks a sleeper and is killed (out of memory, say); the sleeper
        # must not keep the sweep waiting, and must not outlive it.
        pids = tempfile.mkdtemp(prefix="ot-bt-pids-")
        import signal
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(RuntimeError("sweep hung")))
        signal.alarm(60)
        try:
            par = sweep(DiesOn77, self.root, {"size": [5, 77, 6]}, assets=["BTC"],
                        params={"side": "UP", "parent": os.getpid(), "pids": pids}, cell_workers=3)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
        seq = sweep(DiesOn77, self.root, {"size": [5, 77, 6]}, assets=["BTC"],
                    params={"side": "UP", "parent": os.getpid(), "pids": pids})
        self.assertEqual(par.cells, seq.cells)
        sleepers = [int(p) for p in os.listdir(pids)]
        shutil.rmtree(pids, ignore_errors=True)
        self.assertTrue(sleepers, "the cell never forked")
        for pid in sleepers:
            for _ in range(100):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                import time
                time.sleep(0.05)
            else:
                self.fail(f"process {pid} the dead cell started is still running")

    def test_an_interrupt_just_after_fork_still_kills_and_reaps_the_cell(self):
        import signal
        real_fork, forked = os.fork, []

        def fork():
            pid = real_fork()
            if pid:
                forked.append(pid)
                os.kill(os.getpid(), signal.SIGINT)  # Ctrl-C before the cell is registered
            return pid
        os.fork = fork
        try:
            with self.assertRaises(KeyboardInterrupt):
                sweep(HangsOn99, self.root, {"size": [5, 99]}, assets=["BTC"], params={"side": "UP"},
                      cell_workers=2)
        finally:
            os.fork = real_fork
        self.assertEqual(len(forked), 1)
        try:
            with self.assertRaises(ChildProcessError):  # reaped already, by the sweep
                os.waitpid(forked[0], os.WNOHANG)
        except AssertionError:
            for kill in (lambda: os.killpg(forked[0], signal.SIGKILL), lambda: os.kill(forked[0], signal.SIGKILL)):
                try:
                    kill()
                except ProcessLookupError:
                    pass
            os.waitpid(forked[0], 0)
            raise

    def test_an_interrupt_just_after_fork_with_another_os_thread_alive(self):
        # threading does not see a raw _thread, so the sweep still forks; the
        # kernel hands a process-wide SIGINT to whichever thread does not block
        # it, and Python raises it in the main thread wherever it happens to be.
        import _thread
        import time
        stop = []
        _thread.start_new_thread(lambda: [time.sleep(0.01) for _ in iter(lambda: bool(stop), True)], ())
        try:
            for _ in range(5):
                self.test_an_interrupt_just_after_fork_still_kills_and_reaps_the_cell()
        finally:
            stop.append(1)

    def test_without_waitid_cells_run_here(self):
        pids = tempfile.mkdtemp(prefix="ot-bt-pids-")
        waitid = os.waitid
        del os.waitid
        try:
            s = sweep(PidSide, self.root, {"size": [5, 10, 15]}, assets=["BTC"],
                      params={"side": "UP", "pids": pids}, cell_workers=3)
            self.assertEqual(os.listdir(pids), [str(os.getpid())])
        finally:
            os.waitid = waitid
            shutil.rmtree(pids, ignore_errors=True)
        self.assertEqual(len(s.cells), 3)

    def test_no_forking_while_other_threads_run(self):
        import threading
        from outcometick.backtest import _parallel
        pids = tempfile.mkdtemp(prefix="ot-bt-pids-")
        stop = threading.Event()
        holder = threading.Thread(target=lambda: (_parallel._STATE_LOCK.acquire(), stop.wait(),
                                                  _parallel._STATE_LOCK.release()))
        holder.start()
        try:
            s = sweep(PidSide, self.root, {"size": [5, 10, 15]}, assets=["BTC"],
                      params={"side": "UP", "pids": pids}, cell_workers=3)
            self.assertEqual(os.listdir(pids), [str(os.getpid())])
        finally:
            stop.set()
            holder.join()
            shutil.rmtree(pids, ignore_errors=True)
        self.assertEqual(len(s.cells), 3)

    def test_bad_sweeps_are_refused(self):
        with self.assertRaises(BacktestError):
            sweep(SizedSide, self.root, {}, assets=["BTC"])
        with self.assertRaises(BacktestError):
            sweep(SizedSide, self.root, {"size": []}, assets=["BTC"])
        with self.assertRaises(BacktestError):
            sweep(SizedSide, self.root, {"size": [5]}, assets=["BTC"], params={"side": "UP"}, holdout=0.99)
        with self.assertRaises(BacktestError):
            sweep(SizedSide, self.root, {"size": [5]}, assets=["BTC"], params={"side": "UP"}, metric="nope")
        for bad in (0, 1.5, True, "2"):
            with self.assertRaises(BacktestError):
                sweep(SizedSide, self.root, {"size": [5]}, assets=["BTC"], params={"side": "UP"}, cell_workers=bad)


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


class LogsBooks(Strategy):
    def on_market_open(self, ctx, market):
        ctx.log(f"open {market.market_id}")

    def on_book(self, ctx, book):
        ctx.log(f"book {ctx.now}")


class DaysDecodeAlone(unittest.TestCase):
    """A day decodes the same whatever day was decoded before it, as the hosted
    runner and `ot run` decode it: the book throttle starts afresh each day. A
    market across midnight has price changes 8 ms apart on either side of it,
    inside BTC's 20 ms window -- carried over, the throttle would drop the second."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="ot-bt-days-")
        self.cache = tempfile.mkdtemp(prefix="ot-bt-days-cache-")
        T1 = T0 + 86_400_000  # noqa: N806
        start = T1 - 150_000
        up, down = "7" * 70, "8" * 70
        raw = {"cryptoMarketConfig": {"twapLookbackSeconds": 60}, "feesEnabled": False}
        row = {"slug": "btc-updown-5m-x", "asset": "btc", "interval_sec": 300, "condition_id": "0x" + "cd" * 32,
               "token_ids": [up, down], "start_sec": start // 1000, "end_sec": start // 1000 + 300,
               "resolved": True, "outcome_prices": ["1", "0"], "strike_value": "79104869058460346482688", "raw": raw}
        change = lambda ts, px: price_change(ts, [{"asset_id": up, "side": "SELL", "price": px, "size": "10",  # noqa: E731
                                                   "best_bid": "0.40", "best_ask": px}], slug="btc-updown-5m-x")
        for day, t, rows in (("2026-09-08", start + 1000, [change(T1 - 5, "0.55")]),
                             ("2026-09-09", T1 + 1000, [change(T1 + 3, "0.56")])):
            _write(self.root, f"data/polymarket/daily/markets/BTC-5m/BTC-5m-markets-{day}.jsonl.gz", [row])
            _write(self.root, f"data/polymarket/daily/book/BTC-5m/BTC-5m-book-{day}.jsonl.gz", [
                {"slug": "btc-updown-5m-x", "asset_id": tok, "event_type": "book", "event_ts_ms": t, "recv_ms": t,
                 "payload": {"asks": [{"price": "0.60", "size": "50"}], "bids": [{"price": "0.40", "size": "50"}]}}
                for tok in (up, down)])
            add_price_changes(self.root, rows, day=day)
            lo = max(start, T0 if day == "2026-09-08" else T1)
            hi = min(start + 300_000, T1 if day == "2026-09-08" else T1 + 86_400_000)
            _write(self.root, f"data/chainlink-twap-60s/daily/prices/BTCUSD/BTCUSD-twap60s-prices-{day}.csv.gz",
                   [[ts, 79100, "0", ts, ts] for ts in range(lo + 10, hi, 10_000)],
                   csv_header=["feed_ts_ms", "value", "full_accuracy_value", "server_ts_ms", "recv_ms"])

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self.cache, ignore_errors=True)

    def test_two_days_replay_as_each_day_alone(self):
        one = run(LogsBooks, self.root, assets=["BTC"], days=["2026-09-08"], cache=False).logs
        two = run(LogsBooks, self.root, assets=["BTC"], days=["2026-09-09"], cache=False).logs
        self.assertIn(f"book {T0 + 86_400_000 + 3}", two)
        both = run(LogsBooks, self.root, assets=["BTC"], days=["2026-09-08", "2026-09-09"], cache=False).logs
        self.assertEqual(both, one + "\n" + two)
        # And the cache holds each day as it decodes alone, whichever run wrote it.
        run(LogsBooks, self.root, assets=["BTC"], days=["2026-09-08", "2026-09-09"], cache=self.cache)
        self.assertEqual(run(LogsBooks, self.root, assets=["BTC"], days=["2026-09-09"], cache=self.cache).logs, two)


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


# --- generic data: any binary market, your own data -------------------------------

G0 = 1_760_000_000_000


def _gbook(market, token, t, bids, asks):
    return {"event_type": "book", "market": market, "asset_id": token, "timestamp": str(t),
            "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
            "asks": [{"price": str(p), "size": str(s)} for p, s in asks]}


def _gamma(resolved=None):
    g = {"conditionId": "0xg", "slug": "g", "question": "G?", "clobTokenIds": '["ty","tn"]',
         "outcomes": '["Yes","No"]', "outcomePrices": '["0.5","0.5"]',
         "feesEnabled": True, "feeSchedule": {"rate": 0.07, "exponent": 1}}
    if resolved:
        g.update(umaResolutionStatus="resolved", outcomePrices=resolved, umaEndDate="2025-10-09T09:00:00Z")
    return g


class BuyYesOnce(Strategy):
    def on_market_open(self, ctx, market):
        self.done = False
        self.seen = sorted(market.keys()) if hasattr(market, "keys") else None

    def on_book(self, ctx, ev):
        if self.done:
            return None
        self.done = True
        return Order(side="Yes", size=10, limit=0.6)


class BuyNoOnce(Strategy):
    def on_book(self, ctx, ev):
        if ctx.position().size > 0:
            return None
        return Order(side="no", size=10, limit=0.6)


class GenericData(unittest.TestCase):
    from outcometick.backtest import Dataset

    def frames(self):
        return [
            _gbook("0xg", "ty", G0, [(0.48, 100)], [(0.50, 60), (0.52, 40)]),
            _gbook("0xg", "tn", G0, [(0.48, 100)], [(0.52, 100)]),
            _gbook("0xg", "ty", G0 + 60_000, [(0.58, 100)], [(0.62, 100)]),
        ]

    def test_resolved_market_pays_at_its_resolution_time(self):
        ds = self.Dataset.from_frames([_gamma('["1","0"]')], self.frames())
        r = run(BuyYesOnce, ds, datasets=["book"])
        (t,) = r.trades
        # 10 @ 0.50; venue fee round5(10 x 0.07 x 0.5 x 0.5) = 0.175; pays $1 each.
        self.assertEqual((t["how"], t["side"], t["label"], t["outcome"]), ("settled", "UP", "Yes", "UP"))
        self.assertAlmostEqual(t["pnl"], 10 * 1 - 5 - 0.175, places=9)
        self.assertEqual(t["closed_ms"], 1760000400000)  # 2025-10-09T09:00:00Z, after the data
        self.assertEqual(r.report["data_kind"], "generic")
        self.assertEqual(r.report["strategy_on_resolved"]["markets"], 1)

    def test_unresolved_market_is_valued_at_the_external_mid(self):
        ds = self.Dataset.from_frames([_gamma()], self.frames())
        r = run(BuyYesOnce, ds, datasets=["book"])
        (t,) = r.trades
        # The last UP book is 0.58 / 0.62: mid 0.60, valued, not sold.
        self.assertEqual(t["how"], "marked")
        self.assertAlmostEqual(t["mark_px"], 0.60, places=12)
        self.assertAlmostEqual(t["marked_pnl"], 10 * 0.60 - 5, places=9)
        self.assertAlmostEqual(t["pnl"], 10 * 0.60 - 5 - 0.175, places=9)
        self.assertEqual(t["mark_source"], "up_mid")
        v = r.report["valuation"]
        self.assertTrue(v["net_pnl_includes_marks"])
        self.assertTrue(v["complete"])
        self.assertEqual(v["marked_pnl"], 1.0)
        self.assertEqual(v["realised_pnl"], -0.18)

    def test_a_stale_mark_is_refused_and_the_leg_valued_at_zero(self):
        ds = self.Dataset.from_frames([_gamma()], self.frames() + [
            {"event_type": "last_trade_price", "market": "0xg", "asset_id": "tn", "price": "0.3", "size": "1",
             "side": "BUY", "timestamp": str(G0 + 600_000)}])
        r = run(BuyYesOnce, ds, datasets=["book"], mark_max_age_ms=1000)
        (t,) = r.trades
        # The last trade is fresh (1 - 0.3 = 0.7) but the books are 540 s old:
        # each candidate is checked before the next, so the trade marks it.
        self.assertEqual(t["mark_source"], "last_trade")
        self.assertAlmostEqual(t["mark_px"], 0.7, places=12)
        # Ten minutes on, the DOWN book is one-sided and the UP book is old: no fresh mark.
        quiet = self.frames() + [_gbook("0xg", "tn", G0 + 600_000, [], [(0.52, 100)])]
        r2 = run(BuyYesOnce, self.Dataset.from_frames([_gamma()], quiet), datasets=["book"], mark_max_age_ms=1000)
        (z,) = r2.trades
        self.assertIsNone(z["mark_px"])
        self.assertEqual(z["mark_missing"], "stale")
        self.assertFalse(r2.report["valuation"]["complete"])

    def test_the_last_print_in_input_order_marks_when_no_book_can(self):
        def tr(token, t):
            return {"event_type": "last_trade_price", "market": "0xg", "asset_id": token, "price": "0.4",
                    "size": "1", "side": "BUY", "timestamp": str(t)}
        # Asks only: no two-sided quote, so the mark is the last public print.
        frames = [_gbook("0xg", "ty", G0, [], [(0.50, 60)]), tr("ty", G0 + 5), tr("tn", G0 + 5)]
        (t,) = run(BuyYesOnce, self.Dataset.from_frames([_gamma()], frames), datasets=["book"]).trades
        self.assertEqual(t["mark_source"], "last_trade")
        self.assertAlmostEqual(t["mark_px"], 0.6, places=12)   # the DOWN print came last: 1 - 0.4
        frames = [_gbook("0xg", "ty", G0, [], [(0.50, 60)]), tr("tn", G0 + 5), tr("ty", G0 + 5)]
        (t,) = run(BuyYesOnce, self.Dataset.from_frames([_gamma()], frames), datasets=["book"]).trades
        self.assertAlmostEqual(t["mark_px"], 0.4, places=12)

    def test_pmdata_single_book_is_mirrored_atomically(self):
        l2 = [
            {"market_slug": "g", "condition_id": "0xg", "timestamp": G0, "event_type": "book",
             "bid_prices": [0.48], "bid_sizes": [100], "ask_prices": [0.50], "ask_sizes": [60],
             "pc_price": None, "pc_side": None, "pc_size": None, "winning_outcome": None},
            # A new UP bid at 0.49 is a new DOWN ask at 0.51, in the same step.
            {"market_slug": "g", "condition_id": "0xg", "timestamp": G0 + 10, "event_type": "price_change",
             "bid_prices": None, "bid_sizes": None, "ask_prices": None, "ask_sizes": None,
             "pc_price": 0.49, "pc_side": "BUY", "pc_size": 30, "winning_outcome": None},
            {"market_slug": "g", "condition_id": "0xg", "timestamp": G0 + 99_000, "event_type": "market_resolved",
             "bid_prices": None, "bid_sizes": None, "ask_prices": None, "ask_sizes": None,
             "pc_price": None, "pc_side": None, "pc_size": None, "winning_outcome": "No"},
        ]
        trades = [{"timestamp": G0 + 20, "outcome": "No", "price": 0.51, "size": 5, "side": "BUY"}]
        r = run(BuyNoOnce, self.Dataset.from_pmdata(l2, trades, market=_gamma()), datasets=["book", "trades"])
        fills = [f for f in r.fills if f["filled"] > 0]
        # First book: DOWN ask = 1 - 0.48 = 0.52 x 100. After the delta: 0.51 x 30 in front.
        self.assertEqual([(f["avg_px"], f["filled"]) for f in fills], [(0.52, 10)])
        (t,) = r.trades
        self.assertEqual((t["how"], t["side"], t["label"], t["outcome"]), ("settled", "DOWN", "No", "DOWN"))
        self.assertEqual(t["closed_ms"], G0 + 99_000)

    def test_pmdata_rows_of_another_market_refuse_the_market(self):
        row = {"market_slug": "g", "condition_id": "0xg", "timestamp": G0, "event_type": "book",
               "bid_prices": [0.48], "bid_sizes": [100], "ask_prices": [0.50], "ask_sizes": [60]}
        for bad_l2, bad_trades in (
            ([row, {**row, "condition_id": "0xother", "timestamp": G0 + 5}], None),
            ([row, {**row, "market_slug": "other", "timestamp": G0 + 5}], None),
            ([row], [{"condition_id": "0xother", "timestamp": G0 + 9, "outcome": "Yes", "price": 0.5, "size": 1, "side": "BUY"}]),
        ):
            with self.assertRaises(BacktestError) as cm:
                run(BuyNoOnce, self.Dataset.from_pmdata(bad_l2, bad_trades, market=_gamma()), datasets=["book", "trades"])
            self.assertIn("another market", str(cm.exception))
        # A record with no slug, rows with no condition id: two slugs are still two markets.
        bare = {"market_id": "0xg", "token_ids": ["ty", "tn"], "labels": ["Yes", "No"]}
        slug_only = {k: v for k, v in row.items() if k != "condition_id"}
        for bad_l2, bad_trades in (
            ([slug_only, {**slug_only, "market_slug": "h", "timestamp": G0 + 5}], None),
            ([slug_only], [{"market_slug": "h", "timestamp": G0 + 9, "outcome": "Yes", "price": 0.5, "size": 1, "side": "BUY"}]),
        ):
            with self.assertRaises(BacktestError) as cm:
                run(BuyNoOnce, self.Dataset.from_pmdata(bad_l2, bad_trades, market=bare), datasets=["book", "trades"])
            self.assertIn("another market", str(cm.exception))

    def test_a_directory_and_from_frames_are_the_same_dataset(self):
        root = tempfile.mkdtemp()
        try:
            with open(os.path.join(root, "markets.json"), "w") as fh:
                json.dump([_gamma('["0","1"]')], fh)
            with gzip.open(os.path.join(root, "f.jsonl.gz"), "wt") as fh:
                for f in self.frames():
                    fh.write(json.dumps(f) + "\n")
            a = run(BuyYesOnce, root, datasets=["book"])
            b = run(BuyYesOnce, self.Dataset.from_frames([_gamma('["0","1"]')], self.frames()), datasets=["book"])
            self.assertEqual(a.trades, b.trades)
            self.assertEqual(a.trades[0]["outcome"], "DOWN")
            with self.assertRaises(BacktestError):
                run(BuyYesOnce, root, datasets=["book"], days=["2025-10-09"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_a_label_the_market_does_not_have_stops_the_run(self):
        class Wrong(Strategy):
            def on_book(self, ctx, ev):
                return Order(side="Maybe", size=1, limit=0.6)
        with self.assertRaises(BacktestError) as cm:
            run(Wrong, self.Dataset.from_frames([_gamma()], self.frames()), datasets=["book"])
        self.assertIn('"Maybe"', str(cm.exception))
