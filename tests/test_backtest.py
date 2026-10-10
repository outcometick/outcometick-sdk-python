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
from outcometick.backtest import BacktestError, run

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
