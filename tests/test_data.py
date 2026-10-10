"""Tests for the Python data-subscription client.

Driven against a stub whose behaviour was copied from
api/subscription-api.mjs -- the 302-with-checksum on /v1/dl, the
date-XOR-from/to rejection, the 403 that carries floor and ceiling. The point is
not that the client works against a friendly server; it is that it handles what
the real routes actually do.

Mirrors client/data.test.mjs case for case, for the same reason the two engines
have a conformance suite: two clients that drift are worse than one.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from outcometick.data import DataClient, OutcometickError, NO_VALUE, OrderBook

PAYLOAD = gzip.compress(b"ts_ms,value\n1755000000000,65000\n")
SHA = hashlib.sha256(PAYLOAD).hexdigest()

FILE_ROW = {
    "date": "2026-08-12",
    "name": "BTCUSD-prices-2026-08-12.csv.gz",
    "venue": "polymarket",
    "dataset": "prices",
    "asset": "BTCUSD",
    "interval": None,
    "bytes": len(PAYLOAD),
    "sha256": SHA,
}

# The checksum the smart-money redirect announces (a list so a test can swap it).
SMART_SHA = [None]

# Per-test overrides: {path: (status, body_or_bytes)}
OVERRIDES: dict = {}
SEEN: list = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _json(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        SEEN.append({
            "path": u.path,
            "query": {k: v[0] for k, v in parse_qs(u.query).items()},
            "auth": self.headers.get("Authorization"),
        })

        if u.path in OVERRIDES:
            status, body = OVERRIDES[u.path]
            if isinstance(body, bytes):
                self.send_response(status)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(status, body)
            return

        if u.path == "/v1/meta":
            return self._json(200, {
                "firstDay": "2026-06-08", "lastDay": "2026-08-12", "days": 66,
                "venues": ["polymarket"], "assets": ["BTCUSD", "ETHUSD"],
                "intervals": ["5m", "1h"],
                "filterTokens": {"noValue": "none", "appliesTo": ["interval"]},
                "datasets": {"prices": "1 Hz settlement stream"},
                "sampledFrom": "2026-08-12",
            })
        if u.path == "/v1/mirror/days":
            return self._json(200, {
                "days": ["2026-08-11", "2026-08-12"],
                "floor": "2026-07-14", "ceiling": None,
            })
        if u.path == "/v1/files":
            q = parse_qs(u.query)
            if "date" in q and ("from" in q or "to" in q):
                return self._json(400, {"error": "use either date, or from/to — not both"})
            return self._json(200, {
                "from": "2026-08-12", "to": "2026-08-12", "days": 1, "count": 1,
                "bytes": len(PAYLOAD), "files": [dict(FILE_ROW, url="http://x/dl")],
            })
        if u.path == "/v1/mirror/download":
            return self._json(200, {
                "url": "http://x/signed", "name": FILE_ROW["name"],
                "bytes": len(PAYLOAD), "sha256": SHA, "expiresInSec": 900,
            })
        if u.path == "/v1/smart/download":
            self.send_response(302)
            self.send_header("location", f"http://127.0.0.1:{self.server.server_port}/signed-bytes")
            self.send_header("x-outcometick-sha256", SMART_SHA[0])
            self.send_header("content-length", "0")
            self.end_headers()
            return None
        if u.path.startswith("/v1/dl/"):
            # Exactly what the real route does.
            self.send_response(302)
            self.send_header("location", f"http://127.0.0.1:{self.server.server_port}/signed-bytes")
            self.send_header("x-outcometick-sha256", SHA)
            self.send_header("x-amz-meta-sha256", SHA)
            self.send_header("content-length", "0")
            self.end_headers()
            return None
        if u.path == "/signed-bytes":
            self.send_response(200)
            self.send_header("content-length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)
            return None
        if u.path == "/v1/public/coverage":
            return self._json(200, {"venues": {}})
        if u.path == "/v1/health":
            return self._json(200, {"ok": True})
        return self._json(404, {"error": "not found"})


class ClientTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        SEEN.clear()
        OVERRIDES.clear()
        self.ot = DataClient(key="ck_test", base_url=self.base)


class TestAuth(ClientTestCase):
    def test_key_travels_as_a_bearer_header(self):
        self.ot.meta()
        self.assertEqual(SEEN[0]["auth"], "Bearer ck_test")

    def test_key_is_read_from_ot_key(self):
        prev = os.environ.get("OT_KEY")
        os.environ["OT_KEY"] = "ck_from_env"
        try:
            self.assertEqual(DataClient().key, "ck_from_env")
        finally:
            if prev is None:
                del os.environ["OT_KEY"]
            else:
                os.environ["OT_KEY"] = prev

    def test_missing_key_explains_itself(self):
        prev = os.environ.pop("OT_KEY", None)
        try:
            with self.assertRaises(ValueError) as ctx:
                DataClient(base_url=self.base).meta()
            self.assertIn("OT_KEY", str(ctx.exception))
        finally:
            if prev is not None:
                os.environ["OT_KEY"] = prev

    def test_public_endpoints_send_no_key(self):
        ot = DataClient(key=None, base_url=self.base)
        ot.coverage()
        ot.health()
        self.assertEqual([s["auth"] for s in SEEN], [None, None])


class TestDiscovery(ClientTestCase):
    def test_meta_keeps_the_sentinel_out_of_value_arrays(self):
        m = self.ot.meta()
        self.assertEqual(m["assets"], ["BTCUSD", "ETHUSD"])
        # A caller building an enum from intervals must not meet a token.
        self.assertNotIn(NO_VALUE, m["intervals"])
        self.assertEqual(m["filterTokens"]["noValue"], NO_VALUE)

    def test_days_reports_window_bounds(self):
        d = self.ot.days()
        self.assertEqual(d["floor"], "2026-07-14")
        self.assertIsNone(d["ceiling"])

    def test_list_filters_become_comma_alternatives(self):
        self.ot.files(asset=["btc", "eth"], interval=["5m", NO_VALUE], dataset="prices")
        q = SEEN[-1]["query"]
        self.assertEqual(q["asset"], "btc,eth")
        self.assertEqual(q["interval"], "5m,none")

    def test_format_is_passed_through_and_omitted_when_not_asked(self):
        self.ot.files(dataset="book", format="parquet")
        self.assertEqual(SEEN[-1]["query"]["format"], "parquet")
        self.ot.files(dataset="book")
        self.assertNotIn("format", SEEN[-1]["query"])

    def test_empty_filters_are_omitted(self):
        self.ot.files(asset=[], dataset=None, date="2026-08-12")
        self.assertEqual(list(SEEN[-1]["query"]), ["date"])

    def test_from_is_sent_as_from_not_from_underscore(self):
        # The parameter is from_ in Python because `from` is a keyword; the
        # wire name must still be `from`.
        self.ot.files(from_="2026-08-01", to="2026-08-02")
        self.assertIn("from", SEEN[-1]["query"])
        self.assertNotIn("from_", SEEN[-1]["query"])

    def test_date_with_range_is_refused_before_the_round_trip(self):
        with self.assertRaises(ValueError):
            self.ot.files(date="2026-08-12", from_="2026-08-01")
        self.assertEqual(SEEN, [])

    def test_api_error_keeps_status_and_extra_fields(self):
        OVERRIDES["/v1/files"] = (403, {
            "error": "from is outside your coverage",
            "floor": "2026-07-14", "ceiling": "2026-08-12",
        })
        with self.assertRaises(OutcometickError) as ctx:
            self.ot.files(from_="2020-01-01", to="2020-01-02")
        self.assertEqual(ctx.exception.status, 403)
        self.assertEqual(ctx.exception.body["floor"], "2026-07-14")


class TestDownload(ClientTestCase):
    def test_follows_the_redirect_and_verifies(self):
        res = self.ot.files(date="2026-08-12")
        got = self.ot.download(res["files"][0])
        self.assertEqual(got["bytes"], PAYLOAD)
        self.assertEqual(got["sha256"], SHA)
        # The signed URL was fetched WITHOUT our key.
        signed = [s for s in SEEN if s["path"] == "/signed-bytes"]
        self.assertEqual(signed[0]["auth"], None)

    def test_works_from_bare_date_and_name(self):
        got = self.ot.download("2026-08-12", FILE_ROW["name"])
        self.assertEqual(got["bytes"], PAYLOAD)
        # No row supplied, so the checksum came off the 302 header.
        self.assertEqual(got["sha256"], SHA)

    def test_corrupted_download_is_rejected(self):
        OVERRIDES["/signed-bytes"] = (200, b"not the bytes you asked for")
        with self.assertRaises(ValueError) as ctx:
            self.ot.download("2026-08-12", FILE_ROW["name"])
        self.assertIn("checksum mismatch", str(ctx.exception))

    def test_verify_false_returns_bytes_unchecked(self):
        OVERRIDES["/signed-bytes"] = (200, b"not the bytes you asked for")
        got = self.ot.download("2026-08-12", FILE_ROW["name"], verify=False)
        self.assertEqual(got["bytes"], b"not the bytes you asked for")

    def test_save_to_writes_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "x.csv.gz")
            self.ot.download(FILE_ROW, save_to=out)
            with open(out, "rb") as fh:
                self.assertEqual(fh.read(), PAYLOAD)

    def test_a_404_raises_rather_than_saving_an_error_page(self):
        OVERRIDES["/v1/dl/2026-08-12/nope.csv.gz"] = (404, {"error": "file not found"})
        with self.assertRaises(OutcometickError) as ctx:
            self.ot.download("2026-08-12", "nope.csv.gz")
        self.assertEqual(ctx.exception.status, 404)

    def test_sign_url(self):
        s = self.ot.sign_url("2026-08-12", FILE_ROW["name"], expires_in=300)
        self.assertEqual(s["sha256"], SHA)
        self.assertEqual(SEEN[-1]["query"]["expiresIn"], "300")


class TestSmart(ClientTestCase):
    def setUp(self):
        super().setUp()
        SMART_SHA[0] = SHA

    def test_download_follows_verifies_and_never_sends_the_key_to_r2(self):
        got = self.ot.smart_download("2026-10-08", "top1000")
        self.assertEqual(got["bytes"], PAYLOAD)
        self.assertEqual(got["sha256"], SHA)
        self.assertEqual((got["day"], got["list"]), ("2026-10-08", "top1000"))
        dl = [x for x in SEEN if x["path"] == "/v1/smart/download"][0]
        self.assertEqual(dl["query"], {"day": "2026-10-08", "list": "top1000"})
        self.assertEqual(dl["auth"], "Bearer ck_test")
        self.assertEqual([x for x in SEEN if x["path"] == "/signed-bytes"][0]["auth"], None)

    def test_checksum_mismatch_is_rejected(self):
        SMART_SHA[0] = "0" * 64
        with self.assertRaises(ValueError) as ctx:
            self.ot.smart_download("2026-10-08")
        self.assertIn("checksum mismatch for 2026-10-08/top100", str(ctx.exception))

    def test_arguments_are_checked_before_calling_out(self):
        with self.assertRaises(ValueError):
            self.ot.smart_download("2026-10-08", "all")
        with self.assertRaises(ValueError):
            self.ot.smart_download(None)
        self.assertEqual(SEEN, [])

    def test_days_with_key_coverage_and_plans_public(self):
        OVERRIDES["/v1/public/smart-coverage"] = (200, {"firstDay": "2026-10-08", "lastDay": "2026-10-09", "days": 2})
        OVERRIDES["/v1/public/smart-plans"] = (200, {"onSale": False, "plans": []})
        OVERRIDES["/v1/smart/days"] = (503, {"error": "smart-money is not on sale yet"})
        self.assertEqual(self.ot.smart_coverage()["days"], 2)
        self.assertEqual(SEEN[-1]["auth"], None)
        self.assertEqual(self.ot.smart_plans(), {"onSale": False, "plans": []})
        with self.assertRaises(OutcometickError) as ctx:
            self.ot.smart_days()
        self.assertEqual(ctx.exception.status, 503)


class TestBaseUrl(ClientTestCase):
    def test_trailing_slash_does_not_double_the_path(self):
        ot = DataClient(key="ck_test", base_url=self.base + "/")
        ot.meta()
        self.assertEqual(SEEN[0]["path"], "/v1/meta")


# ---------- OrderBook (mirrors the OrderBook cases in client/data.test.mjs) ----------

def book_row(asset_id, bids, asks, recv_ms=1):
    return {"slug": "btc-updown-5m-1", "asset_id": asset_id, "event_type": "book", "event_ts_ms": recv_ms, "recv_ms": recv_ms,
            "payload": {"event_type": "book", "asset_id": asset_id, "market": "0xm", "timestamp": str(recv_ms), "hash": "h",
                        "bids": [{"price": p, "size": s} for p, s in bids], "asks": [{"price": p, "size": s} for p, s in asks]}}


def change_row(items, recv_ms=2):
    return {"slug": "btc-updown-5m-1", "asset_id": None, "event_type": "price_change", "event_ts_ms": recv_ms, "recv_ms": recv_ms,
            "payload": {"event_type": "price_change", "market": "0xm", "timestamp": str(recv_ms),
                        "price_changes": [{"asset_id": a, "side": sd, "price": p, "size": s, "hash": "h", "best_bid": bb, "best_ask": ba}
                                          for a, sd, p, s, bb, ba in items]}}


def bbo_row(asset_id, best_bid, best_ask, recv_ms=3):
    return {"slug": "btc-updown-5m-1", "asset_id": asset_id, "event_type": "best_bid_ask", "event_ts_ms": recv_ms, "recv_ms": recv_ms,
            "payload": {"event_type": "best_bid_ask", "asset_id": asset_id, "market": "0xm", "best_bid": best_bid,
                        "best_ask": best_ask, "spread": "0.01", "timestamp": str(recv_ms)}}


def flat(side):
    return [[lvl["price"], lvl["size"]] for lvl in side]


class TestOrderBook(unittest.TestCase):
    def test_snapshot_sets_the_ladder_best_first_as_spelled(self):
        b = OrderBook().apply(book_row("A", [["0.48", "5"], ["0.50", "10"], ["0.49", "0"]], [["0.53", "2"], ["0.52", "7"]]))
        self.assertEqual(flat(b.ladder("A")["bids"]), [["0.50", "10"], ["0.48", "5"]])
        self.assertEqual(flat(b.ladder("A")["asks"]), [["0.52", "7"], ["0.53", "2"]])
        self.assertEqual(b.best("A"), {"bid": {"price": "0.50", "size": "10"}, "ask": {"price": "0.52", "size": "7"}})
        self.assertEqual(b.assets(), ["A"])

    def test_price_change_sets_absolute_sizes_and_zero_removes(self):
        b = OrderBook().apply(book_row("A", [["0.50", "10"], ["0.48", "5"]], [["0.52", "7"]]))
        b.apply(change_row([["A", "BUY", "0.5", "12", "0.5", "0.52"], ["A", "BUY", "0.48", "0", "0.5", "0.52"],
                            ["A", "SELL", "0.55", "3", "0.5", "0.52"]]))
        self.assertEqual(flat(b.ladder("A")["bids"]), [["0.5", "12"]])
        self.assertEqual(flat(b.ladder("A")["asks"]), [["0.52", "7"], ["0.55", "3"]])

    def test_ghost_from_missed_removal_is_pruned(self):
        b = OrderBook().apply(book_row("A", [["0.50", "10"], ["0.49", "4"]], [["0.52", "7"], ["0.53", "1"]]))
        b.apply(change_row([["A", "BUY", "0.47", "6", "0.49", "0.53"]]))
        self.assertEqual(flat(b.ladder("A")["bids"]), [["0.49", "4"], ["0.47", "6"]])
        self.assertEqual(flat(b.ladder("A")["asks"]), [["0.53", "1"]])

    def test_best_bid_ask_prunes_only_and_never_adds(self):
        b = OrderBook().apply(book_row("A", [["0.50", "10"], ["0.49", "4"]], [["0.52", "7"]]))
        b.apply(bbo_row("A", "0.49", "0.51"))
        self.assertEqual(flat(b.ladder("A")["bids"]), [["0.49", "4"]])
        self.assertEqual(flat(b.ladder("A")["asks"]), [["0.52", "7"]])

    def test_missing_or_out_of_range_best_prunes_nothing(self):
        b = OrderBook().apply(book_row("A", [["0.50", "10"]], [["0.52", "7"]]))
        b.apply(bbo_row("A", "", None))
        b.apply(bbo_row("A", "1.5", "abc"))
        b.apply(bbo_row("A", True, "-0.1"))
        self.assertEqual(flat(b.ladder("A")["bids"]), [["0.50", "10"]])
        self.assertEqual(flat(b.ladder("A")["asks"]), [["0.52", "7"]])

    def test_blank_containers_and_non_decimal_spellings_are_not_numbers(self):
        b = OrderBook().apply(book_row("A", [["0.50", "10"], ["0.49", "4"]], [["0.52", "7"]]))
        for junk in [" ", [], {}, "0x0", "Infinity", "1e400", " 0.5", "1_0"]:
            b.apply(bbo_row("A", junk, junk))
        b.apply(change_row([["A", "BUY", "0.50", " ", "x", "x"], ["A", "BUY", " ", "0", "x", "x"], ["A", "SELL", "0.52", [], "x", "x"]]))
        self.assertEqual(flat(b.ladder("A")["bids"]), [["0.50", "10"], ["0.49", "4"]])
        self.assertEqual(flat(b.ladder("A")["asks"]), [["0.52", "7"]])
        b.apply(bbo_row("A", "5e-1", 1))
        self.assertEqual(flat(b.ladder("A")["bids"]), [["0.50", "10"], ["0.49", "4"]])

    def test_tokens_kept_apart_each_pruned_by_its_own_last_item(self):
        b = OrderBook().apply(book_row("A", [["0.50", "1"]], [])).apply(book_row("B", [["0.50", "2"]], []))
        b.apply(change_row([["A", "BUY", "0.40", "1", "0.45", "0.6"], ["B", "BUY", "0.30", "1", "0.5", "0.6"]]))
        self.assertEqual(flat(b.ladder("A")["bids"]), [["0.40", "1"]])
        self.assertEqual(flat(b.ladder("B")["bids"]), [["0.50", "2"], ["0.30", "1"]])

    def test_snapshot_replaces_whole_jsonl_legacy_and_other_types(self):
        b = OrderBook().apply(json.dumps(book_row("A", [["0.50", "10"]], [["0.52", "7"]])))
        b.apply({"asset_id": "A", "event_type": "price_change", "payload": {"changes": [{"side": "BUY", "price": "0.51", "size": "3"}]}})
        self.assertEqual(flat(b.ladder("A")["bids"]), [["0.51", "3"], ["0.50", "10"]])
        b.apply({"asset_id": "A", "event_type": "last_trade_price", "payload": {"price": "0.9", "size": "1"}})
        b.apply(book_row("A", [["0.45", "1"]], []))
        self.assertEqual(b.ladder("A"), {"bids": [{"price": "0.45", "size": "1"}], "asks": []})
        self.assertEqual(OrderBook().ladder("nope"), {"bids": [], "asks": []})
        self.assertEqual(OrderBook().best("nope"), {"bid": None, "ask": None})

    def test_ladder_hands_out_copies(self):
        b = OrderBook().apply(book_row("A", [["0.50", "10"]], []))
        b.ladder("A")["bids"][0]["size"] = "999"
        self.assertEqual(b.ladder("A")["bids"][0]["size"], "10")


if __name__ == "__main__":
    unittest.main()
