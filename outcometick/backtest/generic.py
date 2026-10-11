"""Backtest ANY Polymarket binary market from your own data.

A port of runner/generic-data.mjs -- read its header for the rules and why.
MUST MATCH it line for line in behaviour: the equivalence suite compares every
event and every market record the two produce from the same dataset.

Three ways in:
  - a dataset directory: markets.json / markets.jsonl at its top, plus any
    number of `*.jsonl` / `*.jsonl.gz` frame files (Polymarket CLOB
    market-channel frames, raw or as archive rows) anywhere under it;
  - Dataset.from_frames(markets, frames): the same, in memory;
  - Dataset.from_pmdata(l2, trades=None, market=...): the single-book table
    shape (bid_prices/ask_prices/... rows), its other token mirrored.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import heapq
import json
import math
import os
import re

from ._js import js_string, js_trim, json_loads
from .decode import FEE_MODEL_PM, OUTCOMES, _fee_of
from .local import _dump

GENERIC_KIND = "generic"

_TS_MIN = 946_684_800_000
_TS_MAX = 4_102_444_800_000

SORT_RUN = 200_000
MERGE_FANIN = 64
_SPOOL_FLUSH = 512


class DatasetError(Exception):
    code = "E_DATA"


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_time_ms(v):
    """Integer epoch milliseconds (number, or a string of digits), or None."""
    n = None
    if _is_num(v):
        n = v
    elif isinstance(v, str) and re.fullmatch(r"[0-9]+", v, re.ASCII):
        n = float(int(v)) if len(v) > 15 else int(v)
    if n is None:
        return None
    if isinstance(n, float):
        if not math.isfinite(n) or not n.is_integer():
            return None
        n = int(n)
    if n < _TS_MIN or n >= _TS_MAX:
        return None
    return n


_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(Z|[+-]\d{2}(?::?\d{2})?)$", re.ASCII)


def parse_iso_ms(v):
    """An ISO-like date-time WITH an explicit zone -> epoch ms. MUST MATCH parseIsoMs."""
    if not isinstance(v, str):
        return None
    m = _ISO.match(v.strip())
    if not m:
        return None
    y, mo, d, hh, mm, ss = (int(m.group(i)) for i in range(1, 7))
    frac, zone = m.group(7), m.group(8)
    if mo < 1 or mo > 12 or d < 1 or d > 31 or hh > 23 or mm > 59 or ss > 59:
        return None
    try:
        base = _dt.datetime(y, mo, d, hh, mm, ss, tzinfo=_dt.timezone.utc)
    except ValueError:
        return None
    ms = int(base.timestamp()) * 1000
    if frac:
        ms += int(frac[:3].ljust(3, "0"))
    if zone != "Z":
        sign = -1 if zone[0] == "-" else 1
        digits = zone[1:].replace(":", "")
        oh = int(digits[:2])
        om = int(digits[2:4]) if len(digits) > 2 else 0
        if oh > 23 or om > 59:
            return None
        ms -= sign * (oh * 60 + om) * 60_000
    return parse_time_ms(ms)


def _json_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            out = json_loads(v)
        except ValueError:
            return None
        return out if isinstance(out, list) else None
    return None


_PRICE = re.compile(r"-?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", re.ASCII)


def _price_num(v):
    if _is_num(v):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, str) and _PRICE.fullmatch(v.strip()):
        f = float(v.strip())
        return f if math.isfinite(f) else None
    return None


def _outcome_of_pair(v):
    p = _json_list(v)
    if not p or len(p) != 2:
        return None

    def val(x):
        if _is_num(x) and math.isfinite(x):
            return float(x)
        if isinstance(x, str) and x.strip() in ("0", "1", "0.5"):
            return float(x.strip())
        return None

    a, b = val(p[0]), val(p[1])
    if a == 1 and b == 0:
        return "UP"
    if a == 0 and b == 1:
        return "DOWN"
    if a == 0.5 and b == 0.5:
        return "TIE"
    return None


def _is_gamma(e: dict) -> bool:
    return "clobTokenIds" in e or "conditionId" in e


def generic_record(entry):
    """One markets-file entry -> (id, record) or (id, None, why). MUST MATCH genericRecord."""
    if not isinstance(entry, dict):
        return None, None, "entry is not an object"
    gamma_only = _is_gamma(entry)
    g = entry if gamma_only else (entry["gamma"] if isinstance(entry.get("gamma"), dict) else None)
    own = {} if gamma_only else entry

    def has(k):
        return k in own

    mid = own.get("market_id")
    if mid is None:
        mid = own.get("condition_id")
    if mid is None and g is not None:
        mid = g.get("conditionId")
    if not isinstance(mid, str) or mid == "":
        return None, None, "no market id (market_id / conditionId as a string)"

    def reject(why):
        return mid, None, why

    tokens = own.get("token_ids") if has("token_ids") else _json_list(g.get("clobTokenIds") if g else None)
    if (not isinstance(tokens, list) or len(tokens) != 2
            or not all(isinstance(t, str) and t != "" for t in tokens) or tokens[0] == tokens[1]):
        return reject("token_ids must be two distinct non-empty strings (a number token id has already lost digits)")
    labels = own.get("labels") if has("labels") else _json_list(g.get("outcomes") if g else None)
    if (not isinstance(labels, list) or len(labels) != 2
            or not all(isinstance(lb, str) and lb.strip() != "" for lb in labels)):
        return reject("labels must be two non-empty strings (or Gamma outcomes)")
    low = [lb.strip().lower() for lb in labels]
    if low[0] == low[1]:
        return reject("labels must differ (case-insensitively)")
    if low[1] == "up" or low[0] == "down":
        return reject("a label named Up/Down must sit at that index (0 = Up, 1 = Down)")

    def opt_time(k):
        if not has(k) or own[k] is None:
            return True, None
        v = parse_time_ms(own[k])
        return (False, None) if v is None else (True, v)

    ok, trading_end = opt_time("trading_end_ts_ms")
    if not ok:
        return reject("trading_end_ts_ms must be integer epoch milliseconds")
    ok, start = opt_time("start_ts_ms")
    if not ok:
        return reject("start_ts_ms must be integer epoch milliseconds")

    fee = None
    if has("fee"):
        f = own["fee"]
        if f is None:
            fee = None
        elif isinstance(f, dict) and f.get("model") == "none":
            fee = {"model": "none"}
        elif (isinstance(f, dict) and f.get("model") == FEE_MODEL_PM and _is_num(f.get("rate"))
              and math.isfinite(f["rate"]) and 0 <= f["rate"] <= 1):
            fee = {"model": FEE_MODEL_PM, "rate": f["rate"]}
        else:
            return reject(f'fee must be null, {{model:"none"}} or {{model:"{FEE_MODEL_PM}", rate}}')
    elif g is not None:
        fee = _fee_of({"raw": g})

    evidence = []
    if has("resolution") and own["resolution"] is not None:
        r = own["resolution"]
        ts = parse_time_ms(r.get("resolved_ts_ms")) if isinstance(r, dict) else None
        if not isinstance(r, dict) or r.get("outcome") not in OUTCOMES or ts is None:
            return reject('resolution must be {outcome: "UP"|"DOWN"|"TIE", resolved_ts_ms}')
        evidence.append({"outcome": r["outcome"], "ts": ts, "source": "explicit"})
    if g is not None and g.get("umaResolutionStatus") == "resolved":
        outcome = _outcome_of_pair(g.get("outcomePrices"))
        if outcome is None:
            return reject("Gamma says resolved but outcomePrices is not exactly 1/0, 0/1 or 0.5/0.5")
        ts = parse_iso_ms(g.get("umaEndDate"))
        if ts is None:
            ts = parse_iso_ms(g.get("closedTime"))
        evidence.append({"outcome": outcome, "ts": ts, "source": "gamma"})

    def s(v):
        return v if isinstance(v, str) and v != "" else None

    def pick(k, gk):
        return own.get(k) if has(k) else (g.get(gk) if g is not None else None)

    return mid, {
        "kind": GENERIC_KIND,
        "venue": "polymarket",
        "market_id": mid,
        "slug": s(pick("slug", "slug")),
        "question": s(pick("question", "question")),
        "token_ids": [tokens[0], tokens[1]],
        "labels": [labels[0], labels[1]],
        "fee": fee,
        "tick_size": _price_num(pick("tick_size", "orderPriceMinTickSize")),
        "min_size": _price_num(pick("min_size", "orderMinSize")),
        "trading_end_ts_ms": trading_end,
        "start_ts_ms": start,
        "asset": None,
        "interval": None,
        "strike": None,
        "evidence": evidence,
    }, None


def index_generic_markets(entries):
    """MUST MATCH indexGenericMarkets."""
    if not isinstance(entries, list):
        raise DatasetError("the markets file must hold a list of markets")
    markets, rejected, by_slug, by_token = {}, [], {}, {}
    for entry in entries:
        mid, rec, why = generic_record(entry)
        if rec is None:
            rejected.append({"market_id": mid, "why": why})
            continue
        if mid in markets:
            raise DatasetError(f"market {mid} appears twice in the markets file")
        if rec["slug"] is not None:
            if rec["slug"] in by_slug:
                raise DatasetError(f"slug {rec['slug']} names two markets")
            by_slug[rec["slug"]] = mid
        for t in rec["token_ids"]:
            if t in by_token:
                raise DatasetError(f"token {t} belongs to two markets")
            by_token[t] = mid
        markets[mid] = rec
    return markets, rejected, by_slug, by_token


def _to_ticks(px):
    from ._engine import load
    return load()["otengine"].to_ticks(px)


def _strict_ladder(levels):
    if not isinstance(levels, list):
        return None
    out, seen = [], set()
    for lv in levels:
        if isinstance(lv, list):
            px = _price_num(lv[0] if len(lv) > 0 else None)
            size = _price_num(lv[1] if len(lv) > 1 else None)
        elif isinstance(lv, dict):
            px, size = _price_num(lv.get("price")), _price_num(lv.get("size"))
        else:
            px = size = None
        if px is None or size is None or px < 0 or px > 1 or size < 0:
            return None
        t = _to_ticks(px)
        if t in seen:
            return None
        seen.add(t)
        if size > 0:
            out.append([px, size])
    return out


def _side_of(rec, token):
    try:
        i = rec["token_ids"].index(token)
    except ValueError:
        return None
    return "UP" if i == 0 else ("DOWN" if i == 1 else None)


_BOOK_TYPES = ("book", "price_change", "best_bid_ask")


def _upper(v):
    return v.upper() if isinstance(v, str) else None


def _nn(a, b):
    """`a ?? b`."""
    return a if a is not None else b


class FrameRouter:
    """MUST MATCH FrameRouter in runner/generic-data.mjs."""

    def __init__(self, markets, by_slug, by_token):
        self.markets = markets
        self.by_slug = by_slug
        self.by_token = by_token
        self.seq = 0
        self.state = {mid: {"bad": None, "n": 0, "first": None, "last": None, "resolution": None, "buf": []}
                      for mid in markets}
        self.counts = {"frames": 0, "bad_json": 0, "not_an_object": 0, "unattributed": 0,
                       "bad_trade_rows": 0, "ignored": {}}

    def _bad(self, mid, why, at):
        st = self.state.get(mid)
        if st is not None and st["bad"] is None:
            st["bad"] = {"why": why, "at": at}

    def line(self, text, at):
        try:
            obj = json_loads(text)
        except ValueError:
            self.counts["bad_json"] += 1
            return
        self.frame(obj, at)

    def frame(self, obj, at):
        if isinstance(obj, list):
            for x in obj:
                self.frame(x, at)
            return
        self.counts["frames"] += 1
        if not isinstance(obj, dict):
            self.counts["not_an_object"] += 1
            return
        payload = obj["payload"] if isinstance(obj.get("payload"), dict) else obj
        typ = _nn(obj.get("event_type"), payload.get("event_type"))

        ids = []
        unknown = False

        def name(v, lookup):
            nonlocal unknown
            if v is None:
                return
            mid = lookup(js_string(v))
            if mid is None:
                unknown = True
            else:
                ids.append(mid)

        name(obj.get("market_id"), lambda v: v if v in self.markets else None)
        name(obj.get("slug"), lambda v: self.by_slug.get(v))
        name(payload.get("market"), lambda v: v if v in self.markets else None)
        tokens = []
        if typ == "price_change":
            changes = payload.get("price_changes") if isinstance(payload.get("price_changes"), list) else []
            for ch in changes:
                tokens.append(ch.get("asset_id") if isinstance(ch, dict) else None)
        else:
            a = _nn(obj.get("asset_id"), payload.get("asset_id"))
            if a is not None:
                tokens.append(a)
        if not ids and not unknown:
            for t in tokens:
                if isinstance(t, str) and t in self.by_token:
                    ids.append(self.by_token[t])
        distinct = list(dict.fromkeys(ids))
        if not distinct:
            self.counts["unattributed"] += 1
            return
        if len(distinct) > 1 or unknown:
            for mid in distinct:
                self._bad(mid, "frame names conflicting markets", at)
            return
        mid = distinct[0]
        rec = self.markets[mid]
        st = self.state[mid]

        if typ not in _BOOK_TYPES and typ != "last_trade_price" and typ != "market_resolved":
            k = typ if isinstance(typ, str) else "(not a string)"
            self.counts["ignored"][k] = self.counts["ignored"].get(k, 0) + 1
            return
        ts_raw = obj["event_ts_ms"] if "event_ts_ms" in obj else payload.get("timestamp")
        ts = parse_time_ms(ts_raw)
        recv = parse_time_ms(_nn(obj.get("recv_ms"), obj.get("recv_ts_ms")))
        is_book = typ in _BOOK_TYPES
        if ts is None:
            if is_book:
                self._bad(mid, f"{typ} row without a readable timestamp", at)
            elif typ == "market_resolved":
                self._bad(mid, "market_resolved without a readable timestamp", at)
            else:
                self.counts["bad_trade_rows"] += 1
            return
        if st["bad"] is not None:
            return

        if typ == "market_resolved":
            outcome = None
            win = payload.get("winning_asset_id")
            if isinstance(win, str):
                outcome = _side_of(rec, win)
            elif isinstance(payload.get("winning_outcome"), str):
                w = payload["winning_outcome"].strip().lower()
                idx = next((i for i, lb in enumerate(rec["labels"]) if lb.strip().lower() == w), -1)
                outcome = "UP" if idx == 0 else ("DOWN" if idx == 1 else None)
            if outcome is None:
                self._bad(mid, "market_resolved names no outcome of this market", at)
                return
            r = st["resolution"]
            if r is not None and r["outcome"] != outcome:
                self._bad(mid, "market_resolved frames disagree on the outcome", at)
                return
            if r is None or ts < r["ts"]:
                st["resolution"] = {"outcome": outcome, "ts": ts, "source": "frame"}
            return

        def token_side(t):
            return _side_of(rec, t) if isinstance(t, str) else None

        def extra(ev):
            ev["seq"] = self.seq
            self.seq += 1
            ev["recv_ts_ms"] = recv
            return ev

        out = []
        if typ == "book":
            side = token_side(_nn(obj.get("asset_id"), payload.get("asset_id")))
            if not side:
                self._bad(mid, "book row for a token this market does not own", at)
                return
            asks = _strict_ladder(_nn(payload.get("asks"), payload.get("sells")))
            bids = _strict_ladder(_nn(payload.get("bids"), payload.get("buys")))
            if asks is None or bids is None:
                self._bad(mid, "book row with an unreadable or duplicated level", at)
                return
            out.append(extra({"kind": "book", "ts_ms": ts, "snapshot": True, "side": side,
                              "levels": {side: {"asks": asks, "bids": bids}}}))
        elif typ == "price_change":
            changes = payload.get("price_changes")
            if not isinstance(changes, list) or len(changes) == 0:
                self._bad(mid, "price_change without price_changes", at)
                return
            for ch in changes:
                d = ch if isinstance(ch, dict) else {}
                side = token_side(d.get("asset_id"))
                px = _price_num(d.get("price"))
                size = _price_num(d.get("size"))
                s = _upper(d.get("side"))
                if (not side or px is None or size is None or px < 0 or px > 1 or size < 0
                        or (s != "BUY" and s != "SELL")):
                    self._bad(mid, "price_change entry with an unknown token, side, price or size", at)
                    return
                out.append({"kind": "book", "ts_ms": ts, "snapshot": False, "side": side,
                            "ladder": "asks" if s == "SELL" else "bids", "px": px, "size": size})
            for ev in out:
                extra(ev)
        elif typ == "best_bid_ask":
            side = token_side(_nn(obj.get("asset_id"), payload.get("asset_id")))
            bid = _price_num(payload.get("best_bid"))
            ask = _price_num(payload.get("best_ask"))
            if (not side or bid is None or ask is None or bid < 0 or bid > 1 or ask < 0 or ask > 1
                    or bid > ask):
                self._bad(mid, "best_bid_ask row with an unknown token or an unreadable / crossed bound", at)
                return
            out.append(extra({"kind": "book", "ts_ms": ts, "snapshot": False, "bbo": True, "side": side,
                              "bid": bid, "ask": ask}))
        else:
            side = token_side(_nn(obj.get("asset_id"), payload.get("asset_id")))
            px = _price_num(payload.get("price"))
            size = _price_num(payload.get("size"))
            s = _upper(payload.get("side"))
            if (not side or px is None or size is None or px < 0 or px > 1 or not (size > 0)
                    or (s != "BUY" and s != "SELL")):
                self.counts["bad_trade_rows"] += 1
                return
            out.append(extra({"kind": "trade", "ts_ms": ts, "market_id": mid, "px": px, "size": size,
                              "side": side, "taker": s}))
        for ev in out:
            self._push(st, ev)

    def events(self, mid, evs):
        """Internal events (the pmdata adapter), routed like decoded frames."""
        st = self.state.get(mid)
        if st is None or st["bad"] is not None:
            return
        for ev in evs:
            ev["seq"] = self.seq
            self.seq += 1
            ev.setdefault("recv_ts_ms", None)
            self._push(st, ev)

    def resolve(self, mid, evidence):
        st = self.state.get(mid)
        if st is None or st["bad"] is not None:
            return
        r = st["resolution"]
        if r is not None and r["outcome"] != evidence["outcome"]:
            self._bad(mid, "resolution frames disagree on the outcome", None)
            return
        if r is None or evidence["ts"] < r["ts"]:
            st["resolution"] = dict(evidence)

    def _push(self, st, ev):
        st["n"] += 1
        if st["first"] is None or ev["ts_ms"] < st["first"]:
            st["first"] = ev["ts_ms"]
        if st["last"] is None or ev["ts_ms"] > st["last"]:
            st["last"] = ev["ts_ms"]
        st["buf"].append(_dump(ev))

    def take(self, mid):
        st = self.state[mid]
        out, st["buf"] = st["buf"], []
        return out

    def pending(self, mid):
        return len(self.state[mid]["buf"])


# --- dataset directory ------------------------------------------------------

def dataset_files(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        for name in filenames:
            r = f"{rel_dir}/{name}" if rel_dir else name
            full = os.path.join(dirpath, name)
            if (os.path.isfile(full) and not os.path.islink(full)
                    and re.search(r"\.jsonl(\.gz)?$", name) and r != "markets.jsonl"):
                out.append(r)
    return sorted(out, key=lambda s: s.encode("utf-8"))


def markets_file_of(root):
    found = [n for n in ("markets.json", "markets.jsonl") if os.path.isfile(os.path.join(root, n))]
    if len(found) > 1:
        raise DatasetError(f"{root} holds both markets.json and markets.jsonl — keep one")
    return found[0] if found else None


def looks_like_generic_dataset(root) -> bool:
    if not os.path.isdir(root):
        return False
    return markets_file_of(root) is not None


def file_lines(path):
    """Node readline (crlfDelay: Infinity) + String.trim, streamed."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            t = js_trim(line)
            if t:
                yield t


def read_markets_file(root):
    name = markets_file_of(root)
    if name is None:
        raise DatasetError(f"{root} has no markets.json or markets.jsonl")
    path = os.path.join(root, name)
    if name == "markets.json":
        with open(path, encoding="utf-8") as fh:
            try:
                return json_loads(fh.read())
            except ValueError as err:
                raise DatasetError(f"markets.json is not valid JSON: {err}") from None
    out = []
    for i, line in enumerate(file_lines(path), 1):
        try:
            out.append(json_loads(line))
        except ValueError:
            raise DatasetError(f"markets.jsonl line {i} is not valid JSON") from None
    return out


# --- sorting one market's events --------------------------------------------

def _key(ev):
    return (ev["ts_ms"], ev["seq"])


def _parsed(path):
    for line in file_lines(path):
        yield json.loads(line)


def sorted_events(path, count, run_size=SORT_RUN, fan_in=MERGE_FANIN):
    """(ts, seq) order in bounded memory. MUST MATCH sortedEvents."""
    if count <= run_size:
        evs = list(_parsed(path))
        evs.sort(key=_key)
        yield from evs
        return
    runs_dir = f"{path}.runs"
    os.makedirs(runs_dir, exist_ok=True)
    try:
        runs, chunk = [], []

        def flush():
            chunk.sort(key=_key)
            f = os.path.join(runs_dir, f"r{len(runs)}.jsonl")
            with open(f, "w", encoding="utf-8") as fh:
                fh.write("\n".join(_dump(e) for e in chunk) + "\n")
            runs.append(f)
            chunk.clear()

        for ev in _parsed(path):
            chunk.append(ev)
            if len(chunk) >= run_size:
                flush()
        if chunk:
            flush()
        p = 0
        while len(runs) > fan_in:
            nxt = []
            for i in range(0, len(runs), fan_in):
                group = runs[i:i + fan_in]
                f = os.path.join(runs_dir, f"p{p}-{len(nxt)}.jsonl")
                with open(f, "w", encoding="utf-8") as fh:
                    for ev in heapq.merge(*(_parsed(g) for g in group), key=_key):
                        fh.write(_dump(ev) + "\n")
                for g in group:
                    os.remove(g)
                nxt.append(f)
            runs = nxt
            p += 1
        yield from heapq.merge(*(_parsed(r) for r in runs), key=_key)
    finally:
        import shutil
        shutil.rmtree(runs_dir, ignore_errors=True)


# --- valuation --------------------------------------------------------------

def apply_book_event(book, ev, generic=True):
    if ev.get("bbo"):
        book.bbo(ev["ts_ms"], ev["side"], ev.get("bid"), ev.get("ask"), generic)
    elif ev.get("snapshot"):
        book.snapshot(ev["ts_ms"], ev.get("levels") or {})
    elif ev.get("side") and ev.get("ladder"):
        if ev.get("mirror") is True:
            book.delta_mirrored(ev["ts_ms"], ev["side"], ev["ladder"], ev["px"], ev["size"])
        else:
            book.delta(ev["ts_ms"], ev["side"], ev["ladder"], ev["px"], ev["size"])


class QuoteState:
    """The external quote per token. MUST MATCH QuoteState in runner/generic-data.mjs."""

    def __init__(self, book_factory):
        self.book = book_factory()
        self.quote = {"UP": {"bid": None, "ask": None}, "DOWN": {"bid": None, "ask": None}}
        self.last_trade = None

    def _restate(self, side, ladder, ts):
        levels = self.book.ladders[side][ladder].levels
        self.quote[side]["bid" if ladder == "bids" else "ask"] = (
            {"px": levels[0][0] / 10_000, "ts": ts} if levels else None)

    def _touch(self, side, ladder, px, ts):
        cur = self.quote[side]["bid" if ladder == "bids" else "ask"]
        t = _to_ticks(px)
        if cur is None or (t >= _to_ticks(cur["px"]) if ladder == "bids" else t <= _to_ticks(cur["px"])):
            self._restate(side, ladder, ts)

    def apply(self, ev):
        if ev["kind"] == "trade":
            self.last_trade = {"up": ev["px"] if ev["side"] == "UP" else 1 - ev["px"], "at": ev["ts_ms"]}
            return
        if ev["kind"] != "book":
            return
        apply_book_event(self.book, ev, True)
        ts = ev["ts_ms"]
        if ev.get("bbo"):
            self.quote[ev["side"]] = {"bid": {"px": ev["bid"], "ts": ts}, "ask": {"px": ev["ask"], "ts": ts}}
        elif ev.get("snapshot"):
            levels = ev.get("levels") or {}
            for side in ("UP", "DOWN"):
                if levels.get(side) is None:
                    continue
                self._restate(side, "bids", ts)
                self._restate(side, "asks", ts)
        elif ev.get("side") and ev.get("ladder"):
            self._touch(ev["side"], ev["ladder"], ev["px"], ts)
            if ev.get("mirror") is True:
                from ._engine import load
                m = load()["otengine"].js_to_fixed10(1 - ev["px"])
                if 0 <= m <= 1:
                    self._touch("DOWN" if ev["side"] == "UP" else "UP",
                                "bids" if ev["ladder"] == "asks" else "asks", m, ts)

    def top(self, side):
        q = self.quote[side]
        bid = dict(q["bid"]) if q["bid"] is not None else None
        ask = dict(q["ask"]) if q["ask"] is not None else None
        if bid is not None and not (bid["px"] > 0):
            bid = None
        if ask is not None and not (ask["px"] < 1):
            ask = None
        return bid, ask

    def mark(self, ts, max_age_ms=None):
        cands = []
        for side in ("UP", "DOWN"):
            bid, ask = self.top(side)
            if bid is not None and ask is not None and bid["px"] < ask["px"]:
                mid = (bid["px"] + ask["px"]) / 2
                cands.append({"up": mid if side == "UP" else 1 - mid,
                              "source": "up_mid" if side == "UP" else "down_mid", "at": min(bid["ts"], ask["ts"])})
        if self.last_trade is not None:
            cands.append({"up": self.last_trade["up"], "source": "last_trade", "at": self.last_trade["at"]})
        if not cands:
            return {"none": "no_quote"}
        for c in cands:
            age = ts - c["at"]
            if max_age_ms is None or age <= max_age_ms:
                return {"up": c["up"], "source": c["source"], "ts_ms": c["at"], "age_ms": age}
        return {"none": "stale"}


def _book_levels(book):
    out = {}
    for side in ("UP", "DOWN"):
        lad = book.ladders[side]
        out[side] = {
            "asks": [[lv[0] / 10_000, lv[1]] for lv in lad["asks"].levels],
            "bids": [[lv[0] / 10_000, lv[1]] for lv in lad["bids"].levels],
        }
    return out


def resolution_of(rec, frame):
    """MUST MATCH resolutionOf."""
    ev = list(rec["evidence"]) + ([frame] if frame else [])
    if not ev:
        return None, None
    outcomes = list(dict.fromkeys(e["outcome"] for e in ev))
    if len(outcomes) > 1:
        return None, "resolution sources disagree: " + ", ".join(f"{e['source']}={e['outcome']}" for e in ev)

    def pick(src):
        return next((e for e in ev if e["source"] == src and e["ts"] is not None), None)

    chosen = pick("frame") or pick("explicit") or pick("gamma")
    if chosen is None:
        return None, "resolution has an outcome but no time it became public"
    return {
        "outcome": chosen["outcome"],
        "ts": chosen["ts"],
        "source": chosen["source"],
        "sources": [{"source": e["source"], "ts_ms": e["ts"]} for e in ev],
    }, None


def _final_record(rec, start, trade_until, resolution):
    out = {k: v for k, v in rec.items() if k not in ("evidence", "start_ts_ms")}
    out["open_ts_ms"] = start
    out["close_ts_ms"] = resolution["ts"] if resolution else trade_until
    out["trade_until_ts_ms"] = trade_until
    out["outcome"] = resolution["outcome"] if resolution else None
    out["resolved_ts_ms"] = resolution["ts"] if resolution else None
    out["resolution_source"] = resolution["source"] if resolution else None
    return out


def rules_readable(rec) -> bool:
    """MUST MATCH rulesReadable in runner/generic-data.mjs."""
    tick, mn = rec.get("tick_size"), rec.get("min_size")

    def num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    if not num(tick) or not tick > 0:
        return False
    from ._engine import load
    t = load()["otengine"].js_round(tick * 10_000)
    if not t >= 1 or abs(t - tick * 10_000) > 1e-6:
        return False
    return num(mn) and mn > 0


def decode_generic(*, entries, frames=None, feed=None, events_dir, book_factory, mark_max_age_ms=None,
                   order_rules=False):
    """MUST MATCH decodeGeneric. `frames`: iterable of (at, text_line) or (at, parsed_frame, True)."""
    markets, rejected, by_slug, by_token = index_generic_markets(entries)
    if not markets:
        first = f" ({rejected[0]['market_id'] or '?'}: {rejected[0]['why']})" if rejected else ""
        raise DatasetError(f"no usable market in the markets file{first}")
    os.makedirs(events_dir, exist_ok=True)
    router = FrameRouter(markets, by_slug, by_token)
    spool = {}

    def file_of(mid):
        f = spool.get(mid)
        if f is None:
            f = os.path.join(events_dir, f"raw-{len(spool)}.jsonl")
            spool[mid] = f
        return f

    def flush_if_full(force=False):
        for mid in markets:
            n = router.pending(mid)
            if n and (force or n >= _SPOOL_FLUSH):
                with open(file_of(mid), "a", encoding="utf-8") as fh:
                    fh.write("\n".join(router.take(mid)) + "\n")

    since = 0
    if frames is not None:
        for item in frames:
            if len(item) == 3:
                router.frame(item[1], item[0])
            else:
                router.line(item[1], item[0])
            since += 1
            if since >= _SPOOL_FLUSH:
                since = 0
                flush_if_full()
    if feed is not None:
        feed(router)
    flush_if_full(True)

    out = []
    unusable = [{"market_id": r["market_id"], "why": r["why"]} for r in rejected]
    dropped = {"after_trading_end": 0}
    k = 0
    for mid, rec in markets.items():
        st = router.state[mid]
        raw = spool.get(mid)

        def done():
            if raw:
                try:
                    os.remove(raw)
                except FileNotFoundError:
                    pass

        if st["bad"] is not None:
            u = {"market_id": mid, "why": st["bad"]["why"]}
            if st["bad"]["at"]:
                u["at"] = st["bad"]["at"]
            unusable.append(u)
            done()
            continue
        if st["n"] == 0:
            unusable.append({"market_id": mid, "why": "no frames for this market"})
            done()
            continue
        resolution, why = resolution_of(rec, st["resolution"])
        if why:
            unusable.append({"market_id": mid, "why": why})
            done()
            continue
        cands = [st["last"]]
        if rec["trading_end_ts_ms"] is not None:
            cands.append(rec["trading_end_ts_ms"])
        if resolution is not None:
            cands.append(resolution["ts"] - 1)
        trade_until = min(cands)
        start = rec["start_ts_ms"] if rec["start_ts_ms"] is not None else st["first"]
        if start > trade_until:
            unusable.append({"market_id": mid, "why": "empty window: trading ends before the window starts"})
            done()
            continue

        events_file = f"m-{k}.jsonl"
        k += 1
        dest = os.path.join(events_dir, events_file)
        pre = book_factory()
        pre_book = False
        quotes = QuoteState(book_factory)
        baseline = book_factory()
        st_out = {"up": None, "down": None, "n": 0, "books": 0, "opened": False}
        lines = []

        def emit(ev, fh):
            if ev["kind"] == "book":
                st_out["books"] += 1
                if st_out["up"] is None:
                    apply_book_event(baseline, ev, True)
                    u, d = baseline.best("UP"), baseline.best("DOWN")
                    if u is not None and d is not None:
                        st_out["up"], st_out["down"] = u, d
            lines.append(_dump(ev))
            st_out["n"] += 1
            if len(lines) >= 4096:
                fh.write("\n".join(lines) + "\n")
                lines.clear()

        def open_window(fh):
            st_out["opened"] = True
            if not pre_book:
                return
            emit({"kind": "book", "ts_ms": start, "snapshot": True, "synthetic": True,
                  "levels": _book_levels(pre), "seq": -1, "recv_ts_ms": None}, fh)

        with open(dest, "w", encoding="utf-8") as fh:
            for ev in sorted_events(raw, st["n"]):
                if ev["ts_ms"] > trade_until:
                    dropped["after_trading_end"] += 1
                    continue
                quotes.apply(ev)
                if ev["ts_ms"] < start:
                    if ev["kind"] == "book":
                        apply_book_event(pre, ev, True)
                        pre_book = True
                    continue
                if not st_out["opened"]:
                    open_window(fh)
                emit(ev, fh)
            if not st_out["opened"]:
                open_window(fh)
            if lines:
                fh.write("\n".join(lines) + "\n")
        done()
        if st_out["books"] == 0:
            os.remove(dest)
            unusable.append({"market_id": mid, "why": "no book inside the trading window"})
            continue
        if order_rules and not rules_readable(rec):
            os.remove(dest)
            unusable.append({"market_id": mid,
                             "why": "order_rules \"venue\" needs this market's tick_size (on a 0.0001 grid) and min_size"})
            continue
        terminal = None
        if resolution is None:
            m = quotes.mark(trade_until, mark_max_age_ms)
            terminal = ({"none": m["none"]} if "none" in m else
                        {"UP": m["up"], "DOWN": 1 - m["up"], "source": m["source"],
                         "ts_ms": m["ts_ms"], "age_ms": m["age_ms"]})
        out.append({
            "market": _final_record(rec, start, trade_until, resolution),
            "stream": None,
            "day": None,
            "events_file": dest,
            "eventsFile": events_file,
            "n": st_out["n"],
            "up_px": st_out["up"],
            "down_px": st_out["down"],
            "terminal": terminal,
            "resolution_sources": resolution["sources"] if resolution else [],
        })
    counts = dict(router.counts)
    counts.update(dropped)
    return {"markets": out, "unusable": unusable, "counts": counts}


def decode_generic_dir(root, *, events_dir, book_factory, mark_max_age_ms=None, order_rules=False):
    entries = read_markets_file(root)
    files = dataset_files(root)

    def frames():
        for rel in files:
            for i, line in enumerate(file_lines(os.path.join(root, rel)), 1):
                yield (f"{rel}:{i}", line)

    out = decode_generic(entries=entries, frames=frames(), events_dir=events_dir,
                         book_factory=book_factory, mark_max_age_ms=mark_max_age_ms, order_rules=order_rules)
    out["inputs"] = files
    return out


# --- in-memory datasets -----------------------------------------------------

def _rows(table):
    """A pandas / polars DataFrame or an iterable of dicts -> list of dicts."""
    if table is None:
        return []
    if hasattr(table, "to_dicts"):          # polars
        return table.to_dicts()
    if hasattr(table, "to_dict"):           # pandas
        return table.to_dict("records")
    return [dict(r) for r in table]


def _plain(v):
    """numpy / pandas scalars and arrays -> plain Python values."""
    if v is None:
        return None
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        return v.tolist()
    if hasattr(v, "item") and not isinstance(v, (str, bytes, list, dict)):
        try:
            return v.item()
        except (TypeError, ValueError):
            return v
    return v


def _ms(v):
    """A pmdata timestamp -> epoch ms: integer ms, or a datetime (naive = UTC)."""
    v = _plain(v)
    if isinstance(v, float) and v != v:
        return None
    if hasattr(v, "value") and hasattr(v, "tz_localize"):   # pandas Timestamp
        return parse_time_ms(int(v.value) // 1_000_000)
    if isinstance(v, _dt.datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=_dt.timezone.utc)
        return parse_time_ms(int(round(v.timestamp() * 1000)))
    return parse_time_ms(v)


def _levels(prices, sizes):
    prices, sizes = _plain(prices), _plain(sizes)
    if prices is None and sizes is None:
        return []
    if not isinstance(prices, list) or not isinstance(sizes, list) or len(prices) != len(sizes):
        return None
    return [[p, s] for p, s in zip(prices, sizes)]


def _mirror(levels):
    from ._engine import load
    fixed = load()["otengine"].js_to_fixed10
    out = []
    for px, size in levels:
        m = fixed(1 - px)
        if 0 <= m <= 1:
            out.append([m, size])
    return out


def _pmdata_feed(mid, rec, l2, trades):
    """The single-book table shape -> internal events on `mid`.

    The l2 table carries ONE book (outcome index 0); the other token is its
    mirror, applied atomically (a snapshot names both tokens, a delta carries
    `mirror: true`). Trades come only from `trades`, so each public print
    reaches the engine once.
    """
    labels = [lb.strip().lower() for lb in rec["labels"]]

    # The slug every row must carry when it carries one: the market's own, or --
    # for a record without one -- the first a row names. Two slugs in one table
    # are two markets, whichever is "right".
    expected = {"slug": rec["slug"]}

    def other_market(row):
        """A row that names a market, naming another one -- never routed here."""
        cid = _plain(row.get("condition_id"))
        if cid is not None and js_string(cid) != mid:
            return f"condition_id {js_string(cid)}"
        slug = _plain(row.get("market_slug"))
        if slug is not None:
            slug = js_string(slug)
            if expected["slug"] is None:
                expected["slug"] = slug
            elif slug != expected["slug"]:
                return f"market_slug {slug}"
        return None

    def feed(router):
        evs = []
        for i, row in enumerate(_rows(l2), 1):
            at = f"l2:{i}"
            other = other_market(row)
            if other is not None:
                router._bad(mid, f"pmdata row names another market ({other})", at)
                return
            typ = row.get("event_type")
            ts = _ms(row.get("timestamp"))
            if typ not in ("book", "price_change", "market_resolved"):
                k = typ if isinstance(typ, str) else "(not a string)"
                router.counts["ignored"][k] = router.counts["ignored"].get(k, 0) + 1
                continue
            if ts is None:
                router._bad(mid, f"{typ} row without a readable timestamp", at)
                return
            if typ == "market_resolved":
                w = _plain(row.get("winning_outcome"))
                w = w.strip().lower() if isinstance(w, str) else None
                if w not in labels:
                    router._bad(mid, "market_resolved names no outcome of this market", at)
                    return
                router.resolve(mid, {"outcome": "UP" if labels.index(w) == 0 else "DOWN", "ts": ts, "source": "frame"})
                continue
            if typ == "book":
                bids = _strict_ladder(_levels(row.get("bid_prices"), row.get("bid_sizes")))
                asks = _strict_ladder(_levels(row.get("ask_prices"), row.get("ask_sizes")))
                if bids is None or asks is None:
                    router._bad(mid, "book row with an unreadable or duplicated level", at)
                    return
                evs.append({"kind": "book", "ts_ms": ts, "snapshot": True,
                            "levels": {"UP": {"asks": asks, "bids": bids},
                                       "DOWN": {"asks": _mirror(bids), "bids": _mirror(asks)}}})
                continue
            px = _price_num(_plain(row.get("pc_price")))
            size = _price_num(_plain(row.get("pc_size")))
            s = _upper(_plain(row.get("pc_side")))
            if px is None and size is None:
                continue        # pmdata writes price_change rows with no change on them
            if px is None or size is None or px < 0 or px > 1 or size < 0 or s not in ("BUY", "SELL"):
                router._bad(mid, "price_change row with an unreadable side, price or size", at)
                return
            evs.append({"kind": "book", "ts_ms": ts, "snapshot": False, "side": "UP",
                        "ladder": "asks" if s == "SELL" else "bids", "px": px, "size": size, "mirror": True})
        for i, row in enumerate(_rows(trades), 1):
            other = other_market(row)
            if other is not None:
                router._bad(mid, f"pmdata trade names another market ({other})", f"trades:{i}")
                return
            ts = _ms(row.get("timestamp"))
            o = _plain(row.get("outcome"))
            o = o.strip().lower() if isinstance(o, str) else None
            px = _price_num(_plain(row.get("price")))
            size = _price_num(_plain(row.get("size")))
            s = _upper(_plain(row.get("side")))
            if (ts is None or o not in labels or px is None or size is None or px < 0 or px > 1
                    or not (size > 0) or s not in ("BUY", "SELL")):
                router.counts["bad_trade_rows"] += 1
                continue
            # A print on the second outcome keeps its own price and side: the
            # engine canonicalises it, so converting here would do it twice.
            evs.append({"kind": "trade", "ts_ms": ts, "market_id": mid, "px": px, "size": size,
                        "side": "UP" if labels.index(o) == 0 else "DOWN", "taker": s})
        router.events(mid, evs)

    return feed


class Dataset:
    """Generic data for `run()`: any Polymarket binary market, your own data.

        Dataset.from_dir("my-data")                      # markets.json + *.jsonl frames
        Dataset.from_frames(markets, frames)             # the same, in memory
        Dataset.from_pmdata(l2, trades, market=gamma)    # single-book table shape

    Datasets add up (`a + b`): one run over every market of both.
    """

    def __init__(self, entries=None, frames=None, pmdata=None, root=None):
        self.entries = list(entries or [])
        self.frames = list(frames or [])
        self.pmdata = list(pmdata or [])
        self.root = root

    def __add__(self, other):
        if not isinstance(other, Dataset):
            return NotImplemented
        if self.root or other.root:
            raise DatasetError("a directory dataset cannot be added to another; put both in one directory")
        return Dataset(self.entries + other.entries, self.frames + other.frames, self.pmdata + other.pmdata)

    @classmethod
    def from_dir(cls, path):
        if not looks_like_generic_dataset(path):
            raise DatasetError(f"{path} has no markets.json or markets.jsonl at its top")
        return cls(root=os.fspath(path))

    @classmethod
    def from_frames(cls, markets, frames):
        """`markets`: generic records and/or Gamma market objects. `frames`: CLOB
        market-channel frames (dicts, or JSON text lines), raw or as archive rows."""
        return cls(entries=list(markets), frames=[list(frames)])

    @classmethod
    def from_pmdata(cls, l2, trades=None, *, market=None):
        """The single-book l2 / trades tables (pandas, polars or lists of dicts).

        `market`: a Gamma market object or generic record for this market (for
        its labels, fee schedule and resolution). Without it, the record is built
        from the l2 rows' condition_id / market_slug with labels ["Yes", "No"].
        """
        rows = _rows(l2)
        if market is None:
            first = rows[0] if rows else {}
            cid = _plain(first.get("condition_id"))
            if not isinstance(cid, str) or not cid:
                raise DatasetError("from_pmdata needs market=... or a condition_id column")
            market = {"market_id": cid, "slug": _plain(first.get("market_slug")),
                      "token_ids": [f"{cid}:0", f"{cid}:1"], "labels": ["Yes", "No"]}
        return cls(entries=[market], pmdata=[(market, rows, trades)])

    def decode(self, *, events_dir, book_factory, mark_max_age_ms=None, order_rules=False):
        if self.root:
            return decode_generic_dir(self.root, events_dir=events_dir, book_factory=book_factory,
                                      mark_max_age_ms=mark_max_age_ms, order_rules=order_rules)

        def frames():
            for s, source in enumerate(self.frames):
                for i, f in enumerate(source, 1):
                    at = f"frames[{s}]:{i}"
                    yield (at, f) if isinstance(f, str) else (at, f, True)

        def feed(router):
            for market, rows, trades in self.pmdata:
                mid, rec, why = generic_record(market)
                if rec is None:
                    continue
                _pmdata_feed(mid, rec, rows, trades)(router)

        out = decode_generic(entries=self.entries, frames=frames(), feed=feed, events_dir=events_dir,
                             book_factory=book_factory, mark_max_age_ms=mark_max_age_ms, order_rules=order_rules)
        # Markets fed through the single-book table shape are one book and its
        # mirror (mirror_depth "shared" needs to know). Internal, never visible
        # to a strategy (generic_view is a whitelist).
        single = {generic_record(m)[0] for m, _rows, _t in self.pmdata}
        for m in out["markets"]:
            if m["market"]["market_id"] in single:
                m["market"]["book_model"] = "single_mirrored"
        out["inputs"] = []
        return out
