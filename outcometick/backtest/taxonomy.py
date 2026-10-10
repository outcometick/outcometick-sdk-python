"""Archive path -> venue / dataset / asset / interval.

A port of `classifyPath` in api/lib/data-taxonomy.mjs and `venueOfPath` in
api/lib/venue-path.mjs. The backtest decides which files to read and which
stream a row belongs to from this, so it must answer exactly as the JS does;
client/backtest/equivalence.test.mjs feeds both every path in the public
samples and compares.
"""

from __future__ import annotations

import re

from ._contract import CONTRACT

ASSETS: tuple = tuple(CONTRACT["assets"])
_TAXONOMY_DATASETS = frozenset(CONTRACT["taxonomy_datasets"])

_PREDICT_BOOK = re.compile(r"([A-Za-z]+)-(\d+[mMhHdD]|HOURLY|DAILY)", re.ASCII)
_POLY_GROUP = re.compile(r"([A-Za-z]+)-(\d+[mMhHdD])", re.ASCII)
_BY_WORD = {"HOURLY": "1h", "DAILY": "1d"}


def venue_of_path(file_path: str) -> str:
    return "predict" if "predict-fun" in str(file_path).lower().split("/") else "polymarket"


def _asset_of(s):
    if not s:
        return None
    up = s.upper()
    for a in ASSETS:
        if up.startswith(a):
            return a
    return None


def _seg(segs, i):
    return segs[i] if i < len(segs) else None


def classify_path(file_path: str) -> dict:
    p = str(file_path)
    segs = p.split("/")
    name = segs[-1] if segs else ""
    venue = venue_of_path(p)
    ext = "csv.gz" if name.endswith(".csv.gz") else "jsonl.gz" if name.endswith(".jsonl.gz") else ""

    def out(dataset, asset=None, interval=None):
        return {"venue": venue, "dataset": dataset, "asset": asset, "interval": interval, "ext": ext}

    s0, s1, s2, s3, s4 = (_seg(segs, i) for i in range(5))

    if s0 == "derived" and s1 == "klines":
        return out("klines", _asset_of(s3), s4)

    if s1 == "predict-fun":
        ds = s2
        if ds == "klines":
            return out("klines", _asset_of(s3), s4)
        if ds == "orderbook":
            m = _PREDICT_BOOK.fullmatch(s3 or "")
            period = m.group(2) if m else None
            interval = None
            if period:
                interval = _BY_WORD.get(period.upper(), period.lower())
            return out("orderbook", _asset_of(m.group(1) if m else s3), interval)
        if ds == "prices":
            return out("prices", _asset_of(s3))
        if ds == "markets":
            return out("markets")
        return out("other")

    if s1 is not None and s1.startswith("chainlink"):
        dataset = "twap30s" if s1 == "chainlink-twap-30s" else "twap60s" if s1 == "chainlink-twap-60s" else "prices"
        return out(dataset, _asset_of(s4))

    if s1 == "polymarket":
        dataset = s3 if s3 in _TAXONOMY_DATASETS else "other"
        m = _POLY_GROUP.fullmatch(s4 or "")
        return out(dataset, _asset_of(m.group(1) if m else s4), m.group(2).lower() if m else None)

    return out("other")
