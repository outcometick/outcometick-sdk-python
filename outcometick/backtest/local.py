"""Reading an unpacked archive off disk, one day at a time.

A port of cli/local-data.mjs (what `ot run` reads) and runner/spool-day.mjs
(how a day is split into one event file per market). Same file selection, same
read order, same per-market spool, so a pure-Python run replays the identical
events a queued run does.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import tempfile

from ._js import json_loads, js_trim
from .datasets import (
    archive_datasets_for_day, file_matches_run, normalize_intervals, ordered_feed, settlement_paths_for,
)
from .decode import (
    build_slug_index, events_from_row, finalise_market, index_markets, market_unusable, parse_row,
    sort_markets_for_replay,
)
from .taxonomy import classify_path

_DAY = re.compile(r"(\d{4}-\d{2}-\d{2})", re.ASCII)
_SPOOL_FLUSH_ROWS = 512


def walk(root: str, prefix: str = "", depth: int = 0) -> list:
    """Every regular file under root, archive-relative, in directory order.

    Symlinks are skipped in both directions, like the JS: an archive is
    something downloaded from the internet, and a link inside it could name any
    file on the machine.
    """
    out: list = []
    if depth > 12:
        return out
    try:
        entries = list(os.scandir(os.path.join(root, prefix) if prefix else root))
    except OSError:
        return out
    for e in entries:
        rel = f"{prefix}/{e.name}" if prefix else e.name
        try:
            if e.is_symlink():
                continue
            if e.is_dir(follow_symlinks=False):
                out.extend(walk(root, rel, depth + 1))
            elif e.is_file(follow_symlinks=False):
                out.append(rel)
        except OSError:
            continue
    return out


def day_of_path(rel: str):
    m = _DAY.search(rel)
    return m.group(1) if m else None


def read_rows(root: str, rel: str):
    """Parsed rows of one archive file, gunzipping by name, CSV or JSONL."""
    full = os.path.join(root, rel)
    opener = gzip.open if rel.endswith(".gz") else open
    is_csv = ".csv" in rel
    # Universal newlines end a line at \n, \r\n or a lone \r -- the same three
    # Node's readline (crlfDelay: Infinity) splits on. Streamed: a day of
    # best_bid_ask decompresses to gigabytes.
    with opener(full, "rb") as raw, io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=None) as fh:
        yield from _rows(fh, is_csv)


def _rows(fh, is_csv):
    header = None
    for line in fh:
        s = js_trim(line)
        if not s:
            continue
        if is_csv and header is None:
            header = s.split(",")
            continue
        row = parse_row(s, is_csv=is_csv, header=header, loads=json_loads)
        if row is not None:
            yield row


def local_days(root: str) -> list:
    return sorted({d for d in (day_of_path(r) for r in walk(root)) if d})


def archive_venues(root: str) -> set:
    return {classify_path(r)["venue"] for r in walk(root) if classify_path(r)["dataset"] != "other"}


def default_venue(venues: set) -> str:
    return "predict" if len(venues) == 1 and "predict" in venues else "polymarket"


def looks_like_archive(root: str) -> bool:
    if not os.path.isdir(root):
        return False
    return any(classify_path(r)["dataset"] != "other" for r in walk(root))


# --- JSON round trip ------------------------------------------------------

def jsonify(v):
    """The value a JSON.stringify -> JSON.parse round trip leaves behind.

    The queue hands every event and market header to the harness as JSON, so an
    integral double arrives in Python as an int and -0 arrives as 0. A
    pure-Python run hands them over in-process; without this, a timestamp would
    reach the strategy as 1788796800000.0 instead of 1788796800000 and every
    `ctx.log` line that prints one would differ.
    """
    if isinstance(v, float):
        if v == 0:
            return 0
        if v.is_integer() and abs(v) < 1e21:
            return int(v)
        return v
    if isinstance(v, dict):
        return {k: jsonify(x) for k, x in v.items()}
    if isinstance(v, list):
        return [jsonify(x) for x in v]
    return v


def _dump(ev) -> str:
    return json.dumps(jsonify(ev), separators=(",", ":"), ensure_ascii=False)


def read_event_lines(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            t = line.strip()
            if t:
                yield json.loads(t)


# --- one day --------------------------------------------------------------

def _market_day_key(m):
    mk = (m or {}).get("market", m) or {}
    if not mk.get("asset"):
        return None
    iv = mk.get("interval")
    return f"{str(mk['asset']).upper()}|{'none' if iv is None else iv}"


def spool_day(*, day, markets, by_slug, throttle, events_dir, feed, rows_of, book_factory):
    """Decode a day's files into one event file per usable market."""
    os.makedirs(events_dir, exist_ok=True)
    buffers: dict = {}
    counts: dict = {}
    parts: dict = {}

    def part_of(key):
        if key not in parts:
            parts[key] = os.path.join(events_dir, f"{day}-{len(parts)}.part")
        return parts[key]

    def flush(key):
        buf = buffers.get(key)
        if not buf:
            return
        buffers[key] = []
        with open(part_of(key), "a", encoding="utf-8") as fh:
            fh.write("".join(buf))

    bbo_seen: set = set()
    try:
        for f in feed:
            meta = classify_path(f)
            if meta["dataset"] == "markets":
                continue
            for row in rows_of(f):
                for market_id, ev in events_from_row(f, row, markets, by_slug, throttle, meta):
                    if ev.get("bbo"):
                        bbo_seen.add(_market_day_key(markets.get(market_id)))
                    key = str(market_id)
                    if key not in markets:
                        continue
                    buffers.setdefault(key, []).append(_dump(ev) + "\n")
                    counts[key] = counts.get(key, 0) + 1
                    if len(buffers[key]) >= _SPOOL_FLUSH_ROWS:
                        flush(key)
        for key in list(buffers):
            flush(key)
    except BaseException:
        for p in parts.values():
            _rm(p)
        raise

    out, unusable = [], []
    try:
        for market_id, market in markets.items():
            events = []
            if market_id in counts:
                with open(parts[market_id], encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            try:
                                events.append(json.loads(line))
                            except ValueError:
                                pass
                _rm(parts[market_id])
            fin = finalise_market(events, market, book_factory) if market else {"events": [], "up_px": None, "down_px": None}
            why = market_unusable(market, fin["events"])
            if why:
                unusable.append({"market_id": market_id, "asset": (market or {}).get("asset"), "day": day, "why": why})
                continue
            fd, events_file = tempfile.mkstemp(prefix=f"{day}-", suffix=".jsonl", dir=events_dir)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for e in fin["events"]:
                    fh.write(_dump(e) + "\n")
            out.append({
                "market": {
                    "market_id": market["market_id"],
                    "asset": market["asset"],
                    "interval": market["interval"],
                    "strike": market["strike"],
                    "outcome": market["outcome"],
                    "open_ts_ms": market["open_ts_ms"],
                    "close_ts_ms": market["close_ts_ms"],
                    # Polymarket only, like the JS: a Predict record has no `fee`.
                    **({"fee": market["fee"]} if "fee" in market else {}),
                },
                "events_file": events_file,
                "stream": market["stream"],
                "day": day,
                "up_px": fin["up_px"],
                "down_px": fin["down_px"],
            })
    except BaseException:
        for p in parts.values():
            _rm(p)
        raise
    return {"out": out, "unusable": unusable, "bbo_seen": bbo_seen}


def _rm(p):
    try:
        os.remove(p)
    except OSError:
        pass


def load_local_day(*, root, day, venue, assets, datasets, intervals, throttle, events_dir, book_factory):
    archive_datasets = archive_datasets_for_day(datasets=datasets, venue=venue, day=day, frm=day, to=day)
    want_intervals = normalize_intervals(intervals)
    all_paths = walk(root)
    wanted = [r for r in all_paths if day_of_path(r) == day and file_matches_run(
        r, venue=venue, assets=assets, archive_datasets=archive_datasets, intervals=want_intervals)]
    if not wanted:
        return {"markets": [], "reason": f"no files for {day} under {root}"}

    market_rows = []
    for rel in wanted:
        if classify_path(rel)["dataset"] == "markets":
            market_rows.extend(read_rows(root, rel))
    indexed = index_markets(market_rows, venue=venue)
    in_scope = {str(a).upper() for a in (assets or [])}
    want_iv = {str(i) for i in want_intervals}
    markets = {}
    for mid, m in indexed.items():
        if m["asset"] and in_scope and str(m["asset"]).upper() not in in_scope:
            continue
        if m["interval"] and want_iv and str(m["interval"]) not in want_iv:
            continue
        markets[mid] = m
    if not markets:
        return {"markets": [], "reason": f"no market metadata for {day}"}
    by_slug = build_slug_index(markets)

    day_paths = [r for r in all_paths if day_of_path(r) == day]
    feed = ordered_feed(wanted + settlement_paths_for(
        list(markets.values()), day_paths, venue=venue, assets=assets, already=wanted))

    spooled = spool_day(day=day, markets=markets, by_slug=by_slug, throttle=throttle, events_dir=events_dir,
                        feed=feed, rows_of=lambda rel: read_rows(root, rel), book_factory=book_factory)
    out, unusable = spooled["out"], spooled["unusable"]
    reason = None
    if not out and unusable:
        reason = f"{len(unusable)} market(s) unusable: {unusable[0]['why']}"
    return {
        "markets": sort_markets_for_replay(out),
        "inputs": feed,
        "bbo_keys": sorted(k for k in spooled["bbo_seen"] if k),
        "unusable": unusable,
        "reason": reason,
    }


# --- decoded-day cache ----------------------------------------------------
#
# Decoding a day is the slow part of a local run (minutes for a busy day in
# pure Python) and the part that does not change while a strategy is being
# iterated on. So a decoded day is kept under the user's cache directory and
# reused, keyed on everything that changes what decoding produces:
#
#   - the decoder itself: a hash of this package's decoding source files and of
#     the engine file whose Book prices the opening quotes. Any edit to them is
#     a different key, so a stale entry can never be read back;
#   - the run shape: venue, day, assets, datasets, intervals, and the book
#     cadence chosen for the RANGE (it changes with the range, see BookThrottle);
#   - the day's files: every archive file dated that day, by path and by the
#     SHA-256 of its bytes. Not size and modification time: two archives with
#     equal metadata and different contents (or a file replaced in place with its
#     timestamp kept) would otherwise share an entry. Hashing a day is under a
#     second against minutes of decoding.
#
# An entry is written to a temporary directory and renamed into place, so a run
# interrupted half way leaves nothing that could be mistaken for a decoded day.

import hashlib as _hashlib
import shutil as _shutil

_SOURCES = ("_contract.json", "_js.py", "datasets.py", "decode.py", "local.py", "taxonomy.py")


def _decoder_digest(engine_file: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    h = _hashlib.sha256()
    for name in _SOURCES:
        with open(os.path.join(here, name), "rb") as fh:
            h.update(name.encode() + b"\0" + fh.read() + b"\0")
    with open(engine_file, "rb") as fh:
        h.update(b"otengine\0" + fh.read())
    return h.hexdigest()


def cache_key(*, root, day, venue, assets, datasets, intervals, throttle, engine_file) -> str:
    files = []
    for rel in sorted(r for r in walk(root) if day_of_path(r) == day):
        h = _hashlib.sha256()
        with open(os.path.join(root, rel), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        files.append([rel, h.hexdigest()])
    spec = {
        "decoder": _decoder_digest(engine_file),
        "venue": venue,
        "day": day,
        "assets": sorted(str(a).upper() for a in (assets or [])),
        "datasets": sorted(datasets or []),
        "intervals": sorted(intervals or []),
        "throttle": {"per_asset": dict(sorted(throttle.per_asset.items())), "fallback": throttle.fallback},
        "files": files,
    }
    return _hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:24]


def load_local_day_cached(*, cache_root, engine_file, **kw):
    """load_local_day, served from the decoded-day cache when it can be."""
    key = cache_key(root=kw["root"], day=kw["day"], venue=kw["venue"], assets=kw["assets"],
                    datasets=kw["datasets"], intervals=kw["intervals"], throttle=kw["throttle"],
                    engine_file=engine_file)
    entry = os.path.join(cache_root, key)
    meta_path = os.path.join(entry, "day.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        for m in meta["markets"]:
            m["events_file"] = os.path.join(entry, m["events_file"])
        return meta
    os.makedirs(cache_root, exist_ok=True)
    work = tempfile.mkdtemp(prefix=".tmp-", dir=cache_root)
    try:
        loaded = load_local_day(**{**kw, "events_dir": work})
        meta = {**loaded, "markets": [{**m, "events_file": os.path.basename(m["events_file"])}
                                      for m in loaded["markets"]]}
        with open(os.path.join(work, "day.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        try:
            os.replace(work, entry)
            work = None
        except OSError:
            # Another run decoded the same day first; theirs is as good as ours.
            # Anything else (a read-only cache, say) is an error worth seeing.
            if not os.path.isfile(meta_path):
                raise
        return load_local_day_cached(cache_root=cache_root, engine_file=engine_file, **kw)
    finally:
        if work is not None:
            _shutil.rmtree(work, ignore_errors=True)
