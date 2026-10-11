"""Reading an unpacked archive off disk, one day at a time.

A port of cli/local-data.mjs (what `ot run` reads) and runner/spool-day.mjs
(how a day is split into one event file per market). Same file selection, same
read order, same per-market spool, so a pure-Python run replays the identical
events a queued run does.
"""

from __future__ import annotations

import base64
import concurrent.futures
import gzip
import io
import json
import os
import pickle
import re
import tempfile
import threading

from ._js import json_loads, js_string, js_trim
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
        # SORTED BY BYTES, as Node's fs.readdir returns them (libuv sorts
        # scandir results with strcmp). os.scandir is raw filesystem order, and
        # the order files are found in decides the order of a day's markets and
        # of the dropped-market list in the report: equal on one filesystem,
        # different on another (CI caught it on ext4 vs the author's machine).
        entries = sorted(os.scandir(os.path.join(root, prefix) if prefix else root),
                         key=lambda e: os.fsencode(e.name))
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
    t = type(v)
    if t is float:
        if v == 0:
            return 0
        if v.is_integer() and abs(v) < 1e21:
            return int(v)
        return v
    if t is dict:
        # Strings, ints, bools and None come back as they went in, and a float
        # is converted inline, so neither costs a call — most of an event's
        # values, and most of the decode time this used to cost. The float rule
        # is the one above, spelled out.
        return {k: x if type(x) in _AS_IS else (
            (0 if x == 0 else int(x) if x.is_integer() and -1e21 < x < 1e21 else x)
            if type(x) is float else jsonify(x)) for k, x in v.items()}
    if t is list:
        return [x if type(x) in _AS_IS else (
            (0 if x == 0 else int(x) if x.is_integer() and -1e21 < x < 1e21 else x)
            if type(x) is float else jsonify(x)) for x in v]
    if t in _AS_IS:
        return v
    if isinstance(v, float):
        return jsonify(float(v))
    if isinstance(v, dict):
        return {k: jsonify(x) for k, x in v.items()}
    if isinstance(v, list):
        return [jsonify(x) for x in v]
    return v


_AS_IS = frozenset((str, int, bool, type(None)))


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
            # The text each event was spooled as, by object: it is already the
            # final JSON (dumps of a loads of it gives the same bytes), so the
            # kept events are written back verbatim instead of encoded again.
            raw = {}
            if market_id in counts:
                with open(parts[market_id], encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            try:
                                ev = json.loads(line)
                            except ValueError:
                                continue
                            events.append(ev)
                            raw[id(ev)] = line if line.endswith("\n") else line + "\n"
                _rm(parts[market_id])
            kind, entry = _finish_market(market_id, market, events, raw, day=day, events_dir=events_dir,
                                         book_factory=book_factory)
            (out if kind == "out" else unusable).append(entry)
    except BaseException:
        for p in parts.values():
            _rm(p)
        raise
    return {"out": out, "unusable": unusable, "bbo_seen": bbo_seen}


def _finish_market(market_id, market, events, raw, *, day, events_dir, book_factory):
    """Order, cut and check one market's spooled events; write its event file.

    `raw` maps id(event) to the line it was spooled as. Shared by the sequential
    and the parallel spool, so the two cannot drift.
    """
    fin = finalise_market(events, market, book_factory) if market else {"events": [], "up_px": None, "down_px": None}
    why = market_unusable(market, fin["events"])
    if why:
        return "unusable", {"market_id": market_id, "asset": (market or {}).get("asset"), "day": day, "why": why}
    fd, events_file = tempfile.mkstemp(prefix=f"{day}-", suffix=".jsonl", dir=events_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for e in fin["events"]:
            fh.write(raw[id(e)])
    return "out", {
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
    }


# --- the same spool, on several processes ----------------------------------
#
# Two phases, both deterministic, and the output is the sequential spool's,
# byte for byte (client/backtest/tests: Parallel; equivalence.test.mjs).
#
#  1. STRIPES. Worker k reads every file of the day but parses only the
#     non-blank data lines whose index is k mod N (decompressing is cheap; the
#     JSON parse and the event building are what cost). Each event is written,
#     tagged with (file index, line index, position in the row) and its market,
#     to one of B bucket files by market. The book throttle is STATEFUL per
#     market and order-dependent, so it is not applied here: every row that
#     consults it leaves a marker (j = -1) with the timestamp it was consulted
#     with — even a row that then yields no event, because consulting it still
#     moves the throttle.
#  2. BUCKETS. Each bucket's markets are rebuilt from all stripes in exactly
#     the sequential order (file, line, position), the throttle is replayed on
#     the markers from the state the run's throttle already had, and each market
#     is finished by the same _finish_market the sequential spool uses. The
#     throttle state of the bucket's markets goes back to the parent, so the
#     next day starts where a sequential run would.

_PSTATE: dict = {}
# The children read _PSTATE as it was at fork time, and the pool forks lazily,
# as tasks are submitted: one run's state must stay put for its pool's whole
# life. Concurrent run() calls in one process take turns here.
_PSTATE_LOCK = threading.Lock()


def _striped_rows(root, rel, k, n):
    full = os.path.join(root, rel)
    opener = gzip.open if rel.endswith(".gz") else open
    is_csv = ".csv" in rel
    with opener(full, "rb") as raw, io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=None) as fh:
        header = None
        idx = -1
        for line in fh:
            s = js_trim(line)
            if not s:
                continue
            if is_csv and header is None:
                header = s.split(",")
                continue
            idx += 1
            if idx % n != k:
                continue
            row = parse_row(s, is_csv=is_csv, header=header, loads=json_loads)
            if row is not None:
                yield idx, row


class _ThrottleProbe:
    """Stands in for the throttle in phase 1: keeps everything, remembers who asked."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.calls = []

    def keep(self, ds, market_id, ts, asset):
        if ds == self.dataset:
            self.calls.append((market_id, ts))
        return True


def _stripe(k):
    st = _PSTATE
    n, markets, bucket = st["n"], st["markets"], st["bucket"]
    paths = [os.path.join(st["dir"], f"s{k}-b{b}.part") for b in range(st["buckets"])]
    bufs = [[] for _ in paths]
    handles = [open(p, "w", encoding="utf-8") for p in paths]
    bbo_seen = set()
    probe = _ThrottleProbe(st["throttle"].dataset if st["throttle"] is not None else None)
    try:
        for f_idx, f in enumerate(st["feed"]):
            meta = classify_path(f)
            if meta["dataset"] == "markets":
                continue
            for line_no, row in _striped_rows(st["root"], f, k, n):
                probe.calls = []
                try:
                    evs = events_from_row(f, row, markets, st["by_slug"],
                                          probe if st["throttle"] is not None else None, meta)
                except Exception as err:  # noqa: BLE001
                    if not probe.calls:
                        raise
                    # Thrown AFTER the row consulted the throttle: sequentially
                    # it is thrown only if the throttle keeps this row. Carried
                    # on the marker and decided in phase 2.
                    blob = base64.b64encode(pickle.dumps(err)).decode()
                    for mid, ts in probe.calls:
                        key = str(mid)
                        if key in markets:
                            bufs[bucket[key]].append(
                                f"{f_idx} {line_no} -1\t{key}\t{json.dumps({'ts': ts, 'err': blob})}\n")
                    continue
                for mid, ts in probe.calls:
                    key = str(mid)
                    if key in markets:
                        b = bucket[key]
                        bufs[b].append(f"{f_idx} {line_no} -1\t{key}\t{json.dumps(ts)}\n")
                for j, (mid, ev) in enumerate(evs):
                    if ev.get("bbo"):
                        bbo_seen.add(_market_day_key(markets.get(mid)))
                    key = str(mid)
                    if key not in markets:
                        continue
                    b = bucket[key]
                    bufs[b].append(f"{f_idx} {line_no} {j}\t{key}\t{_dump(ev)}\n")
                    if len(bufs[b]) >= 4096:
                        handles[b].write("".join(bufs[b]))
                        bufs[b] = []
        for b, h in enumerate(handles):
            if bufs[b]:
                h.write("".join(bufs[b]))
    finally:
        for h in handles:
            h.close()
    return bbo_seen


def _merge_bucket(b):
    st = _PSTATE
    markets, throttle = st["markets"], st["throttle"]
    recs: dict = {}
    for k in range(st["n"]):
        path = os.path.join(st["dir"], f"s{k}-b{b}.part")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                head, key, payload = line.rstrip("\n").split("\t", 2)
                f_idx, line_no, j = head.split(" ")
                recs.setdefault(key, []).append((int(f_idx), int(line_no), int(j), payload))
        _rm(path)
    results = []
    for market_id in st["bucket_markets"][b]:
        market = markets[market_id]
        items = recs.get(market_id, [])
        items.sort(key=lambda r: (r[0], r[1], r[2]))
        events, raw = [], {}
        dropped_row = None
        for f_idx, line_no, j, payload in items:
            if j == -1:
                mark = json.loads(payload)
                err = None
                if isinstance(mark, dict):
                    mark, err = mark["ts"], mark["err"]
                kept = throttle.keep(throttle.dataset, market_id, mark, market["asset"])
                if kept and err is not None:
                    raise pickle.loads(base64.b64decode(err))
                dropped_row = None if kept else (f_idx, line_no)
                continue
            if dropped_row == (f_idx, line_no):
                continue
            ev = json.loads(payload)
            events.append(ev)
            raw[id(ev)] = payload + "\n"
        results.append((market_id, _finish_market(market_id, market, events, raw, day=st["day"],
                                                  events_dir=st["events_dir"], book_factory=st["book_factory"])))
    last = {}
    if throttle is not None:
        for market_id in st["bucket_markets"][b]:
            key = js_string(market_id)
            if key in throttle._last:
                last[key] = throttle._last[key]
    return results, last


def spool_day_parallel(*, day, markets, by_slug, throttle, events_dir, feed, root, book_factory, workers, ctx):
    """spool_day, on `workers` processes. Same output, same throttle state after."""
    os.makedirs(events_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"{day}-par-", dir=events_dir)
    ids = list(markets)
    buckets = max(1, min(len(ids), 128))
    bucket = {mid: i % buckets for i, mid in enumerate(ids)}
    bucket_markets = [[mid for i, mid in enumerate(ids) if i % buckets == b] for b in range(buckets)]
    _PSTATE_LOCK.acquire()
    _PSTATE.clear()
    _PSTATE.update(n=workers, markets=markets, by_slug=by_slug, throttle=throttle, feed=feed, root=root,
                   dir=work, buckets=buckets, bucket=bucket, bucket_markets=bucket_markets, day=day,
                   events_dir=events_dir, book_factory=book_factory)
    try:
        # ProcessPoolExecutor rather than multiprocessing.Pool: a worker that is
        # killed (out of memory, say) raises BrokenProcessPool here instead of
        # leaving the run waiting forever for a result that will not come.
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            seen_sets = list(pool.map(_stripe, range(workers)))
            merged = list(pool.map(_merge_bucket, range(buckets)))
    finally:
        _PSTATE.clear()
        _PSTATE_LOCK.release()
        _shutil.rmtree(work, ignore_errors=True)
    bbo_seen = set().union(*seen_sets) if seen_sets else set()
    by_market = {}
    for results, last in merged:
        for market_id, res in results:
            by_market[market_id] = res
        if throttle is not None:
            throttle._last.update(last)
    out, unusable = [], []
    for market_id in ids:
        kind, entry = by_market[market_id]
        (out if kind == "out" else unusable).append(entry)
    return {"out": out, "unusable": unusable, "bbo_seen": bbo_seen}


def _rm(p):
    try:
        os.remove(p)
    except OSError:
        pass


def load_local_day(*, root, day, venue, assets, datasets, intervals, throttle, events_dir, book_factory,
                   workers=1, ctx=None):
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

    if workers > 1 and ctx is not None and len(markets) > 1:
        spooled = spool_day_parallel(day=day, markets=markets, by_slug=by_slug, throttle=throttle,
                                     events_dir=events_dir, feed=feed, root=root, book_factory=book_factory,
                                     workers=workers, ctx=ctx)
    else:
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
        "throttle": ("off" if throttle is None
                     else {"per_asset": dict(sorted(throttle.per_asset.items())), "fallback": throttle.fallback}),
        "files": files,
        # 2: each day decoded with its own throttle. Days cached before that were
        # thinned with the previous day's state carried in; a new key retires them.
        "format": 2,
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
