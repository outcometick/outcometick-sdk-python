"""The job stream, exactly as the worker writes it.

market mode: a header per market, then its events (runner/worker.mjs,
cli/commands/run.mjs). session mode: one time order across every market -- a
port of runner/session-feed.mjs, whose comments explain why. The shared loop in
otharness.py consumes these lines whether they came over a pipe from Node or
from this generator, so a pure-Python run and a queued run are framed the same.
"""

from __future__ import annotations

import heapq
import json
import math
import re

_TS = re.compile(r'"ts_ms"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', re.ASCII)


def _line_time(line: str, lags) -> float:
    m = _TS.search(line)
    ts = float(m.group(1)) if m else math.nan
    if line.startswith('{"kind":"ext"') or line.startswith('{"kind":"ref"'):
        try:
            row = json.loads(line)
            lag = (lags or {}).get(row.get("name"))
            lag = lag if isinstance(lag, (int, float)) and not isinstance(lag, bool) and math.isfinite(lag) else 0
            ts = float(row.get("ts_ms")) + lag
        except (ValueError, TypeError):
            pass
    return ts


_SEQ = re.compile(r'"seq"\s*:\s*(-?\d+)', re.ASCII)


def _line_seq(line: str) -> int:
    m = _SEQ.search(line)
    return int(m.group(1)) if m else 0


def _dumps(obj) -> str:
    """JSON.stringify for the header objects this module writes."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def market_lines(markets, *, lines_of, header_of, cursor_of=None):
    """market mode: a header with the line count, then the lines.

    A generic market (generic.py) is streamed through `cursor_of`, its count
    known from decoding."""
    for m in markets:
        if (m.get("market") or {}).get("kind") == "generic":
            yield _dumps({**header_of(m), "n": m["n"]})
            yield from cursor_of(m)
            continue
        lines = lines_of(m)
        yield _dumps({**header_of(m), "n": len(lines)})
        yield from lines


def session_lines(markets, *, lines_of, header_of, cursor_of=None):
    """session mode: every market merged into one time order.

    Ties at one millisecond: an opening first, then events in market order,
    then closings -- MUST MATCH sessionLines in runner/session-feed.mjs.
    """
    heap: list = []

    def open_at(m):
        v = (m.get("market") or {}).get("open_ts_ms")
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            return v
        return -math.inf

    def push(c):
        if c["i"] < len(c["lines"]):
            t, phase = c["times"][c["i"]], 1
        else:
            t, phase = c["close_at"], 2
        heapq.heappush(heap, (t, phase, 0, c["k"], id(c), c))

    def push_generic(c):
        """A generic market: streamed, `end` (phase 2) then `close` (phase 3);
        events tie-break on the global input sequence. MUST MATCH
        advanceGeneric in runner/session-feed.mjs."""
        line = next(c["it"], None)
        if line is not None:
            t = _line_time(line, None)
            if math.isfinite(t) and t > c["run"]:
                c["run"] = t
            c["line"] = line
            c["t"], phase, s = c["run"], 1, _line_seq(line)
        elif not c["end_sent"]:
            c["line"] = None
            c["t"], phase, s = max(c["end_at"], c["run"]), 2, 0
        else:
            c["line"] = None
            c["t"], phase, s = max(c["close_at"], c["t"]), 3, 0
        heapq.heappush(heap, (c["t"], phase, s, c["k"], id(c), c))

    nxt = 0
    while True:
        top = heap[0] if heap else None
        if nxt < len(markets) and (top is None or open_at(markets[nxt]) <= top[0]):
            k = nxt
            m = markets[nxt]
            nxt += 1
            header = header_of(m)
            if (m.get("market") or {}).get("kind") == "generic":
                if cursor_of is None:
                    raise ValueError("a generic market needs a streaming cursor (cursor_of)")
                yield _dumps({"open": {"i": k, **header}})
                mk = m["market"]
                push_generic({"k": k, "generic": True, "it": iter(cursor_of(m)), "run": open_at(m),
                              "end_at": mk["trade_until_ts_ms"], "close_at": mk["close_ts_ms"],
                              "end_sent": False, "line": None, "t": 0})
                continue
            lines = lines_of(m)
            lags = header.get("lags")
            times = []
            run = open_at(m)
            for line in lines:
                t = _line_time(line, lags)
                if math.isfinite(t) and t > run:
                    run = t
                times.append(run)
            declared = (m.get("market") or {}).get("close_ts_ms")
            if isinstance(declared, (int, float)) and not isinstance(declared, bool) and math.isfinite(declared):
                close_at = max(declared, run)
            else:
                close_at = run
            yield _dumps({"open": {"i": k, **header}})
            push({"k": k, "lines": lines, "times": times, "i": 0, "close_at": close_at})
            continue
        if top is None:
            break
        _, phase, _, k, _, c = heapq.heappop(heap)
        if c.get("generic"):
            if phase == 1:
                yield '{"i":%d,"e":%s}' % (k, c["line"])
            elif phase == 2:
                yield '{"end":%d}' % k
                c["end_sent"] = True
            else:
                yield '{"close":%d}' % k
                continue
            push_generic(c)
            continue
        if phase == 2:
            yield '{"close":%d}' % k
            continue
        yield '{"i":%d,"e":%s}' % (k, c["lines"][c["i"]])
        c["i"] += 1
        push(c)
