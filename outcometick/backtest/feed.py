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


def _dumps(obj) -> str:
    """JSON.stringify for the header objects this module writes."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def market_lines(markets, *, lines_of, header_of):
    """market mode: a header with the line count, then the lines."""
    for m in markets:
        lines = lines_of(m)
        yield _dumps({**header_of(m), "n": len(lines)})
        yield from lines


def session_lines(markets, *, lines_of, header_of):
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
        heapq.heappush(heap, (t, phase, c["k"], id(c), c))

    nxt = 0
    while True:
        top = heap[0] if heap else None
        if nxt < len(markets) and (top is None or open_at(markets[nxt]) <= top[0]):
            k = nxt
            m = markets[nxt]
            nxt += 1
            header = header_of(m)
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
        _, phase, k, _, c = heapq.heappop(heap)
        if phase == 2:
            yield '{"close":%d}' % k
            continue
        yield '{"i":%d,"e":%s}' % (k, c["lines"][c["i"]])
        c["i"] += 1
        push(c)
