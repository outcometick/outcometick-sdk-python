"""Replay independent markets in parallel with sequential output.

Merge trades, fills and totals in market order to preserve floating-point sums.
The log budget is run-wide. Once exhausted, Ctx.log skips message formatting,
which can have side effects. Record stored lines' byte costs without formatting
again; replay sequentially if any shard exhausts the budget or the merged costs
would cause a call after exhaustion.

Parallel replay is opt-in and requires instance-local strategy state. Changed
or uncheckable class attributes (including nested classes) or params trigger
sequential replay. Module globals and function attributes remain the caller's
responsibility. The wall-clock event budget is checked per market here rather
than across the run.

Session mode shares state and is sequential. Platforms without fork also run
sequentially, since notebook-defined strategies cannot be pickled for spawn.
"""

from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import pickle
import threading
import types

_STATE: dict = {}
# Held for a whole parallel replay: the pool forks lazily and its children read
# _STATE as it was then, so concurrent run() calls in one process take turns.
_STATE_LOCK = threading.Lock()

LOG_NOTICE = ("[runner] log budget spent -- the rest of this run's ctx.log output was dropped. "
              "ctx.log is for reading, not for exporting; see the SDK docs for the limit.")


class SharedState(Exception):
    """The parallel result cannot be the sequential one; the strategy's class or
    params changed under a shard, or the run-wide log budget ran out: replay
    sequentially."""


def fork_context():
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return None


def default_workers() -> int:
    return max(1, min(os.cpu_count() or 1, 32))


def _fingerprint(klass, params):
    """Fingerprint class and base-class data attributes and params.

    Return None if any value cannot be pickled, since changes to such state
    cannot be detected reliably.
    """
    parts = []
    classes = [c for c in getattr(klass, "__mro__", ()) if c is not object]
    seen = set()
    while classes:
        c = classes.pop(0)
        if id(c) in seen:
            continue
        seen.add(id(c))
        data = {}
        for k, v in vars(c).items():
            if k.startswith("__") and k.endswith("__"):
                continue
            if isinstance(v, type):
                # A class defined inside the strategy carries state the same way.
                if v.__qualname__.startswith(c.__qualname__ + "."):
                    classes.append(v)
                continue
            if isinstance(v, (types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.ModuleType,
                              staticmethod, classmethod, property)):
                continue
            try:
                data[k] = pickle.dumps(v, protocol=4)
            except Exception:  # noqa: BLE001
                return None
        parts.append((c.__module__, c.__qualname__, sorted(data.items())))
    try:
        parts.append(("params", pickle.dumps(params, protocol=4)))
    except Exception:  # noqa: BLE001
        return None
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def _child_init():
    """In the child only: record the cost of each line Ctx.log keeps."""
    otreplay = _STATE["engine"]["otreplay"]
    Ctx = otreplay.Ctx
    original = Ctx.log
    internals = otreplay._INTERNALS

    def log(self, msg):
        logs = internals[id(self)]["logs"]
        before = len(logs)
        original(self, msg)
        if len(logs) > before:
            costs = _STATE.get("costs")
            if costs is not None:
                costs.append(len(logs[-1].encode("utf-8")) + 1)

    Ctx.log = log


def _replay_one(index: int):
    st = _STATE
    m = st["markets"][index]
    lines = st["lines_of"](m)
    stream = iter([st["dumps"]({**st["header_of"](m), "n": len(lines)}), *lines])
    out: list = []
    costs: list = []
    st["costs"] = costs
    code = st["run_job"](st["job"], st["load_class"], lambda: next(stream, None),
                         lambda channel, payload: out.append((channel, payload)))
    changed = _fingerprint(st["get_class"](), st["job"].get("params")) != st["fingerprint"]
    return code, out, costs, changed


def replay_markets(*, workers, markets, job, load_class, get_class, lines_of, header_of, run_job, dumps,
                   engine, channels):
    """Run every market through the harness on `workers` processes.

    Returns (code, out_lines) shaped exactly like one sequential run_job call,
    or raises SharedState when the run has to be replayed sequentially.
    """
    fingerprint = _fingerprint(get_class(), job.get("params"))
    if fingerprint is None:
        raise SharedState()
    out_lines: list = []
    _STATE_LOCK.acquire()
    _STATE.clear()
    _STATE.update(markets=markets, job=job, load_class=load_class, get_class=get_class, lines_of=lines_of,
                  header_of=header_of, run_job=run_job, dumps=dumps, engine=engine, fingerprint=fingerprint)
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=fork_context(),
                                                    initializer=_child_init) as pool:
            pending: collections.deque = collections.deque()
            try:
                total, final_code = _drain(pool, pending, len(markets), 2 * workers, job, channels, out_lines)
            finally:
                # Anything that ends the merge early cancels what has not started.
                for f in pending:
                    f.cancel()
    finally:
        _STATE.clear()
        _STATE_LOCK.release()
    total = total or {}
    total["log_truncated"] = False
    out_lines.append((channels["result"], json.dumps(total, separators=(",", ":"), ensure_ascii=False)))
    return final_code, out_lines


def _drain(pool, pending, n, window, job, channels, out_lines):
    """Merge shard results in market order, with at most `window` of them in flight."""
    ch_log, ch_result, ch_progress = channels["log"], channels["result"], channels["progress"]
    ch_record = channels.get("record")
    record_limit = _record_limits()[0]
    record_kept = 0
    budget_bytes = (job.get("limits") or {}).get("logBytesPerRun")
    submitted = 0
    spent = 0
    total = None
    while submitted < n and len(pending) < window:
        pending.append(pool.submit(_replay_one, submitted))
        submitted += 1
    while pending:
        code, out, costs, changed = pending.popleft().result()
        if submitted < n:
            pending.append(pool.submit(_replay_one, submitted))
            submitted += 1
        if changed:
            raise SharedState()
        result = None
        logs = []
        for channel, payload in out:
            if channel == ch_result:
                result = json.loads(payload)
            elif channel == ch_log:
                if payload == LOG_NOTICE:
                    raise SharedState()
                logs.append(payload)
            elif channel == ch_progress:
                continue
            elif channel == ch_record and record_kept >= record_limit:
                # Past the run's row budget the record is cut anyway (_add_result
                # marks it): not kept, so memory stays bounded across shards.
                continue
            else:
                if channel == ch_record:
                    record_kept += 1
                out_lines.append((channel, payload))
        for cost in costs:
            if budget_bytes is not None and spent >= budget_bytes:
                # The sequential run would have returned from this call
                # without formatting the message.
                raise SharedState()
            spent += cost
        out_lines.extend((ch_log, text) for text in logs)
        total = _add_result(total, result or {})
        if code != 0:
            return total, code
    return total, 0


def _record_limits():
    h = _STATE.get("engine", {}).get("otharness") if _STATE else None
    if h is None:
        from ._engine import load
        h = load()["otharness"]
    return h.MAX_RECORD_ROWS, h.MAX_RECORD_BYTES


def _add_result(total, r):
    if total is None:
        total = {"markets_run": 0, "events_seen": 0, "fees_paid": 0, "log_truncated": False, "budget": None,
                 "market_summaries": [], "crosschecks": [], "rejection": None, "maker": None}
    total["markets_run"] += r.get("markets_run") or 0
    total["events_seen"] += r.get("events_seen") or 0
    total["fees_paid"] += r.get("fees_paid") or 0
    total["market_summaries"].extend(r.get("market_summaries") or [])
    total["crosschecks"].extend(r.get("crosschecks") or [])
    if r.get("rejection") and not total["rejection"]:
        total["rejection"] = r["rejection"]
    # Execution options' counters (job.engine): absent unless the run enabled one.
    o = r.get("orders")
    if isinstance(o, dict):
        t = total.setdefault("orders", {"rejected": {}, "rebates": 0})
        for k, v in (o.get("rejected") or {}).items():
            t["rejected"][k] = t["rejected"].get(k, 0) + (v or 0)
        t["rebates"] += o.get("rebates") or 0
    # The recorder's budgets: summed in market order, and over either budget the
    # sequential run would have stopped recording; so this one says so too.
    rc = r.get("record")
    if isinstance(rc, dict):
        t = total.setdefault("record", {"rows": 0, "bytes": 0, "truncated": False})
        t["rows"] += rc.get("rows") or 0
        t["bytes"] += rc.get("bytes") or 0
        t["truncated"] = bool(t["truncated"] or rc.get("truncated")
                              or t["rows"] > _record_limits()[0] or t["bytes"] > _record_limits()[1])
    # Spot counters (the spot dataset declared): absent otherwise, like the harness.
    sp = r.get("spot")
    if isinstance(sp, dict):
        t = total.setdefault("spot", {"markets": 0, "markets_without": 0})
        for k in t:
            t[k] += sp.get(k) or 0
    # book_age counters: integers, so the sum does not depend on order; absent
    # when no taker fill took archive depth, like the harness.
    ba = r.get("book_age")
    if isinstance(ba, dict):
        t = total.setdefault("book_age", {"fills": 0, "size_micro": 0, "by_age_micro": [0, 0, 0, 0], "max_age_ms": 0})
        t["fills"] += ba.get("fills") or 0
        t["size_micro"] += ba.get("size_micro") or 0
        t["by_age_micro"] = [x + (y or 0) for x, y in zip(t["by_age_micro"], ba.get("by_age_micro") or [0, 0, 0, 0])]
        t["max_age_ms"] = max(t["max_age_ms"], ba.get("max_age_ms") or 0)
    # Timer counters (on_timer declared): absent otherwise, like the harness.
    tm = r.get("timers")
    if isinstance(tm, dict):
        t = total.setdefault("timers", {"created": 0, "fired": 0, "dropped": 0})
        for k in t:
            t[k] += tm.get(k) or 0
    if r.get("maker"):
        if total["maker"] is None:
            total["maker"] = {k: 0 for k in r["maker"]}
        for k, v in r["maker"].items():
            total["maker"][k] = total["maker"].get(k, 0) + (v or 0)
    b = r.get("budget")
    if b:
        if total["budget"] is None:
            total["budget"] = dict(b)
        else:
            t = total["budget"]
            events = t.get("events", 0) + b.get("events", 0)
            t["avg_micros"] = ((t.get("avg_micros", 0) * t.get("events", 0)) + b.get("avg_micros", 0) * b.get("events", 0)) / events if events else 0
            t["events"] = events
            t["breaches"] = t.get("breaches", 0) + b.get("breaches", 0)
            t["breach_rate"] = t["breaches"] / events if events else 0
            t["max_micros"] = max(t.get("max_micros", 0), b.get("max_micros", 0))
            t["window_micros"] = max(t.get("window_micros", 0), b.get("window_micros", 0))
    return total
