"""Replay markets in parallel, with output identical to a sequential run.

In market mode every market is replayed with a fresh strategy instance and a
fresh portfolio, so markets are independent and can run on separate cores. What
is NOT per market, and has to be put back together exactly:

  - rows: trades and fills are emitted per market, in market order — the shards
    are merged in that order;
  - totals: fees_paid, events_seen, the maker counters are sums over markets in
    market order — summed the same way here, so the floats agree to the bit;
  - the log budget: ONE allowance for the whole run, spent in market order.
    Once it is spent, Ctx.log returns WITHOUT formatting the message — so a
    message whose __str__ has side effects behaves differently from then on,
    and no merge can undo that. So the parallel result is only used when the
    budget was never spent: each shard records the exact byte cost of every
    line it keeps (read off the line Ctx.log stored, never by formatting the
    message again), the parent adds them up in market order, and the moment
    the sequential run would have had a call with the allowance gone — or a
    shard ran out on its own — the whole run is replayed sequentially.

Not per market, and NOT reproducible on separate processes: state a strategy
keeps outside its instance, which a sequential run carries from market to
market. Market mode's contract (and the hosted runner's sharding) rules that
out, and this path is only taken when the caller asks for parallel replay. As a
safety net each shard checks the strategy class, the classes defined inside it
and params, and the parent replays the whole run sequentially if any changed or
cannot be checked; state elsewhere (module globals, function attributes) is the
caller's contract to keep.

The per-event budget is the one thing that is not reproduced: it is a wall-clock
measurement, checked here per market rather than over the whole run.

Session mode shares one instance and one portfolio across markets and is never
parallelised. Neither is a platform without `fork` (a strategy defined in a
notebook cell cannot be pickled to a spawned process).
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
    """The parallel result cannot be the sequential one — the strategy's class or
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
    """What a sequential run would carry from one market to the next outside
    the instance: the class's (and its bases') data attributes, and params.

    None when any of it cannot be pickled: a change inside such an object
    cannot be seen, so the run must not be split.
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
    # sequential run would have stopped recording -- so this one says so too.
    rc = r.get("record")
    if isinstance(rc, dict):
        t = total.setdefault("record", {"rows": 0, "bytes": 0, "truncated": False})
        t["rows"] += rc.get("rows") or 0
        t["bytes"] += rc.get("bytes") or 0
        t["truncated"] = bool(t["truncated"] or rc.get("truncated")
                              or t["rows"] > _record_limits()[0] or t["bytes"] > _record_limits()[1])
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
