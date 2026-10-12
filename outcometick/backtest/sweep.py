"""Grid search with optional chronological holdout validation.

Rank cells on earlier days, then replay the top cells on held-out later days.
Each cell is a run() using cached decoded days.

cell_workers runs cells in forked processes. The first cell of each pass runs
locally to warm the cache. If a cell fails, dies or changes class state
(including nested classes) or params, resume sequentially from that cell and
kill remaining children. Module globals are outside run()'s state contract.
Do not fork while other threads are running.
"""

from __future__ import annotations

import importlib
import itertools
import math
import os
import pickle
import shutil
import signal
import tempfile
import threading

from ._parallel import _fingerprint, fork_context

MAX_CELLS = 1000

# Metrics where a smaller number is the better one. (max_drawdown and fees are
# reported as negative numbers, so for them, as for PnL, larger is better.)
_SMALLER_IS_BETTER = frozenset(("max_drawdown_abs", "brier_score", "worst_losing_run"))


class SweepResult:
    """Every cell's metrics, ranked, with the held-out replay of the top cells."""

    def __init__(self, *, keys, cells, metric, maximize, in_days, out_days):
        self.keys = keys
        self.cells = cells
        self.metric = metric
        self.maximize = maximize
        self.in_days = in_days
        self.out_days = out_days

    @property
    def ranked(self) -> list:
        """Cells best first on the in-sample metric; a cell without a value sorts last."""
        def score(c):
            v = c["metrics"].get(self.metric)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                return math.inf
            return -v if self.maximize else v
        return sorted(self.cells, key=score)

    @property
    def best(self) -> dict:
        return self.ranked[0]

    def table(self):
        """One row per cell: params, the in-sample metrics, and held-out ones where run (needs pandas)."""
        import pandas as pd
        rows = []
        for c in self.cells:
            row = dict(c["params"])
            row.update({f"in:{k}": v for k, v in c["metrics"].items()})
            if c.get("holdout") is not None:
                row.update({f"out:{k}": v for k, v in c["holdout"].items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def heatmap(self, x, y, ax=None):
        """The in-sample metric over two params (needs matplotlib)."""
        import matplotlib.pyplot as plt
        xs = sorted({c["params"][x] for c in self.cells})
        ys = sorted({c["params"][y] for c in self.cells})
        grid = [[None] * len(xs) for _ in ys]
        for c in self.cells:
            grid[ys.index(c["params"][y])][xs.index(c["params"][x])] = c["metrics"].get(self.metric)
        ax = ax or plt.subplots(figsize=(1.2 * len(xs) + 2, 0.8 * len(ys) + 1.5))[1]
        data = [[float("nan") if v is None else v for v in row] for row in grid]
        im = ax.imshow(data, cmap="RdYlGn" if self.maximize else "RdYlGn_r", aspect="auto", origin="lower")
        ax.set_xticks(range(len(xs)), [str(v) for v in xs])
        ax.set_yticks(range(len(ys)), [str(v) for v in ys])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{self.metric}, in sample ({self.in_days[0]}..{self.in_days[-1]})")
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v is not None:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        ax.figure.colorbar(im, ax=ax)
        return ax

    def summary(self) -> str:
        lines = [f"  {len(self.cells)} cells · ranked by {self.metric} on {self.in_days[0]}..{self.in_days[-1]}"
                 + (f" · held out {self.out_days[0]}..{self.out_days[-1]}" if self.out_days else "")]
        for i, c in enumerate(self.ranked[:5]):
            p = ", ".join(f"{k}={v}" for k, v in c["params"].items())
            line = f"  #{i + 1}  {p:<40} in {c['metrics'].get(self.metric)}"
            if c.get("holdout") is not None:
                line += f"   out {c['holdout'].get(self.metric)}"
            lines.append(line)
        if self.out_days and any(c.get("holdout") is not None for c in self.cells):
            lines.append("  An edge that does not survive the held-out days was fitted to the in-sample ones.")
        elif not self.out_days:
            lines.append("  No days were held out: these are in-sample numbers. Pass holdout= to check them.")
        return "\n".join(lines)

    def __repr__(self):
        b = self.best
        return f"<SweepResult cells={len(self.cells)} best={b['params']} {self.metric}={b['metrics'].get(self.metric)}>"


def _cells(run_cell, params_list, cell_workers, state):
    """[run_cell(p) for p in params_list], the same list whether or not it ran in parallel.

    `state()` fingerprints what one cell can leave behind for the next (None:
    cannot tell, so no forking).
    """
    if not params_list:
        return []
    done = [run_cell(params_list[0])]
    rest = params_list[1:]
    ctx = fork_context()
    # fork copies the calling thread only: a lock another thread held at that
    # moment (the engine's, a logging handler's) would stay locked in the child.
    # os.waitid: on macOS from Python 3.13 only. signal.signal: main thread only.
    before = (state() if rest and cell_workers and cell_workers > 1 and ctx is not None
              and hasattr(os, "waitid") and threading.current_thread() is threading.main_thread()
              and threading.active_count() == 1 else None)
    if before is None:
        return done + [run_cell(p) for p in rest]
    accepted = _forked(run_cell, rest, cell_workers, state, before)
    # Resume sequentially at the first cell not accepted from parallel replay.
    return done + accepted + [run_cell(p) for p in rest[len(accepted):]]


def _forked(run_cell, params, workers, state, before):
    """Return the longest prefix of cells safe to accept from parallel replay.
    Stop at the first failed, dead or state-mutating cell. Run up to workers
    children, each in its own process group. Publish answers by rename and read
    them only after the child exits and its process group is gone. Kill all
    remaining groups on return or interruption.

    Use os.fork so reaping stays under this function's control. Kill the group
    before reaping the child to prevent process-group id reuse.
    """
    out_dir = tempfile.mkdtemp(prefix="ot-sweep-")
    answer = lambda i: os.path.join(out_dir, f"{i}.pkl")  # noqa: E731

    def fork(i):
        # Defer Ctrl-C until the child is tracked for cleanup. Use a handler;
        # another OS thread could receive SIGINT despite a signal mask.
        noted = []
        previous = signal.signal(signal.SIGINT, lambda *a: noted.append(a))
        try:
            pid = os.fork()
            if pid == 0:
                code = 1
                try:
                    signal.signal(signal.SIGINT, previous)
                    os.setpgid(0, 0)
                    payload = pickle.dumps((run_cell(params[i]), state() != before), protocol=4)
                    with open(answer(i) + ".tmp", "wb") as fh:
                        fh.write(payload)
                    os.replace(answer(i) + ".tmp", answer(i))
                    code = 0
                except BaseException:  # noqa: BLE001; no answer: the parent reruns it and raises as it would have
                    pass
                finally:
                    os._exit(code)
            try:
                os.setpgid(pid, pid)
            except OSError:
                pass  # it died already
            running[i] = pid
        finally:
            signal.signal(signal.SIGINT, previous)
            if noted:
                if callable(previous):
                    previous(*noted[0])  # the default one raises KeyboardInterrupt
                elif previous == signal.SIG_DFL:
                    os.kill(os.getpid(), signal.SIGINT)

    running = {}  # cell index -> pid
    accepted = []
    launched = 0
    try:
        while len(accepted) < len(params):
            while launched < len(params) and len(running) < workers:
                try:
                    fork(launched)
                except OSError:  # cannot fork now: what is running still counts
                    break
                launched += 1
            i = len(accepted)
            if i not in running:
                break
            # Keep the exited child tracked until reaped so interrupts still clean it up.
            os.waitid(os.P_PID, running[i], os.WEXITED | os.WNOWAIT)
            _kill_group(running[i])
            os.waitpid(running.pop(i), 0)
            try:
                with open(answer(i), "rb") as fh:
                    metrics, changed = pickle.load(fh)
            except FileNotFoundError:  # failed, or killed (out of memory, say)
                break
            if changed:
                break
            accepted.append(metrics)
    finally:
        # None of these has been reaped, so each group id is still theirs.
        for pid in running.values():
            _kill_group(pid)
        for pid in running.values():
            os.waitpid(pid, 0)
        shutil.rmtree(out_dir, ignore_errors=True)
    return accepted


def _kill_group(pid):
    # The parent made the group right after fork, so while the cell lives it
    # exists; gone means the cell and everything it started are gone.
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def sweep(strategy, data, grid, *, metric="net_pnl", maximize=None, holdout=0.0, top=3, days=None, params=None,
          cell_workers=None, **run_kwargs) -> SweepResult:
    """Run `strategy` once per combination of `grid` values and rank the cells.

    grid      {param: [values, ...]}, layered over `params` (or the class's own).
    metric    a key of report["metrics"]; maximize defaults to True except for
              drawdown-, loss- and error-type metrics.
    holdout   fraction of the days (the latest ones) kept out of the ranking;
              the `top` cells are then replayed on them. 0 = rank on everything.
    cell_workers  None (default): one cell after another. A number: that many
              cells at once on forked processes; the same result (see the
              module docstring; needs fork, else sequential).
    Everything else (venue, assets, datasets, latency_ms, workers, ...) is
    passed to run() for every cell.
    """
    # Import late to avoid a cycle with __init__, which defines run and BacktestError.
    pkg = importlib.import_module(__package__)
    BacktestError, run = pkg.BacktestError, pkg.run
    from .local import local_days

    if not isinstance(grid, dict) or not grid:
        raise BacktestError("E_MANIFEST", "grid must be a non-empty {param: [values]} dict")
    keys = list(grid)
    values = []
    for k in keys:
        v = grid[k]
        if isinstance(v, (str, bytes)) or not hasattr(v, "__iter__"):
            raise BacktestError("E_MANIFEST", f"grid[{k!r}] must be a list of values")
        v = list(v)
        if not v:
            raise BacktestError("E_MANIFEST", f"grid[{k!r}] is empty")
        values.append(v)
    n_cells = math.prod(len(v) for v in values)
    if n_cells > MAX_CELLS:
        raise BacktestError("E_LIMIT", f"{n_cells} cells is more than {MAX_CELLS}; narrow the grid")
    if not (isinstance(holdout, (int, float)) and not isinstance(holdout, bool) and 0 <= holdout < 1):
        raise BacktestError("E_MANIFEST", "holdout is a fraction of the days, from 0 up to (not including) 1")

    all_days = sorted(days) if days else local_days(str(data))
    if not all_days:
        raise BacktestError("E_DATA", "no dated files in that archive")
    n_out = 0 if holdout == 0 else max(1, round(len(all_days) * holdout))
    if n_out >= len(all_days):
        raise BacktestError("E_SCOPE", f"holdout={holdout} leaves no day to rank on among {len(all_days)} day(s); "
                                       "hold out a smaller fraction, or add days")
    in_days, out_days = all_days[: len(all_days) - n_out], all_days[len(all_days) - n_out:]
    base = dict(params if params is not None else (getattr(strategy, "params", None) or {}))
    smaller = metric in _SMALLER_IS_BETTER
    maximize = (not smaller) if maximize is None else bool(maximize)

    if cell_workers is not None and (not isinstance(cell_workers, int) or isinstance(cell_workers, bool)
                                     or cell_workers < 1):
        raise BacktestError("E_MANIFEST", f"cell_workers must be a whole number from 1, got {cell_workers!r}")

    def state():
        # The grid's values too: a cell's params share them, and the held-out
        # replay of a cell reuses them.
        return _fingerprint(strategy, (base, values))

    def ranked_cell(cell):
        r = run(strategy, data, days=in_days, params={**base, **cell}, **run_kwargs)
        if metric not in r.metrics:
            raise BacktestError("E_MANIFEST", f"unknown metric {metric!r}; known: {', '.join(sorted(r.metrics))}")
        return r.metrics

    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    cells = [{"params": cell, "metrics": m, "holdout": None}
             for cell, m in zip(combos, _cells(ranked_cell, combos, cell_workers, state))]

    result = SweepResult(keys=keys, cells=cells, metric=metric, maximize=maximize, in_days=in_days, out_days=out_days)
    if out_days:
        best = result.ranked[: max(0, int(top))]
        held = _cells(lambda cell: run(strategy, data, days=out_days, params={**base, **cell}, **run_kwargs).metrics,
                      [c["params"] for c in best], cell_workers, state)
        for c, m in zip(best, held):
            c["holdout"] = m
    return result
