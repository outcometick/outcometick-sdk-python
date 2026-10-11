"""Parameter sweeps — and the out-of-sample check that keeps them honest.

A grid search over one archive always finds a winner; whether it found an edge
or the noise in those particular days is a different question, and the only
answer is days the search never saw. So `sweep` can hold days back: the grid is
ranked on the earlier days, and the top cells are then replayed on the held-out
later ones and reported side by side. A cell that tops the in-sample table and
loses money out of sample is a fitted coincidence, and the table says so.

Every cell is an ordinary `run()` over the same decoded days (decoded once, then
served from the cache), so a cell's numbers are exactly what a single run with
those params reports.
"""

from __future__ import annotations

import importlib
import itertools
import math

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


def sweep(strategy, data, grid, *, metric="net_pnl", maximize=None, holdout=0.0, top=3, days=None, params=None,
          **run_kwargs) -> SweepResult:
    """Run `strategy` once per combination of `grid` values and rank the cells.

    grid      {param: [values, ...]}, layered over `params` (or the class's own).
    metric    a key of report["metrics"]; maximize defaults to True except for
              drawdown-, loss- and error-type metrics.
    holdout   fraction of the days (the LATEST ones) kept out of the ranking;
              the `top` cells are then replayed on them. 0 = rank on everything.
    Everything else (venue, assets, datasets, latency_ms, workers, ...) is
    passed to run() for every cell.
    """
    # The package itself, imported late: sweep.py is imported BY the package's
    # __init__, which defines run() and BacktestError.
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

    cells = []
    for combo in itertools.product(*values):
        cell = dict(zip(keys, combo))
        r = run(strategy, data, days=in_days, params={**base, **cell}, **run_kwargs)
        if metric not in r.metrics:
            raise BacktestError("E_MANIFEST", f"unknown metric {metric!r}; known: {', '.join(sorted(r.metrics))}")
        cells.append({"params": cell, "metrics": r.metrics, "holdout": None})

    result = SweepResult(keys=keys, cells=cells, metric=metric, maximize=maximize, in_days=in_days, out_days=out_days)
    if out_days:
        for c in result.ranked[: max(0, int(top))]:
            r = run(strategy, data, days=out_days, params={**base, **c["params"]}, **run_kwargs)
            c["holdout"] = r.metrics
    return result
