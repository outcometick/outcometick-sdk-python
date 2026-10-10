<!--
  GENERATED — do not edit this repository directly.

  Every file here is built from the outcometick monorepo by
  scripts/publish-sdk-repos.mjs and overwritten wholesale on each publish.
  An edit made here survives until the next publish and then disappears.

  Generated from monorepo revision 6be4f1349887566741e87024f36991470b47b330.
-->

# outcometick

The Python strategy SDK for [outcometick.com](https://outcometick.com) —
tick-level data for Polymarket and Predict.fun crypto Up/Down markets.

```
pip install outcometick
```

```python
from outcometick import Strategy, Order


class MeanReversion(Strategy):
    def on_market_open(self, ctx, market):
        self.entered = False

    def on_tick(self, ctx, tick):
        z = ctx.zscore(tick.value, window=180)
        if self.entered or abs(z) < ctx.p.entry_z:
            return None
        side = "DOWN" if z > 0 else "UP"
        limit = ctx.book().best(side)
        if limit is None:
            return None
        self.entered = True
        return Order(side=side, size=ctx.p.size, limit=limit)
```

## What is in here

The SDK surface your strategy imports, and nothing else:

| | |
|---|---|
| `Strategy` | the base class you subclass |
| `Order` | what a hook returns; validates side, size and limit on construction |
| `SIDES` | `("UP", "DOWN")` |

It is typed (`py.typed`), so your editor and `mypy` know the API. Everything a
strategy can actually *do* arrives through `ctx`, which the runner constructs —
there is deliberately nothing here to reach out with.

The hooks are not defined on the base class on purpose. A default no-op
`on_tick` would turn "you declared a hook you did not implement" — a rejection
fixable in seconds — into a run that quietly never trades and bills you for an
empty equity curve.

## Testing

```
pip install . && python -m unittest discover -s tests
```

## Downloading data

The other half of the package, on a separate import because it has nothing to do
with writing a strategy:

```python
from outcometick.data import DataClient, NO_VALUE

ot = DataClient()                                  # key from OT_KEY

meta = ot.meta()                                   # what can this key see?

res = ot.files(
    from_="2026-08-01", to="2026-08-12",           # or date="2026-08-12"
    asset=["BTC", "ETH"],                          # the BASE symbol, not BTCUSD
    dataset="prices",
    interval=["5m", NO_VALUE],                     # "5m" alone EXCLUDES the
)                                                  # period-less settlement streams

ot.download(res["files"][0], save_to="btc.csv.gz")  # checksum verified
```

`from_` rather than `from`, because `from` is a Python keyword; it goes on the
wire as `from`.

`meta()["intervals"]` holds real durations only — the `none` sentinel is
reported separately under `filterTokens`, so code that builds an enum from it
or parses the values as durations never meets a token.

### Rebuilding a Polymarket order book

`book` and `price_change` are stored at a capture cadence, so a removal can
fall between two stored frames and leave a stale level behind. `OrderBook`
applies the documented rebuild rule: snapshots replace a token's
ladder, changes set absolute sizes (0 removes), and levels crossed by the newest
best bid/ask (from `best_bid_ask` and from each change) are dropped. Feed it
rows from the three files merged by `recv_ms`.

```python
from outcometick.data import OrderBook

book = OrderBook()
for row in rows:
    book.apply(row)        # dicts or JSONL lines
book.ladder(token_id)      # {"bids": [{"price", "size"}], "asks": [...]}, best first
book.best(token_id)        # {"bid": ..., "ask": ...}
```

That removes every level the best prices have moved past, but it cannot
restore what the dropped frames added or resized: until the next snapshot a
level, at the top too, can be missing or carry an old size (`best_bid_ask` has
prices only). `best()` is the best level of the rebuilt ladder, not the
market's latest best bid/ask — read `best_bid_ask` rows for that.

### Smart-money trade history

A separate subscription with its own key: daily files of the trades made by the
top-ranked Polymarket traders.

```python
import os
smart = DataClient(key=os.environ["OT_SMART_KEY"])
days = smart.smart_days()["days"]                          # newest first
day = next((d["day"] for d in days if d["lists"].get("top100", {}).get("status") == "published"), None)
if day:
    smart.smart_download(day, "top100", save_to="top100.csv.zst")   # verified
```

What the lists are and what each column means:
https://outcometick.com/polymarket-smart-money-data

Standard library only: no `requests`, no dependency added to your project.

## Running a backtest

Submitting and replaying is done with the `ot` command line, which is
distributed on npm because there is exactly one of it for both languages:

```
npm i -g outcometick
ot check .          # the same validator the queue runs
ot run   .          # replay locally against sample data
ot submit . --assets btc --days 30   # send it to the queue
```

Backtests cover only the most recent 35 archived days (breaking in
2.0); an earlier range is refused with `E_SCOPE`, naming the current window.

It runs Python strategies by spawning your local `python3`. Two CLIs would mean
two copies of the validator, and the second copy is what makes
"if it passes locally it will not be rejected on submit" stop being true.

Full reference: https://outcometick.com/docs/sdk

## Links

- [outcometick.com](https://outcometick.com) — what this is, and what the data covers
- [Run a backtest](https://outcometick.com/backtest) — paste a strategy, watch it run
- [SDK reference](https://outcometick.com/docs/sdk) — manifest, hooks, `ctx`, limits
- [Data API](https://outcometick.com/docs) — the archive these strategies read
