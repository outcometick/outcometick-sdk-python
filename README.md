<!--
  GENERATED — do not edit this repository directly.

  Every file here is built from the outcometick monorepo by
  scripts/publish-sdk-repos.mjs and overwritten wholesale on each publish.
  An edit made here survives until the next publish and then disappears.

  Generated from monorepo revision 0ad1fd0cb6d49a6ceca5b70175f0ebeeaa207b6d.
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

## Running a backtest

Submitting and replaying is done with the `ot` command line, which is
distributed on npm because there is exactly one of it for both languages:

```
npm i -g outcometick
ot check .          # the same validator the queue runs
ot run   .          # replay locally against sample data
ot submit .         # send it to the queue
```

It runs Python strategies by spawning your local `python3`. Two CLIs would mean
two copies of the validator, and the second copy is what makes
"if it passes locally it will not be rejected on submit" stop being true.

Full reference: https://outcometick.com/docs/sdk
