<!--
  GENERATED — do not edit this repository directly.

  Every file here is built from the outcometick monorepo by
  scripts/publish-sdk-repos.mjs and overwritten wholesale on each publish.
  An edit made here survives until the next publish and then disappears.

  Generated from monorepo revision 6ac78cb3be9732070cce236ed975dc53bc3a01b7.
-->

# outcometick

The Python strategy SDK for [outcometick.com](https://outcometick.com).

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
        book = ctx.book()
        self.entered = True
        return Order(side=side, size=ctx.p.size, limit=book.best(side))
```

This package is the SDK surface: the `Strategy` base class and the `Order`
value object, so your editor and type checker know the API and your own tests
can import it.

**The `ot` command line is distributed via npm**, not here:

```
npm i -g outcometick
```

It runs Python strategies by spawning your local `python3`. Shipping one CLI
rather than two is deliberate — `ot check` must be the same validator the queue
runs, and a second implementation in another language would be the first thing
to drift.

Full reference: https://outcometick.com/docs/sdk
