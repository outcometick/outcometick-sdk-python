"""Check public SDK exports and order validation at construction time."""

import unittest

from outcometick import Order, Strategy, SIDES


class TestSurface(unittest.TestCase):
    def test_exports_are_exactly_what_is_documented(self):
        import outcometick

        self.assertEqual(sorted(outcometick.__all__), ["Order", "SIDES", "Strategy"])
        # __all__ excludes the future-import name from wildcard imports.
        self.assertNotIn("annotations", outcometick.__all__)

    def test_sides_are_the_two_outcome_tokens(self):
        self.assertEqual(tuple(SIDES), ("UP", "DOWN"))


class TestOrder(unittest.TestCase):
    def test_accepts_a_well_formed_order(self):
        o = Order(side="UP", size=100, limit=0.55)
        self.assertEqual(o.side, "UP")
        self.assertEqual(o.size, 100.0)
        self.assertEqual(o.limit, 0.55)
        self.assertFalse(o.reduce_only)

    def test_limit_is_optional(self):
        self.assertIsNone(Order(side="DOWN", size=1).limit)

    def test_rejects_a_side_that_is_not_a_name(self):
        for bad in (None, "", "  ", 1, ["UP"]):
            with self.assertRaises(ValueError):
                Order(side=bad, size=100)

    def test_accepts_an_outcome_label(self):
        # The runner validates labels against the market's outcomes.
        self.assertEqual(Order(side="Yes", size=1).side, "Yes")

    def test_rejects_a_non_positive_size(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                Order(side="UP", size=bad)

    def test_rejects_a_limit_outside_zero_to_one(self):
        # Binary token prices are bounded by 0 and 1; reject rather than clamp.
        for bad in (-0.01, 1.5, 100):
            with self.assertRaises(ValueError):
                Order(side="UP", size=1, limit=bad)


class TestStrategy(unittest.TestCase):
    def test_params_default_to_empty_so_a_hook_can_read_them(self):
        self.assertEqual(Strategy().p, {})

    def test_hooks_are_not_defined_by_the_base_class(self):
        # Missing declared hooks must be reported, not hidden by base-class no-ops.
        for hook in ("on_tick", "on_market_open", "on_book", "on_trade", "on_settle"):
            self.assertFalse(hasattr(Strategy, hook), hook)


if __name__ == "__main__":
    unittest.main()


class TestStrictNumbers(unittest.TestCase):
    """Accept numeric values without coercion.

    Python and JS differ on coercion: float([0.5]) raises, while
    Number([0.5]) returns 0.5. Apply the same validation in both SDKs.
    """

    def test_a_limit_must_be_a_number(self):
        for bad in ['0.5', [0.5], True, float('nan'), float('inf')]:
            with self.assertRaises(ValueError, msg=f'{bad!r} was accepted as a limit'):
                Order(side='UP', size=10, limit=bad)

    def test_a_size_must_be_a_number(self):
        for bad in ['10', [10], True, float('inf'), float('nan'), 0, -1]:
            with self.assertRaises(ValueError, msg=f'{bad!r} was accepted as a size'):
                Order(side='UP', size=bad)

    def test_a_notional_must_be_a_number_with_a_limit(self):
        for bad in ['80', [80], True, float('inf'), float('nan'), 0, -1]:
            with self.assertRaises(ValueError, msg=f'{bad!r} was accepted as a notional'):
                Order(side='UP', notional=bad, limit=0.5)
        # Notional sizing requires a price ceiling.
        with self.assertRaises(ValueError):
            Order(side='UP', notional=80)

    def test_the_legal_values_still_work(self):
        # Strict validation must still accept legal numeric values.
        for limit in [0, 0.5, 1]:
            self.assertEqual(Order(side='UP', size=10, limit=limit).limit, float(limit))
        self.assertEqual(Order(side='UP', size=2.5).size, 2.5)
        # floor(80 / 0.64) is 125; Python floor division gives 124.
        self.assertEqual(Order(side='UP', notional=80, limit=0.64).size, 125)

    def test_a_resting_order_needs_a_limit_and_takes_its_own_fields(self):
        # Keep in sync with sdk-order.test.mjs: gtc requires a resting price.
        o = Order(side='UP', size=10, limit=0.4, tif='gtc', post_only=True, client_id='q-1')
        self.assertEqual((o.tif, o.post_only, o.client_id), ('gtc', True, 'q-1'))
        self.assertEqual((Order(side='UP', size=10).post_only, Order(side='UP', size=10).client_id), (False, None))
        bad = [
            dict(size=10, tif='gtc'),                                  # no price to rest at
            dict(size=10, limit=0.4, tif='gtc', hold_s=5),             # no single fill to time from
            dict(size=10, limit=0.4, post_only=True),                  # gtc-only field on an ioc
            dict(size=10, limit=0.4, client_id='a'),
            dict(size=10, limit=0.4, tif='gtc', client_id='o7'),       # looks like an engine id
            dict(size=10, limit=0.4, tif='gtc', client_id='x' * 65),
            dict(size=10, limit=0.4, tif='gtc', client_id='bad id'),
            dict(size=10, limit=0.4, tif='day'),
        ]
        for kw in bad:
            with self.assertRaises(ValueError, msg=f'{kw!r} was accepted'):
                Order(side='UP', **kw)
