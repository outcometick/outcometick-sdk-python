"""Tests for the public Python SDK surface.

Small on purpose: this package is the SDK surface and nothing else, so these
assert the contract a strategy author actually depends on — the names that
exist, and the argument validation that rejects a bad order at construction
rather than silently at fill time.

They ship with the published package so the mirror is a repository someone can
clone and verify rather than take on faith.
"""

import unittest

from outcometick import Order, Strategy, SIDES


class TestSurface(unittest.TestCase):
    def test_exports_are_exactly_what_is_documented(self):
        import outcometick

        self.assertEqual(sorted(outcometick.__all__), ["Order", "SIDES", "Strategy"])
        # __all__ also keeps `annotations` from `from __future__ import` out of
        # the public namespace, which is what a reader sees in dir().
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

    def test_rejects_a_side_that_is_not_an_outcome_token(self):
        with self.assertRaises(ValueError):
            Order(side="SIDEWAYS", size=100)

    def test_rejects_a_non_positive_size(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                Order(side="UP", size=bad)

    def test_rejects_a_limit_outside_zero_to_one(self):
        # A binary outcome token trades between 0 and 1. Anything else is not a
        # price, and clamping it would fill an order nobody asked for.
        for bad in (-0.01, 1.5, 100):
            with self.assertRaises(ValueError):
                Order(side="UP", size=1, limit=bad)


class TestStrategy(unittest.TestCase):
    def test_params_default_to_empty_so_a_hook_can_read_them(self):
        self.assertEqual(Strategy().p, {})

    def test_hooks_are_not_defined_by_the_base_class(self):
        # Deliberate: a default no-op on_tick would turn "you declared a hook you
        # did not implement" — fixable in seconds — into a run that quietly never
        # trades and bills for an empty equity curve.
        for hook in ("on_tick", "on_market_open", "on_book", "on_trade", "on_settle"):
            self.assertFalse(hasattr(Strategy, hook), hook)


if __name__ == "__main__":
    unittest.main()


class TestStrictNumbers(unittest.TestCase):
    """The SDK accepts numbers, not things that can be coerced into numbers.

    The engine was tightened first, and the SDK kept coercing — so
    ``Order(limit=[0.5])`` raised TypeError in Python while
    ``new Order({limit:[0.5]})`` quietly became 0.5 in JavaScript. One
    published SDK, two languages, two behaviours, from the same input.

    `float('0.5')` and `Number('0.5')` agree; `float([0.5])` and
    `Number([0.5])` do not. Coercion is not validation, and the two languages
    do not coerce alike.
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
        # And a budget with no ceiling has no conversion.
        with self.assertRaises(ValueError):
            Order(side='UP', notional=80)

    def test_the_legal_values_still_work(self):
        # Tightening a check is only correct if it does not refuse the product.
        for limit in [0, 0.5, 1]:
            self.assertEqual(Order(side='UP', size=10, limit=limit).limit, float(limit))
        self.assertEqual(Order(side='UP', size=2.5).size, 2.5)
        # floor(80 / 0.64) is 125 — the boundary where Python's `//` disagrees.
        self.assertEqual(Order(side='UP', notional=80, limit=0.64).size, 125)
