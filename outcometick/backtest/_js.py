"""JavaScript value semantics the decoder depends on.

The archive is decoded by runner/events.mjs for the hosted queue and `ot run`,
and by this package for a pure-Python run. The two must turn the same row into
the same event, so wherever the JS leans on a language rule -- `Number(" 12 ")`
is 12, `Number("")` is 0, `(0.125).toFixed(2)` is "0.13", `String(1.0)` is "1",
JSON numbers are doubles -- the rule is reproduced here rather than
approximated with Python's nearest equivalent, which differs in each of those
cases.
"""

from __future__ import annotations

import json
import math
import re
from decimal import ROUND_HALF_UP, Decimal

# ECMAScript WhiteSpace + LineTerminator: what String.prototype.trim() and
# Number() strip. Not str.strip(), which also strips \x1c-\x1f and \x85 and
# keeps ﻿.
_JS_WS = (
    "\t\n\v\f\r          "
    "        　﻿"
)

_DECIMAL = re.compile(r"[+-]?(?:Infinity|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", re.ASCII)
_RADIX = re.compile(r"0([xXbBoO])([0-9a-fA-F]+)", re.ASCII)
_RADIX_BASE = {"x": 16, "b": 2, "o": 8}
_RADIX_DIGITS = {16: "0123456789abcdef", 2: "01", 8: "01234567"}


def js_trim(s: str) -> str:
    return s.strip(_JS_WS)


def _string_to_number(s: str) -> float:
    t = js_trim(s)
    if t == "":
        return 0.0
    m = _RADIX.fullmatch(t)
    if m:
        base = _RADIX_BASE[m.group(1).lower()]
        digits = m.group(2).lower()
        if any(ch not in _RADIX_DIGITS[base] for ch in digits):
            return math.nan
        try:
            return float(int(digits, base))
        except OverflowError:
            return math.inf
    if not _DECIMAL.fullmatch(t):
        return math.nan
    if t.endswith("Infinity"):
        return -math.inf if t.startswith("-") else math.inf
    return float(t)


def js_number(v) -> float:
    """`Number(v)` for the values JSON and CSV can produce."""
    if v is None:
        return 0.0
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, int):
        try:
            return float(v)
        except OverflowError:
            return math.inf if v > 0 else -math.inf
    if isinstance(v, float):
        return v
    if isinstance(v, str):
        return _string_to_number(v)
    if isinstance(v, list):
        # Number([]) is 0 and Number([x]) is Number(String(x)): an array
        # converts through its join.
        return _string_to_number(js_string(v))
    return math.nan


def num(v):
    """`num` in runner/events.mjs: a finite number, or None.

    An empty or blank string is None, not 0 -- the reason this exists.
    """
    if v is None:
        return None
    if isinstance(v, str) and js_trim(v) == "":
        return None
    n = js_number(v)
    return n if math.isfinite(n) else None


def _number_to_string(x: float) -> str:
    """`Number.prototype.toString()` (radix 10)."""
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0:
        return "0"
    if x < 0:
        return "-" + _number_to_string(-x)
    # repr gives the shortest round-tripping digits, the same digits JS picks;
    # only the layout differs.
    r = repr(x)
    if "e" in r:
        mant, exp = r.split("e")
        e = int(exp)
    else:
        mant, e = r, 0
    if "." in mant:
        ip, fp = mant.split(".")
    else:
        ip, fp = mant, ""
    digits = (ip + fp).lstrip("0")
    # n: position of the decimal point relative to the digit string.
    lead_zeros = len(ip + fp) - len((ip + fp).lstrip("0"))
    n = len(ip) - lead_zeros + e
    digits = digits.rstrip("0") or "0"
    k = len(digits)
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + digits
    exp_val = n - 1
    sign = "+" if exp_val >= 0 else "-"
    if k == 1:
        return f"{digits}e{sign}{abs(exp_val)}"
    return f"{digits[0]}.{digits[1:]}e{sign}{abs(exp_val)}"


def js_string(v) -> str:
    """`String(v)` for JSON values."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return _number_to_string(js_number(v))
    if isinstance(v, float):
        return _number_to_string(v)
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return ",".join("" if x is None else js_string(x) for x in v)
    if isinstance(v, dict):
        return "[object Object]"
    return str(v)


def js_to_fixed(x: float, digits: int) -> str:
    """`x.toFixed(digits)`: ties go away from zero, on the exact binary value."""
    if not math.isfinite(x):
        return _number_to_string(x)
    if abs(x) >= 1e21:
        return _number_to_string(x)
    q = Decimal(x).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    s = format(q, "f")
    if s.startswith("-") and Decimal(s) == 0:
        # (-0.0001).toFixed(2) is "-0.00" in JS too, so keep the sign; but
        # (-0).toFixed(2) is "0.00".
        return s if x != 0 else s[1:]
    return s


def js_round_fixed(x, digits: int):
    """`Number(x.toFixed(digits))` when finite, else None -- report.mjs r2/r4."""
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
        return None
    return float(js_to_fixed(float(x), digits)) + 0.0


def js_math_round(x: float) -> float:
    """`Math.round`: halves go towards +Infinity.

    Not floor(x + 0.5), which is wrong where x + 0.5 itself rounds:
    0.49999999999999994 and odd integers past 2**52.
    """
    f = math.floor(x)
    return float(f + 1 if x - f >= 0.5 else f)


# --- JSON -----------------------------------------------------------------

_MAX_SAFE = 2 ** 53


def _parse_int(s: str):
    # JSON.parse yields doubles. Exact below 2^53, where an int and its double
    # agree; above it, the double JS would hold.
    v = int(s)
    return v if -_MAX_SAFE <= v <= _MAX_SAFE else float(s)


def _reject_constant(name):
    raise ValueError(f"{name} is not JSON")


def json_loads(text: str):
    """`JSON.parse`: no NaN/Infinity literals, numbers as doubles."""
    return json.loads(text, parse_int=_parse_int, parse_constant=_reject_constant)


def split_lines(text: str):
    """Node readline with crlfDelay: Infinity -- \\n, \\r\\n and a lone \\r end a line."""
    return re.split(r"\r\n|\r|\n", text)
