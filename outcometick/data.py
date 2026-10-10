"""The data-subscription client: ``outcometick.data``.

    from outcometick.data import DataClient

    ot = DataClient()                                    # key from OT_KEY
    res = ot.files(asset=["btc", "eth"], dataset="prices")
    ot.download(res["files"][0], save_to="btc.csv.gz")   # verifies the checksum

Deliberately a SUBMODULE, not part of ``outcometick`` itself. The top level
exports the strategy SDK -- ``Strategy`` and ``Order`` -- which is what a
backtest imports, and that code runs in a container with no network at all.
Keeping the HTTP client one import away means a strategy cannot reach it by
accident, and the submission analyser rejects the import outright.

Uses only the standard library. A strategy SDK that drags in ``requests`` costs
every user a dependency for something ``urllib`` already does.

The shapes here were read off api/subscription-api.mjs rather than the docs
page: the published curl example shows only ``asset`` and ``dataset``, while
/v1/files also takes a date range, a venue and an interval, and every filter
accepts comma-separated alternatives plus a ``none`` sentinel.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request

__all__ = ("DataClient", "OrderBook", "OutcometickError", "NO_VALUE", "DEFAULT_BASE_URL")

DEFAULT_BASE_URL = "https://outcometick.com"

#: The sentinel that names files with no value for a dimension.
NO_VALUE = "none"


class OutcometickError(Exception):
    """An API error, carrying whatever the server said alongside the status.

    A 403 on a date outside coverage also reports the real ``floor`` and
    ``ceiling``. That is the difference between "you cannot have this" and "you
    cannot have this, here is what you can have", so the body is kept rather
    than flattened into a message.
    """

    def __init__(self, status: int, body, url: str) -> None:
        detail = body.get("error") if isinstance(body, dict) else str(body)[:200]
        super().__init__(f"{status} {detail}")
        self.status = status
        self.detail = detail
        self.body = body
        self.url = url


def _filter_value(value):
    """Render one filter.

    A list joins with commas because that is exactly what the API means by
    ``asset=btc,eth`` -- alternatives, not a nested structure. Passing a list is
    the friendlier spelling of the same request, so both work.
    """
    if value is None:
        return None
    parts = value if isinstance(value, (list, tuple)) else [value]
    parts = [str(p).strip() for p in parts if str(p).strip()]
    return ",".join(parts) if parts else None


class DataClient:
    """Client for the outcometick data subscription API."""

    def __init__(self, key=None, base_url=DEFAULT_BASE_URL, timeout=60, opener=None):
        # Read at construction so "you have not set a key" is raised once and
        # early, rather than as a 401 from whichever call happened to be first.
        self.key = key if key is not None else os.environ.get("OT_KEY")
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout
        self._opener = opener or urllib.request.build_opener(_NoRedirect())

    # -- plumbing ---------------------------------------------------------

    def _require_key(self) -> str:
        if not self.key:
            raise ValueError(
                "no API key.\n"
                "  Pass one as DataClient(key=...), or set OT_KEY:\n"
                '    export OT_KEY="ck_..."'
            )
        return self.key

    def _request(self, path, query=None, auth=True, follow=False):
        url = self.base_url + path
        params = {}
        for k, v in (query or {}).items():
            rendered = _filter_value(v)
            if rendered is not None:
                params[k] = rendered
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url)
        if auth:
            req.add_header("Authorization", f"Bearer {self._require_key()}")

        opener = urllib.request.build_opener() if follow else self._opener
        try:
            return opener.open(req, timeout=self.timeout), url
        except urllib.error.HTTPError as err:
            if 300 <= err.code < 400 and not follow:
                # _NoRedirect turns a redirect into an HTTPError so the caller
                # can read the headers off it; that is a result here, not a
                # failure.
                return err, url
            raw = err.read()
            try:
                body = json.loads(raw)
            except Exception:
                body = raw.decode("utf-8", "replace")
            raise OutcometickError(err.code, body, url) from None
        except urllib.error.URLError as err:
            raise OSError(f"could not reach {self.base_url}: {err.reason}") from None

    def _json(self, path, query=None, auth=True):
        res, url = self._request(path, query=query, auth=auth, follow=True)
        with res:
            raw = res.read()
        try:
            return json.loads(raw)
        except Exception:
            raise OutcometickError(res.status, raw.decode("utf-8", "replace"), url) from None

    # -- discovery --------------------------------------------------------

    def meta(self):
        """The date window this key can see, and every dimension value in it.

        ``assets`` and ``intervals`` hold real symbols and real durations only.
        The ``none`` sentinel is reported separately under ``filterTokens``,
        because a caller that builds an enum from ``intervals`` or parses them
        as durations must not meet a token.
        """
        return self._json("/v1/meta")

    def days(self):
        """The days this key may download, with the window's floor and ceiling."""
        return self._json("/v1/mirror/days")

    def files(self, date=None, from_=None, to=None, venue=None, dataset=None,
              asset=None, interval=None, format=None):
        """Search for files across a date range.

        :param date:     one day -- sugar for ``from_ == to``. Not combinable
                         with ``from_``/``to``.
        :param from_:    inclusive start; defaults to the newest day in scope.
        :param to:       inclusive end; defaults to ``from_``.
        :param venue:    ``polymarket`` | ``predict-fun``
        :param dataset:  ``prices`` | ``twap60s`` | ``book`` | ... (see meta())
        :param asset:    the BASE symbol -- ``BTC``, ``ETH``, ``SOL``, ... NOT
                         the pair. Files are named ``BTCUSD-...`` but the
                         dimension is ``BTC``; ``"BTCUSD"`` matches nothing.
                         ``meta()["assets"]`` lists the real values.
        :param interval: ``5m``, ``1h``, ... or ``NO_VALUE`` for the streams
                         that have no period

        Every filter takes a string or a list; a list means "any of these".
        ``interval=["5m", NO_VALUE]`` is how you ask for 5-minute files AND the
        period-less settlement streams -- asking for ``"5m"`` alone
        deliberately excludes them.

        The server caps the range (92 days by default) and answers 400 past it.
        """
        if date and (from_ or to):
            # The server rejects this too; catching it here saves a round trip
            # and says the same thing, so the two cannot describe it
            # differently.
            raise ValueError("use either date, or from_/to -- not both")
        return self._json("/v1/files", {
            "date": date, "from": from_, "to": to,
            "venue": venue, "dataset": dataset, "asset": asset, "interval": interval,
            # "parquet" lists the Parquet copy of each file instead; None = the .gz archive files.
            "format": format,
        })

    # -- download ---------------------------------------------------------

    def sign_url(self, date, name, expires_in=None):
        """A presigned URL for one file, without fetching it.

        Useful when something else does the fetching -- pandas, a job runner, a
        browser. The URL is short-lived; hold the ``(date, name)`` pair and ask
        again rather than storing it.
        """
        query = {"date": date, "name": name}
        if expires_in:
            query["expiresIn"] = expires_in
        return self._json("/v1/mirror/download", query)

    def download(self, file_or_date, name=None, verify=True, save_to=None):
        """Download one file.

        Accepts either a row from :meth:`files` or an explicit date and name.

        The checksum is verified by default. /v1/dl answers 302 with the sha256
        in a header and the bytes come from R2 behind the redirect, so the
        redirect is followed MANUALLY: letting urllib follow it would discard
        the header, and with it the only checksum available without a second
        API call. A row from :meth:`files` carries its own sha256, which is
        used when present.
        """
        if isinstance(file_or_date, dict):
            row = file_or_date
            date, name, expected = row.get("date"), row.get("name"), row.get("sha256")
        else:
            date, expected = file_or_date, None
        if not date or not name:
            raise ValueError("download needs a file row, or a date and a name")

        path = f"/v1/dl/{urllib.parse.quote(str(date))}/{urllib.parse.quote(str(name))}"
        payload, expected = self._download_via(path, None, expected, f"{date}/{name}", verify, save_to)
        return {"bytes": payload, "sha256": expected, "name": name, "date": date}

    def _download_via(self, path, query, expected, label, verify=True, save_to=None):
        """GET a route that answers 302 to a signed URL, fetch, verify.

        Shared by the archive and smart-money downloads.
        """
        res, url = self._request(path, query=query)

        status = getattr(res, "status", None) or getattr(res, "code", None)
        if status and 300 <= status < 400:
            expected = expected or res.headers.get("x-outcometick-sha256") \
                or res.headers.get("x-amz-meta-sha256")
            location = res.headers.get("location")
            res.close()
            if not location:
                raise OutcometickError(status, {"error": "redirect with no location"}, url)
            # The signed URL carries its own auth; sending ours to R2 as well
            # would leak the key to a host that has no use for it.
            try:
                with urllib.request.urlopen(location, timeout=self.timeout) as blob:
                    payload = blob.read()
            except urllib.error.HTTPError as err:
                raise OutcometickError(err.code, err.read().decode("utf-8", "replace"), url) from None
        else:
            with res:
                payload = res.read()

        if verify and expected:
            got = hashlib.sha256(payload).hexdigest()
            if got != expected:
                raise ValueError(
                    f"checksum mismatch for {label}\n"
                    f"  expected {expected}\n  got      {got}"
                )

        if save_to:
            with open(save_to, "wb") as fh:
                fh.write(payload)

        return payload, expected

    # -- smart money --------------------------------------------------------
    #
    # A separate subscription with its OWN key (a data key gets 403 here, and a
    # smart-money key gets 403 on everything above). Daily files of the trades
    # made by the top-ranked Polymarket traders: list ``top100``, or
    # ``top1000`` on the Top 1000 plan. While it is not on sale these answer 503.

    def smart_days(self):
        """The days this smart-money key may download, newest first, with each list's status."""
        return self._json("/v1/smart/days")

    def smart_download(self, day, list_="top100", verify=True, save_to=None):
        """Download one day's smart-money file (a zstd-compressed CSV).

        Verified against the sha256 the server sends with the redirect.

        :param day:   ``YYYY-MM-DD`` -- take it from :meth:`smart_days`
        :param list_: ``top100`` | ``top1000``
        """
        if not day:
            raise ValueError("smart_download needs a day")
        if list_ not in ("top100", "top1000"):
            raise ValueError("list_ must be 'top100' or 'top1000'")
        payload, expected = self._download_via(
            "/v1/smart/download", {"day": day, "list": list_}, None, f"{day}/{list_}", verify, save_to,
        )
        return {"bytes": payload, "sha256": expected, "day": day, "list": list_}

    # -- public, no key needed --------------------------------------------

    def coverage(self):
        """Coverage across all venues. Public -- works without a key."""
        return self._json("/v1/public/coverage", auth=False)

    def plans(self):
        """Plans and live prices. Public."""
        return self._json("/v1/public/plans", auth=False)

    def smart_coverage(self):
        """How many smart-money days are published, and from when. Public."""
        return self._json("/v1/public/smart-coverage", auth=False)

    def smart_plans(self):
        """Smart-money plans and prices (USD), and whether it is on sale. Public."""
        return self._json("/v1/public/smart-plans", auth=False)

    def health(self):
        """Liveness. Public."""
        return self._json("/v1/health", auth=False)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surface redirects instead of following them.

    /v1/dl answers 302 and the checksum rides on that response. urllib follows
    redirects by default, which would throw the header away before anything
    could read it.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# ---------- rebuilding a Polymarket order book from archive rows ----------

_NUMERIC = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", re.ASCII)


def _num(x):
    """A finite float, or NaN. Only a finite number or a plain decimal string
    ("0.5", "12", "1e-1") counts -- float() alone would take " 1 ", "1_0" and
    "inf". Same rule as the JS client."""
    if isinstance(x, bool):
        return float("nan")
    if isinstance(x, (int, float)) or (isinstance(x, str) and _NUMERIC.fullmatch(x)):
        try:
            n = float(x)
        except OverflowError:  # an int too large for a float
            return float("nan")
        return n if math.isfinite(n) else float("nan")  # "1e400"
    return float("nan")


def _price_key(p):
    """"0.50" and "0.5" are the same level."""
    return repr(_num(p))


def _ladder_side(side):
    s = str(side if side is not None else "").upper()
    return "bids" if s == "BUY" else "asks" if s == "SELL" else None


def _prune_bound(x):
    """A best price that can be pruned against: a number in [0, 1]."""
    n = _num(x)
    return n if n == n and 0 <= n <= 1 else None


def _text(x):
    """The archive's own spelling of a price or size (mirrors JS String())."""
    if isinstance(x, str):
        return x
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


class OrderBook:
    """Rebuilds a Polymarket order book, per outcome token, from the archive's
    ``book``, ``price_change`` and ``best_bid_ask`` rows fed in receipt order
    (``recv_ms``; merge the three files by it)::

        book = OrderBook()
        for row in rows:
            book.apply(row)
        book.ladder(token_id)   # {"bids": [{"price", "size"}, ...], "asks": [...]}, best first

    - ``book`` replaces that token's whole ladder.
    - ``price_change`` sets each level to an absolute size; size 0 removes it.
    - Then any bid above the best bid, or ask below the best ask, is dropped --
      the best prices come from ``best_bid_ask`` and from the ``best_bid`` /
      ``best_ask`` each ``price_change`` item carries.

    Why the last step: book and price_change are captured at a cadence, so a
    removal can fall between two stored frames and leave its level behind until
    the next snapshot. Pruning against the newest best prices removes every
    level they have moved past. It cannot restore what dropped frames added or
    resized: until the next snapshot a level -- at the top too -- can be missing
    or carry an old size, because best_bid_ask carries prices only. ``best()`` is
    the best level of the rebuilt ladder, not the market's latest best bid and
    ask; read best_bid_ask rows for those.

    Prices and sizes are returned as the archive wrote them (strings). Rows of
    other types are ignored. Predict.fun order-book rows are full snapshots on
    their own and need none of this. Mirrors ``OrderBook`` in the JS client.
    """

    def __init__(self):
        self._tokens = {}

    def _ladders(self, asset_id):
        if asset_id not in self._tokens:
            self._tokens[asset_id] = {"bids": {}, "asks": {}}
        return self._tokens[asset_id]

    def _prune(self, asset_id, best_bid, best_ask):
        lad = self._tokens.get(asset_id)
        if lad is None:
            return
        bid = _prune_bound(best_bid)
        ask = _prune_bound(best_ask)
        if bid is not None:
            for k in [k for k, lvl in lad["bids"].items() if _num(lvl["price"]) > bid]:
                del lad["bids"][k]
        if ask is not None:
            for k in [k for k, lvl in lad["asks"].items() if _num(lvl["price"]) < ask]:
                del lad["asks"][k]

    def apply(self, row):
        """Apply one archive row (a dict, or the JSONL line itself). Returns self."""
        r = json.loads(row) if isinstance(row, str) else row
        if not isinstance(r, dict):
            return self
        p = r.get("payload") if isinstance(r.get("payload"), dict) else {}
        kind = r.get("event_type") if r.get("event_type") is not None else p.get("event_type")
        row_asset = r.get("asset_id") if r.get("asset_id") is not None else p.get("asset_id")
        if kind == "book":
            if row_asset is None:
                return self
            lad = self._ladders(str(row_asset))
            for side in ("bids", "asks"):
                lad[side].clear()
                levels = p.get(side) if isinstance(p.get(side), list) else []
                for lvl in levels:
                    if isinstance(lvl, list):
                        price, size = (lvl + [None, None])[:2]
                    elif isinstance(lvl, dict):
                        price, size = lvl.get("price"), lvl.get("size")
                    else:
                        continue
                    if _num(price) == _num(price) and _num(size) > 0:
                        lad[side][_price_key(price)] = {"price": _text(price), "size": _text(size)}
        elif kind == "price_change":
            items = p.get("price_changes") if isinstance(p.get("price_changes"), list) else (
                p.get("changes") if isinstance(p.get("changes"), list) else [])
            bests = {}
            for it in items:
                if not isinstance(it, dict):
                    continue
                asset = it.get("asset_id") if it.get("asset_id") is not None else row_asset
                side = _ladder_side(it.get("side"))
                size = _num(it.get("size"))
                if asset is None or side is None or _num(it.get("price")) != _num(it.get("price")) or size != size:
                    continue
                lad = self._ladders(str(asset))
                if size > 0:
                    lad[side][_price_key(it["price"])] = {"price": _text(it["price"]), "size": _text(it["size"])}
                else:
                    lad[side].pop(_price_key(it["price"]), None)
                bests.pop(str(asset), None)
                bests[str(asset)] = it
            for asset, it in bests.items():
                self._prune(asset, it.get("best_bid"), it.get("best_ask"))
        elif kind == "best_bid_ask":
            if row_asset is not None:
                self._prune(str(row_asset), p.get("best_bid"), p.get("best_ask"))
        return self

    def assets(self):
        """Token ids seen so far."""
        return list(self._tokens)

    def ladder(self, asset_id):
        """The token's ladder, best level first on each side."""
        lad = self._tokens.get(str(asset_id))
        if lad is None:
            return {"bids": [], "asks": []}
        return {
            "bids": [dict(x) for x in sorted(lad["bids"].values(), key=lambda x: -_num(x["price"]))],
            "asks": [dict(x) for x in sorted(lad["asks"].values(), key=lambda x: _num(x["price"]))],
        }

    def best(self, asset_id):
        """The best bid and ask of the rebuilt ladder (None when that side is empty)."""
        lad = self.ladder(asset_id)
        return {"bid": lad["bids"][0] if lad["bids"] else None, "ask": lad["asks"][0] if lad["asks"] else None}
