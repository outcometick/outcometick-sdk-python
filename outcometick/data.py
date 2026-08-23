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
import os
import urllib.error
import urllib.parse
import urllib.request

__all__ = ("DataClient", "OutcometickError", "NO_VALUE", "DEFAULT_BASE_URL")

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
              asset=None, interval=None):
        """Search for files across a date range.

        :param date:     one day -- sugar for ``from_ == to``. Not combinable
                         with ``from_``/``to``.
        :param from_:    inclusive start; defaults to the newest day in scope.
        :param to:       inclusive end; defaults to ``from_``.
        :param venue:    ``polymarket`` | ``predict-fun``
        :param dataset:  ``prices`` | ``twap60s`` | ``book`` | ... (see meta())
        :param asset:    ``BTCUSD``, ``ETHUSD``, ...
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
        res, url = self._request(path)

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
                    f"checksum mismatch for {date}/{name}\n"
                    f"  expected {expected}\n  got      {got}"
                )

        if save_to:
            with open(save_to, "wb") as fh:
                fh.write(payload)

        return {"bytes": payload, "sha256": expected, "name": name, "date": date}

    # -- public, no key needed --------------------------------------------

    def coverage(self):
        """Coverage across all venues. Public -- works without a key."""
        return self._json("/v1/public/coverage", auth=False)

    def plans(self):
        """Plans and live prices. Public."""
        return self._json("/v1/public/plans", auth=False)

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
