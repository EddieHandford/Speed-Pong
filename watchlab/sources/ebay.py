"""Adapter for eBay's Browse API (https://developer.ebay.com/api-docs/buy/browse/overview.html).

Built against the documented OAuth2 client-credentials grant and the
``item_summary/search`` resource -- unlike Chrono24, this is a real,
free-to-register REST API with a generous default quota (5,000 calls/day),
so it replaces the "save pages by hand" workflow entirely: no scraping, no
ToS conflict, no HTML selectors to keep fixing.

READ THIS BEFORE RUNNING IT
----------------------------
Two credentials, never committed to the repo: ``EBAY_CLIENT_ID`` and
``EBAY_CLIENT_SECRET`` (an eBay "application keyset" -- production keys, not
the sandbox ones, since sandbox has no real listings). Get them free at
https://developer.ebay.com by registering and creating a keyset.

The client-credentials grant produces an *Application* access token (per
eBay's docs, ~2 hours), which is enough for ``item_summary/search`` -- no
user login flow needed since we're only reading public listings, not acting
on anyone's account. The token itself is fetched fresh per process and never
written to disk (re-fetching costs nothing; it isn't part of the metered
quota). Search results *are* cached to disk like
:mod:`watchlab.sources.thewatchapi`, since those calls do count against the
daily limit.

Condition mapping is best-effort against eBay's documented condition
vocabulary (``New``, ``Used``, ``For parts or not working``, etc.) -- an
unrecognised string stays unmapped (``None``) rather than guessed, the same
rule ``normalize.py`` already applies to box/papers status.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .chrono24 import RawListing

TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
BASE_URL = "https://api.ebay.com/buy/browse/v1"
DEFAULT_SCOPE = "https://api.ebay.com/oauth/api_scope"
USER_AGENT = "watchlab/0.1 (personal research)"

_RETRYABLE_STATUSES = frozenset({429, 500, 503})

TokenFetcher = Callable[[str, str, str, str, float], tuple[int, dict[str, Any]]]
Opener = Callable[[str, str, str, float], tuple[int, dict[str, str], bytes]]


class EbayError(RuntimeError):
    """A documented eBay REST error envelope, an OAuth failure, or a transport failure."""

    def __init__(self, code: str, message: str, http_status: int):
        super().__init__(f"{code} (HTTP {http_status}): {message}" if message else
                         f"{code} (HTTP {http_status})")
        self.code = code
        self.message = message
        self.http_status = http_status


def _default_token_fetcher(
    client_id: str, client_secret: str, scope: str, token_url: str, timeout: float,
) -> tuple[int, dict[str, Any]]:
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    body = urllib.parse.urlencode({"grant_type": "client_credentials", "scope": scope}).encode()
    request = urllib.request.Request(
        token_url, data=body, method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {credentials}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads((exc.read() or b"{}").decode("utf-8") or "{}")


def _default_opener(url: str, token: str, marketplace_id: str, timeout: float
                     ) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": marketplace_id,
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


@dataclass
class Client:
    """OAuth2 client-credentials REST client for the Browse API: auth, caching, retry.

    ``marketplace_id`` scopes both the currency of returned prices and which
    site's inventory is searched -- ``EBAY_GB`` for GBP/UK listings (the
    default here, since this project's budget-watch work is priced in GBP),
    ``EBAY_US`` for USD, etc. See eBay's marketplace ID list for the full set.
    """

    client_id: str
    client_secret: str
    marketplace_id: str = "EBAY_GB"
    cache_dir: str | None = None
    scope: str = DEFAULT_SCOPE
    token_url: str = TOKEN_URL
    base_url: str = BASE_URL
    timeout: float = 20.0
    max_retries: int = 3
    token_fetcher: TokenFetcher = _default_token_fetcher
    opener: Opener = _default_opener

    _token: str | None = field(default=None, init=False, repr=False)
    _token_expires_at: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "no eBay credentials. Set EBAY_CLIENT_ID/EBAY_CLIENT_SECRET or pass "
                "--client-id/--client-secret; never hardcode them."
            )
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _token_value(self) -> str:
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token
        status, payload = self.token_fetcher(
            self.client_id, self.client_secret, self.scope, self.token_url, self.timeout
        )
        if status != 200:
            message = payload.get("error_description") or payload.get("error") or ""
            raise EbayError("oauth_failed", str(message), status)
        self._token = payload["access_token"]
        self._token_expires_at = time.time() + float(payload.get("expires_in", 7200))
        return self._token

    def _cache_path(self, cache_key: str) -> str:
        digest = hashlib.sha256(cache_key.encode()).hexdigest()[:24]
        return os.path.join(self.cache_dir, f"{digest}.json")  # type: ignore[arg-type]

    def get(self, path: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> dict:
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        cache_key = f"{self.marketplace_id}:{path}?" + urllib.parse.urlencode(sorted(clean_params.items()))

        if use_cache and self.cache_dir:
            cache_path = self._cache_path(cache_key)
            if os.path.exists(cache_path):
                with open(cache_path, encoding="utf-8") as handle:
                    return json.load(handle)

        url = f"{self.base_url}{path}?" + urllib.parse.urlencode(clean_params)

        delay = 2.0
        status = payload = None
        retried_on_auth = False
        for attempt in range(self.max_retries + 1):
            status, _headers, body = self.opener(url, self._token_value(), self.marketplace_id, self.timeout)
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise EbayError("invalid_response", str(exc), status or 0) from exc

            if status == 200:
                if use_cache and self.cache_dir:
                    with open(self._cache_path(cache_key), "w", encoding="utf-8") as handle:
                        json.dump(payload, handle)
                return payload

            if status == 401 and not retried_on_auth:
                # Expired/invalid token: force one refetch, outside the
                # generic retry budget below.
                self._token = None
                retried_on_auth = True
                continue

            if status in _RETRYABLE_STATUSES and attempt < self.max_retries:
                time.sleep(delay)
                delay *= 3
                continue
            break

        errors = (payload or {}).get("errors") or []
        first = errors[0] if errors else {}
        raise EbayError(
            str(first.get("errorId", "unknown_error")), first.get("message", ""), status or 0
        )


def search_items(
    client: Client,
    query: str,
    limit: int = 50,
    filter: str | None = None,
    category_ids: str | None = None,
) -> list[dict[str, Any]]:
    """One page of ``item_summary/search`` results (max 200/page per the docs)."""
    params = {"q": query, "limit": limit, "filter": filter, "category_ids": category_ids}
    payload = client.get("/item_summary/search", params)
    return payload.get("itemSummaries") or []


_CONDITION_MAP = {
    "new": "new",
    "new with tags": "new",
    "new without tags": "new",
    "new with defects": "new",
    "new other": "new",
    "open box": "unworn",
    "certified refurbished": "very_good",
    "excellent refurbished": "very_good",
    "excellent - refurbished": "very_good",
    "very good refurbished": "good",
    "very good - refurbished": "good",
    "good refurbished": "good",
    "good - refurbished": "good",
    "seller refurbished": "good",
    "used": "good",
    "pre-owned": "good",
    "for parts or not working": "poor",
}


def _map_condition(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return _CONDITION_MAP.get(value.strip().lower())


def to_raw_listing(item: dict[str, Any]) -> RawListing | None:
    """Map one Browse API ``ItemSummary`` onto watchlab's ``RawListing`` shape.

    Returns ``None`` for anything missing an id or a price -- there is
    nothing useful to store without either, and silently skipping is right
    here (unlike an unresolved *reference*, which ``ingest`` deliberately
    keeps so the coverage gap stays visible).
    """
    item_id = item.get("itemId")
    price = item.get("price") or {}
    if not item_id or price.get("value") is None:
        return None
    location = item.get("itemLocation") or {}
    seller = item.get("seller") or {}
    seller_type = None
    account_type = seller.get("sellerAccountType")
    if account_type == "BUSINESS":
        seller_type = "dealer"
    elif account_type == "INDIVIDUAL":
        seller_type = "private"
    return RawListing(
        listing_id=f"ebay:{item_id}",
        url=item.get("itemWebUrl"),
        title=item.get("title"),
        price_text=str(price.get("value")),
        currency=price.get("currency"),
        condition_text=_map_condition(item.get("condition")),
        seller_country=location.get("country"),
        seller_type=seller_type,
        extras={"source_strategy": "ebay_browse_api"},
    )


def search_watchlist(
    client: Client, queries: Sequence[str], limit: int = 50, filter: str | None = None,
) -> list[RawListing]:
    """One ``item_summary/search`` call per query, flattened and deduplicated.

    One call per watchlist entry -- for a watchlist the size of a personal
    screen (tens of references, not thousands), that's nowhere near the
    5,000/day default quota.
    """
    out: list[RawListing] = []
    seen: set[str] = set()
    for query in queries:
        for item in search_items(client, query, limit=limit, filter=filter):
            listing = to_raw_listing(item)
            if listing and listing.listing_id not in seen:
                seen.add(listing.listing_id)
                out.append(listing)
    return out
