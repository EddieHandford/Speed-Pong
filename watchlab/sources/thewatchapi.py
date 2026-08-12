"""Adapter for thewatchapi.com (https://www.thewatchapi.com/documentation).

Written against the real documentation and example responses, not guessed --
this sandbox cannot reach the live API to verify anything itself, so every
response shape parsed here matches an example payload actually shown in the
docs, and ``tests/test_thewatchapi.py`` exercises the parser against those
same examples with the network layer swapped for a fake.

READ THIS BEFORE LOOPING ANYTHING
----------------------------------
thewatchapi meters usage by "data credits" and flags some endpoints
'HIGH USAGE': ``model/search``, ``brand/price/history``,
``model/price/history``. There is no documented pagination on any list/search
endpoint, and a brand-wide query can hit ``too_many_results`` (400) if it
exceeds your plan's allowance. This module reflects that split in its API:

  * cheap, "All plans" endpoints (``brand/list``, ``model/list``,
    ``reference/list``, ``brand/search``, ``reference/search``) are what
    :func:`sync_brand_references` uses by default -- they give you every
    reference number for a brand plus nothing else, at low cost.
  * enrichment via ``model/search`` (case, movement, production years) is
    opt-in and always one reference at a time -- nothing here loops a whole
    brand through it automatically. Do that deliberately, on a curated
    watchlist, not on every reference a brand has ever made.
  * ``*/price/history`` requires a Standard-plan-or-above subscription and
    returns *indicative asking prices*, calculated from online listings --
    the same caveat as everything else asks-based in this project. Stored
    separately, in ``provider_price_series``, never merged with the hedonic
    index.

Do not commit an API token to this repository. Pass it via the
``THEWATCHAPI_TOKEN`` environment variable or a ``--token`` CLI argument that
you type at the shell, not a hardcoded default.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable

BASE_URL = "https://api.thewatchapi.com/v1"
USER_AGENT = "watchlab/0.1 (personal research)"

# Endpoints the docs mark HIGH USAGE -- surfaced here so callers (and the
# CLI) can warn before spending credits on them.
HIGH_USAGE_ENDPOINTS = frozenset({
    "/model/search", "/brand/price/history", "/model/price/history",
})

# Retried with backoff: transient (429 rate limit, 500/503 server-side).
_RETRYABLE_STATUSES = frozenset({429, 500, 503})

Opener = Callable[[str, float], tuple[int, dict[str, str], bytes]]


class ThewatchapiError(RuntimeError):
    """One of the documented error envelopes, or a transport failure."""

    def __init__(self, code: str, message: str, http_status: int):
        super().__init__(f"{code} (HTTP {http_status}): {message}" if message else
                         f"{code} (HTTP {http_status})")
        self.code = code
        self.message = message
        self.http_status = http_status


def _default_opener(url: str, timeout: float) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


@dataclass
class Client:
    """Thin REST client: auth, caching, retry/backoff, typed errors.

    Caching is keyed on the full request URL (including params, excluding the
    token) and never expires automatically -- reference/brand/model data is
    near-static, and this is a metered API, so a stale cache hit costs
    nothing while a fresh one costs a credit. Pass ``use_cache=False`` on a
    call where you specifically want current data (e.g. re-running price
    history after the token's ``date_to`` has moved forward).
    """

    token: str
    cache_dir: str | None = None
    base_url: str = BASE_URL
    timeout: float = 20.0
    max_retries: int = 3
    opener: Opener = _default_opener

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError(
                "no API token. Set THEWATCHAPI_TOKEN or pass --token; never hardcode one."
            )
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, cache_key: str) -> str:
        digest = hashlib.sha256(cache_key.encode()).hexdigest()[:24]
        return os.path.join(self.cache_dir, f"{digest}.json")  # type: ignore[arg-type]

    def get(self, path: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> dict:
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        cache_key = path + "?" + urllib.parse.urlencode(sorted(clean_params.items()))

        if use_cache and self.cache_dir:
            cache_path = self._cache_path(cache_key)
            if os.path.exists(cache_path):
                with open(cache_path, encoding="utf-8") as handle:
                    return json.load(handle)

        request_params = dict(clean_params)
        request_params["api_token"] = self.token
        url = f"{self.base_url}{path}?{urllib.parse.urlencode(request_params)}"

        delay = 2.0
        status = payload = None
        for attempt in range(self.max_retries + 1):
            status, _headers, body = self.opener(url, self.timeout)
            try:
                payload = json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ThewatchapiError("invalid_response", str(exc), status or 0) from exc

            if status == 200:
                if use_cache and self.cache_dir:
                    with open(self._cache_path(cache_key), "w", encoding="utf-8") as handle:
                        json.dump(payload, handle)
                return payload

            if status in _RETRYABLE_STATUSES and attempt < self.max_retries:
                time.sleep(delay)
                delay *= 3
                continue
            break

        error = (payload or {}).get("error", {}) if isinstance(payload, dict) else {}
        raise ThewatchapiError(
            error.get("code", "unknown_error"), error.get("message", ""), status or 0
        )


# --- lists: cheap, "All plans" -------------------------------------------

def list_brands(client: Client) -> list[str]:
    return client.get("/brand/list")["data"]


def list_models(client: Client, brand: str) -> list[str]:
    return client.get("/model/list", {"brand": brand})["data"]


def list_references(client: Client, brand: str) -> list[str]:
    """Bare reference numbers for a brand -- no metadata, cheap, 'All plans'."""
    return client.get("/reference/list", {"brand": brand})["data"]


# --- search: brand/reference search are cheap; model/search is HIGH USAGE -

def search_brands(client: Client, search: str) -> list[str]:
    return client.get("/brand/search", {"search": search})["data"]


def search_references(client: Client, search: str) -> list[dict[str, str]]:
    """Minimal: [{'brand', 'reference_number'}, ...]. 'All plans'."""
    return client.get("/reference/search", {"search": search})["data"]


def search_models(
    client: Client,
    search: str,
    search_attributes: str | None = None,
    brand: str | None = None,
    model: str | None = None,
    reference_number: str | None = None,
    movement: str | None = None,
    case_material: str | None = None,
    case_diameter: str | None = None,
) -> list[dict[str, Any]]:
    """Rich per-watch records: case, movement, production years, description.

    HIGH USAGE per the docs. Call this per reference you actually care about,
    not in a loop over an entire brand's catalogue.
    """
    params = {
        "search": search, "search_attributes": search_attributes, "brand": brand,
        "model": model, "reference_number": reference_number, "movement": movement,
        "case_material": case_material, "case_diameter": case_diameter,
    }
    return client.get("/model/search", params)["data"]


def enrich_reference(client: Client, brand: str, reference_number: str) -> dict[str, Any] | None:
    """One HIGH USAGE call for one reference's full metadata, or None if unmatched."""
    results = search_models(
        client, search=reference_number, search_attributes="reference_number",
        brand=brand, reference_number=reference_number,
    )
    return results[0] if results else None


# --- price history: Standard plan+, asking prices, HIGH USAGE for brand/model

def brand_price_history(
    client: Client, brand: str, date_from: str | None = None, date_to: str | None = None
) -> dict[str, Any]:
    return client.get("/brand/price/history", {"brand": brand, "date_from": date_from, "date_to": date_to})


def model_price_history(
    client: Client, model: str, date_from: str | None = None, date_to: str | None = None
) -> dict[str, Any]:
    return client.get("/model/price/history", {"model": model, "date_from": date_from, "date_to": date_to})


def reference_price_history(
    client: Client, reference_number: str, date_from: str | None = None, date_to: str | None = None
) -> dict[str, Any]:
    """Not HIGH USAGE per the docs, unlike its brand/model siblings."""
    return client.get(
        "/reference/price/history",
        {"reference_number": reference_number, "date_from": date_from, "date_to": date_to},
    )


# --- conversion into watchlab's intermediate catalogue format -------------

_YEAR = re.compile(r"\d{4}")
_ONGOING = re.compile(r"present|current|ongoing|now", re.IGNORECASE)
_CASE_MM = re.compile(r"([\d.]+)\s*mm", re.IGNORECASE)


def _parse_year_range(text: str | None) -> tuple[int | None, int | None]:
    """'1989 - 2018' -> (1989, 2018); '2020 - Present' -> (2020, None)."""
    if not text:
        return None, None
    years = [int(y) for y in _YEAR.findall(text)]
    if not years:
        return None, None
    start = years[0]
    if len(years) >= 2:
        return start, years[1]
    if _ONGOING.search(text):
        return start, None
    return start, start


def _parse_case_mm(text: str | None) -> float | None:
    if not text:
        return None
    match = _CASE_MM.search(text)
    return float(match.group(1)) if match else None


def to_catalogue_record(record: dict[str, Any]) -> dict[str, Any]:
    """Map a rich model/search result onto watchlab.catalogue's record shape.

    thewatchapi has no retail-price field, so ``retail_price`` is left unset
    here; the catalogue upsert leaves any existing value untouched rather
    than clearing it, so this can be layered under a source that does supply
    retail prices without data loss in either direction.
    """
    start, end = _parse_year_range(record.get("year_of_production"))
    return {
        "reference": record["reference_number"],
        "brand": record["brand"],
        "model_name": record.get("model"),
        "production_start": start,
        "production_end": end,
        "case_mm": _parse_case_mm(record.get("case_diameter")),
    }


def minimal_catalogue_record(brand: str, reference_number: str) -> dict[str, Any]:
    """Map a bare brand+reference pair (from list/search) onto the same shape."""
    return {"reference": reference_number, "brand": brand}


def sync_brand_references(client: Client, brand: str) -> list[dict[str, Any]]:
    """The cheap default: every reference number for a brand, brand-tagged only.

    Uses ``reference/list`` (one call, 'All plans'), not ``model/search`` in a
    loop. Feed the result straight into
    :func:`watchlab.catalogue.upsert_catalogue`; enrich specific references
    afterwards with :func:`enrich_reference` if you want case/movement/year
    detail for a curated watchlist.
    """
    return [minimal_catalogue_record(brand, ref) for ref in list_references(client, brand)]


def price_series_rows(
    provider: str, scope_type: str, scope_value: str, payload: dict[str, Any], fetched_at: str,
) -> Iterable[tuple[str, str, str, str, int, str, str]]:
    """Flatten a price-history response into ``provider_price_series`` rows.

    thewatchapi quotes in USD with sub-cent float precision (e.g.
    20588.861509); rounded to the nearest cent, which is the finest unit the
    schema's INTEGER minor-units convention supports and is far inside the
    noise floor of an indicative asking-price average anyway.
    """
    for point in payload.get("data", []):
        date = str(point["date"])[:10]  # 'YYYY-MM-DDT...' -> 'YYYY-MM-DD'
        cents = round(float(point["price"]) * 100)
        yield provider, scope_type, scope_value, date, cents, "USD", fetched_at
