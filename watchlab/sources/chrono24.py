"""Chrono24 listing ingest.

READ THIS BEFORE RUNNING IT
---------------------------
Chrono24's terms of service prohibit automated data collection, and the site
runs bot protection (JS challenges and fingerprinting) that a plain HTTP
client will not pass. Two consequences:

  * ``fetch_page`` checks robots.txt and refuses by default when the path is
    disallowed. The override exists for people who have their own arrangement
    with the site; it is not a way to pretend the rule is not there.
  * Even with the override, expect challenge pages rather than listings. The
    supported workflow is ``--from-file``: save pages from a normal browser
    session, point the parser at them, and calibrate the selectors. That keeps
    request volume at zero and is the only approach that reliably works.

The parsers below were written without live access to the site, so the CSS
class names in ``SELECTORS`` are unverified guesses and will need fixing on
first use. ``parse_listing_html`` therefore tries schema.org JSON-LD first,
which is a published standard and far less likely to drift than class names.
Run ``python -m watchlab calibrate <saved.html>`` to see what each strategy
extracts from a real page.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Iterator

BASE_URL = "https://www.chrono24.com"
USER_AGENT = "watchlab/0.1 (personal research; contact: set WATCHLAB_CONTACT)"
DEFAULT_DELAY_SECONDS = 8.0

# Unverified. Fix these against a real saved page before trusting the output.
SELECTORS = {
    "listing_container_attr": ("div", "class", re.compile(r"article-item|js-article-item")),
    "listing_id_attr": "data-article-id",
    "price_attr": ("span", "class", re.compile(r"currency|price")),
    "title_attr": ("div", "class", re.compile(r"text-sm text-sm-md text-bold|article-title")),
}

_ISO_COUNTRY = re.compile(r"\b([A-Z]{2})\b")


@dataclass
class RawListing:
    """Whatever we could pull off the page, before normalisation."""

    listing_id: str
    url: str | None = None
    title: str | None = None
    price_text: str | None = None
    currency: str | None = None
    condition_text: str | None = None
    seller_country: str | None = None
    seller_type: str | None = None
    production_year: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class PoliteFetcher:
    """Rate-limited, on-disk-cached HTTP client.

    The cache is the point: every page fetched is written to disk and never
    requested twice. Re-parsing during selector calibration then costs nothing
    and puts no further load on the site.
    """

    def __init__(
        self,
        cache_dir: str,
        delay: float = DEFAULT_DELAY_SECONDS,
        respect_robots: bool = True,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.cache_dir = cache_dir
        self.delay = delay
        self.respect_robots = respect_robots
        self.user_agent = user_agent
        self._last_request = 0.0
        self._robots: urllib.robotparser.RobotFileParser | None = None
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, url: str) -> str:
        digest = hashlib.sha256(url.encode()).hexdigest()[:20]
        return os.path.join(self.cache_dir, f"{digest}.html.gz")

    def _load_robots(self) -> urllib.robotparser.RobotFileParser:
        if self._robots is None:
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(urllib.parse.urljoin(BASE_URL, "/robots.txt"))
            try:
                parser.read()
            except Exception:
                # A robots.txt we cannot read is treated as "disallowed", not
                # as permission. Failing open here would be the wrong default.
                parser.disallow_all = True
            self._robots = parser
        return self._robots

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        return self._load_robots().can_fetch(self.user_agent, url)

    def get(self, url: str, force: bool = False) -> str | None:
        path = self._cache_path(url)
        if os.path.exists(path) and not force:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                return handle.read()

        if not self.allowed(url):
            raise PermissionError(
                f"robots.txt disallows {url}. Use saved pages (--from-file) instead, "
                "or pass respect_robots=False only if you have permission to crawl."
            )

        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-GB,en;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                raise RuntimeError(
                    f"{exc.code} from Chrono24 -- bot protection or rate limit. "
                    "Increase the delay, or switch to saved pages."
                ) from exc
            raise
        finally:
            self._last_request = time.monotonic()

        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(body)
        return body


class _JsonLdExtractor(HTMLParser):
    """Collect every application/ld+json payload on the page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Any] = []
        self._capturing = False
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script" and dict(attrs).get("type") == "application/ld+json":
            self._capturing = True
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._capturing:
            self._capturing = False
            text = "".join(self._buffer).strip()
            if text:
                try:
                    self.blocks.append(json.loads(text))
                except json.JSONDecodeError:
                    pass

    def handle_data(self, data: str) -> None:
        if self._capturing:
            self._buffer.append(data)


class _AttributeHarvester(HTMLParser):
    """Collect elements carrying a given attribute, plus their inner text.

    Used as the fallback when JSON-LD is absent. Deliberately attribute-driven
    rather than class-driven where possible, since data-* attributes tend to
    survive redesigns that rewrite every CSS class.
    """

    def __init__(self, attribute: str) -> None:
        super().__init__(convert_charrefs=True)
        self.attribute = attribute
        self.found: list[dict[str, Any]] = []
        self._depth = 0
        self._current: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        mapping = dict(attrs)
        if self._current is not None:
            self._depth += 1
            if tag == "a" and mapping.get("href") and not self._current.get("href"):
                self._current["href"] = mapping["href"]
            return
        if self.attribute in mapping:
            self._current = {
                "value": mapping[self.attribute],
                "attrs": mapping,
                "text": [],
                "href": mapping.get("href"),
            }
            self._depth = 0

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if self._depth == 0:
            self._current["text"] = " ".join(
                part.strip() for part in self._current["text"] if part.strip()
            )
            self.found.append(self._current)
            self._current = None
        else:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._current["text"].append(data)


def _walk(node: Any) -> Iterator[dict]:
    """Yield every dict inside an arbitrarily nested JSON-LD structure."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def parse_jsonld(html: str) -> list[RawListing]:
    """Pull schema.org Product/Offer entries out of the page."""
    extractor = _JsonLdExtractor()
    extractor.feed(html)

    out: list[RawListing] = []
    for block in extractor.blocks:
        for node in _walk(block):
            node_type = node.get("@type")
            types = node_type if isinstance(node_type, list) else [node_type]
            if not any(t in ("Product", "IndividualProduct") for t in types if t):
                continue

            offer = next(
                (o for o in _walk(node.get("offers")) if o.get("@type") in ("Offer", "AggregateOffer")),
                {},
            )
            identifier = (
                node.get("productID") or node.get("sku") or node.get("mpn")
                or offer.get("sku") or node.get("@id") or node.get("name")
            )
            if not identifier:
                continue

            price = offer.get("price") or offer.get("lowPrice")
            seller = offer.get("seller") or {}
            address = seller.get("address") if isinstance(seller, dict) else None
            country = None
            if isinstance(address, dict):
                country = address.get("addressCountry")
                if isinstance(country, dict):
                    country = country.get("name")

            out.append(
                RawListing(
                    listing_id=f"c24:{identifier}",
                    url=node.get("url") or offer.get("url"),
                    title=node.get("name"),
                    price_text=str(price) if price is not None else None,
                    currency=offer.get("priceCurrency"),
                    condition_text=_condition_from_schema(offer.get("itemCondition")),
                    seller_country=_country_code(country),
                    extras={"source_strategy": "jsonld"},
                )
            )
    return out


def _condition_from_schema(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    tail = value.rsplit("/", 1)[-1].lower()
    if "new" in tail:
        return "new"
    if "refurbished" in tail:
        return "very_good"
    if "used" in tail:
        return "good"
    return None


def _country_code(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value) == 2 and value.isalpha():
        return value.upper()
    match = _ISO_COUNTRY.search(value)
    return match.group(1) if match else None


def parse_dom(html: str) -> list[RawListing]:
    """Fallback DOM scrape driven by ``SELECTORS``. Expect to fix this."""
    harvester = _AttributeHarvester(SELECTORS["listing_id_attr"])
    harvester.feed(html)

    out: list[RawListing] = []
    for element in harvester.found:
        text = element.get("text") or ""
        href = element.get("href")
        out.append(
            RawListing(
                listing_id=f"c24:{element['value']}",
                url=urllib.parse.urljoin(BASE_URL, href) if href else None,
                title=text[:200] or None,
                price_text=_first_price(text),
                seller_country=_country_code(element["attrs"].get("data-country")),
                extras={"source_strategy": "dom", "raw_text": text[:500]},
            )
        )
    return out


_PRICE_IN_TEXT = re.compile(r"(?:€|\$|£|CHF|USD|EUR|GBP)\s?[\d.,]{3,}", re.IGNORECASE)


def _first_price(text: str) -> str | None:
    match = _PRICE_IN_TEXT.search(text)
    return match.group(0) if match else None


def parse_listing_html(html: str) -> list[RawListing]:
    """JSON-LD first, DOM fallback, deduplicated by listing id."""
    listings = parse_jsonld(html)
    if not listings:
        listings = parse_dom(html)
    seen: dict[str, RawListing] = {}
    for listing in listings:
        seen.setdefault(listing.listing_id, listing)
    return list(seen.values())


def search_url(query: str, page: int = 1, page_size: int = 60) -> str:
    """Build a search URL. Verify the shape against the real site first."""
    params = {"query": query, "dosearch": "true", "pageSize": page_size, "showpage": page}
    return f"{BASE_URL}/search/index.htm?" + urllib.parse.urlencode(params)


def iter_saved_pages(paths: list[str]) -> Iterator[tuple[str, str]]:
    """Yield (path, html) for saved pages, transparently handling .gz."""
    for path in paths:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                yield path, handle.read()
        else:
            with open(path, encoding="utf-8", errors="replace") as handle:
                yield path, handle.read()
