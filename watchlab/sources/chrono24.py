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

The parsers below were originally written without live access to the site, so
the CSS class names in ``SELECTORS`` were unverified guesses. ``parse_jsonld``
has since been fixed against a real saved search-results page (2026-08-12):
Chrono24's ``@graph`` carries a single ``AggregateOffer`` node with
``priceCurrency`` at the node level and a bare ``offers`` array -- no
Product/IndividualProduct wrapper, and no per-offer id, so the listing id is
parsed out of each offer's URL (Chrono24 URLs always end ``--id<digits>.htm``).
Notably absent from that JSON-LD: condition, seller country, seller type --
none of it is there. Some of it often recovers anyway because the offer
``name`` is the seller's free-text title (e.g. "... Good Condition Box",
"... Full set"), which ``normalize.parse_title`` already knows how to read;
what it can't recover, ``normalize`` correctly leaves ``None`` rather than
guessing. ``parse_dom`` (``SELECTORS``) remains an unverified fallback for
whatever page shape doesn't carry this JSON-LD.
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

# listing_id_attr verified against a real saved search-results page
# (2026-08-12): each listing card is a
# `<div class="js-listing-item-container ... wt-search-result" data-search-hash="...">`.
# The card's own text (title, price, seller badge, location) all fall inside
# it, so _AttributeHarvester's flattened text is enough for parse_dom's
# regex-based price extraction even without dedicated per-field selectors.
SELECTORS = {
    "listing_id_attr": "data-search-hash",
}

_ISO_COUNTRY = re.compile(r"\b([A-Z]{2})\b")
_LISTING_ID_IN_URL = re.compile(r"--id(\d+)\.htm")


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
    """Pull the AggregateOffer's per-listing Offer entries out of a page.

    Chrono24 search-results pages carry one ``AggregateOffer`` node per page,
    ``priceCurrency`` set once at that node (not per offer), and a bare
    ``offers`` array with no Product/IndividualProduct wrapper and no
    per-offer identifier -- so the listing id comes out of each offer's URL.
    """
    extractor = _JsonLdExtractor()
    extractor.feed(html)

    out: list[RawListing] = []
    for block in extractor.blocks:
        for node in _walk(block):
            if node.get("@type") != "AggregateOffer":
                continue
            currency = node.get("priceCurrency")
            for offer in node.get("offers") or []:
                if not isinstance(offer, dict) or offer.get("@type") != "Offer":
                    continue
                url = offer.get("url")
                match = _LISTING_ID_IN_URL.search(url or "")
                if not match:
                    continue
                price = offer.get("price")
                out.append(
                    RawListing(
                        listing_id=f"c24:{match.group(1)}",
                        url=url,
                        title=offer.get("name"),
                        price_text=str(price) if price is not None else None,
                        currency=currency,
                        extras={"source_strategy": "jsonld"},
                    )
                )
    return out


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
