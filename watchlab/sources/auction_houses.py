"""Auction-house results ingest: Phillips, Christie's, Sotheby's, Bonhams, etc.

Why this is the right source for anything historical
------------------------------------------------------
Every other source in this project is asking prices. This one is not: a
"price realised" figure on an auction house's results page is what a real
buyer actually paid for a specific lot on a specific date, published because
publishing sale results is core to how these houses establish credibility
and attract future consignments -- they have every incentive to keep this
public and none to take it down. Four independent houses means no single
point of failure the way a scraper aimed at one marketplace has.

Two ways in, and the second is the one to prefer
--------------------------------------------------
1. **Generic import** (see ``watchlab/auctions.py``): a documented JSON/CSV
   record shape, the same pattern as ``catalogue.py``. Use this if you can
   get results as structured data at all -- a research subscription export,
   a spreadsheet you compile by hand from results pages, anything. This path
   needs no HTML parsing and nothing here can go stale on it.

2. **Saved-page HTML parsing** (this module): unverified against live pages,
   same as ``sources/chrono24.py``, for the same reason -- this sandbox has
   no network access to any of these sites. It is weaker evidence here than
   it was for Chrono24, too: schema.org's ``Offer`` type models something
   currently for sale, which is a poor semantic fit for a lot that has
   already been sold and closed. Some auction sites may not mark results up
   with structured data at all. Treat ``parse_jsonld`` as worth trying, not
   as something to trust without running ``calibrate`` against a real saved
   page first.

Nothing in this module fetches a live page. It only parses HTML you already
have on disk, via ``iter_saved_pages`` (imported from ``chrono24.py``, which
is generic despite the module name).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .chrono24 import _JsonLdExtractor, _walk, iter_saved_pages  # noqa: F401  (re-exported)

_PRICE_LABELS = re.compile(r"price realised|price realized|sold for|hammer price", re.IGNORECASE)
_ESTIMATE_LABEL = re.compile(r"estimate", re.IGNORECASE)


@dataclass
class RawLot:
    """Whatever could be pulled off a saved auction results page."""

    house: str
    lot_url: str | None = None
    lot_number: str | None = None
    sale_date: str | None = None
    raw_title: str | None = None
    price_text: str | None = None
    currency: str | None = None
    estimate_low_text: str | None = None
    estimate_high_text: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def parse_jsonld(html: str, house: str) -> list[RawLot]:
    """Best-effort schema.org extraction. See module docstring for the caveat."""
    extractor = _JsonLdExtractor()
    extractor.feed(html)

    out: list[RawLot] = []
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
            price = offer.get("price") or offer.get("lowPrice")
            if price is None:
                continue

            out.append(
                RawLot(
                    house=house,
                    lot_url=node.get("url") or offer.get("url"),
                    lot_number=node.get("sku") or node.get("mpn"),
                    raw_title=node.get("name"),
                    price_text=str(price),
                    currency=offer.get("priceCurrency"),
                )
            )
    return out
