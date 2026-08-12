"""Turn raw scraped listings into normalised, snapshotted database rows."""

from __future__ import annotations

import datetime as _dt
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Sequence

from . import normalize
from .sources.chrono24 import RawListing
from .sources.thewatchapi import price_series_rows


@dataclass
class IngestReport:
    observed_at: str
    seen: int = 0
    new_listings: int = 0
    snapshots_written: int = 0
    delisted: int = 0
    unresolved_reference: int = 0
    low_confidence: int = 0

    def summary(self) -> str:
        return (
            f"{self.observed_at}: {self.seen} seen, {self.new_listings} new, "
            f"{self.snapshots_written} snapshots, {self.delisted} delisted, "
            f"{self.unresolved_reference} without a reference, "
            f"{self.low_confidence} low-confidence"
        )


def load_catalogue(conn: sqlite3.Connection) -> dict[str, str]:
    """Build the alias -> canonical reference map used by the normaliser."""
    catalogue: dict[str, str] = {}
    for row in conn.execute("SELECT reference FROM refs"):
        catalogue[row["reference"]] = row["reference"]
    for row in conn.execute("SELECT alias, reference FROM ref_aliases"):
        catalogue[row["alias"]] = row["reference"]
    return catalogue


def upsert_listings(
    conn: sqlite3.Connection,
    raw: Sequence[RawListing],
    observed_at: str | None = None,
    source: str = "chrono24",
    min_confidence: float = 0.35,
) -> IngestReport:
    """Insert/update listings and append one snapshot each.

    Rows whose reference could not be resolved are still stored -- they are
    the queue for improving the normaliser, and silently dropping them would
    hide exactly the coverage problem you need to see.
    """
    observed_at = observed_at or _dt.date.today().isoformat()
    catalogue = load_catalogue(conn)
    report = IngestReport(observed_at=observed_at)

    for item in raw:
        report.seen += 1
        parsed = normalize.parse_title(item.title or "", catalogue=catalogue)

        price_cents, currency = (None, item.currency)
        if item.price_text:
            price_cents, parsed_currency = normalize.parse_price(item.price_text)
            currency = item.currency or parsed_currency
        currency = currency or "EUR"

        condition = parsed.condition or _map_condition(item.condition_text)
        if not parsed.reference:
            report.unresolved_reference += 1
        if parsed.confidence < min_confidence:
            report.low_confidence += 1

        existing = conn.execute(
            "SELECT listing_id, first_price_cents FROM listings WHERE listing_id = ?",
            (item.listing_id,),
        ).fetchone()

        if existing is None:
            conn.execute(
                """
                INSERT INTO listings (listing_id, source, url, raw_title, brand, model, reference,
                    production_year, condition, has_box, has_papers, seller_country, seller_type,
                    first_seen_at, last_seen_at, first_price_cents, last_price_cents, currency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.listing_id, source, item.url, item.title or "", parsed.brand,
                    parsed.model, parsed.reference,
                    parsed.production_year or item.production_year, condition,
                    parsed.has_box, parsed.has_papers,
                    item.seller_country, item.seller_type,
                    observed_at, observed_at, price_cents, price_cents, currency,
                ),
            )
            report.new_listings += 1
        else:
            # A listing that reappears after we marked it gone was a relist or
            # a scrape gap, not a sale. Clearing delisted_at keeps days-on-
            # market honest instead of recording a phantom short sale.
            conn.execute(
                """
                UPDATE listings
                SET last_seen_at = ?, last_price_cents = ?, delisted_at = NULL,
                    reference = COALESCE(reference, ?), brand = COALESCE(brand, ?)
                WHERE listing_id = ?
                """,
                (observed_at, price_cents, parsed.reference, parsed.brand, item.listing_id),
            )

        if price_cents is not None:
            conn.execute(
                """
                INSERT OR REPLACE INTO listing_snapshots
                    (listing_id, observed_at, price_cents, currency)
                VALUES (?, ?, ?, ?)
                """,
                (item.listing_id, observed_at, price_cents, currency),
            )
            report.snapshots_written += 1

    return report


def mark_delisted(
    conn: sqlite3.Connection, seen_ids: Iterable[str], observed_at: str, source: str = "chrono24"
) -> int:
    """Flag listings from ``source`` that were live but absent from this crawl.

    Only call this after a crawl you believe was *complete* for the scope you
    are tracking. Running it on a partial crawl marks everything you did not
    happen to fetch as delisted, which silently corrupts every days-on-market
    number in the database.
    """
    seen = set(seen_ids)
    rows = conn.execute(
        "SELECT listing_id FROM listings WHERE source = ? AND delisted_at IS NULL "
        "AND first_seen_at <= ?",
        (source, observed_at),
    ).fetchall()
    stale = [r["listing_id"] for r in rows if r["listing_id"] not in seen]
    conn.executemany(
        "UPDATE listings SET delisted_at = ? WHERE listing_id = ?",
        [(observed_at, lid) for lid in stale],
    )
    return len(stale)


_CONDITION_TEXT = {
    "new": "new", "unworn": "unworn", "very good": "very_good",
    "good": "good", "fair": "fair", "poor": "poor", "incomplete": "poor",
}


def _map_condition(text: str | None) -> str | None:
    if not text:
        return None
    lowered = text.strip().lower()
    return _CONDITION_TEXT.get(lowered) or normalize.extract_condition(lowered)


def hedonic_rows(
    conn: sqlite3.Connection,
    reference: str | None = None,
    brand: str | None = None,
    fresh_only: bool = True,
) -> list[dict]:
    """Assemble observation rows for the hedonic model.

    ``fresh_only`` (the default, and the right choice for an index) keeps only
    each listing's FIRST observation. This matters more than it looks.

    A live listing's ask is anchored to whenever the seller posted it. In the
    reference simulation the median listing is ~49 days old at any given
    observation, so an index built from all live listings is measuring the
    market of seven weeks ago, smeared. Worse, the staleness is not random:
    overpriced listings are exactly the ones that fail to sell and keep
    accumulating observations, so stale asks are systematically high asks.

    A first observation, by contrast, is a fresh quote -- a seller's current
    read of the market. That makes it the right input for a price index, at
    the cost of a smaller sample.

    Set ``fresh_only=False`` to get one row per listing-month (that month's
    last ask) when you want to study seller behaviour rather than price level.
    """
    clauses, params = ["l.reference IS NOT NULL"], []
    if reference:
        clauses.append("l.reference = ?")
        params.append(reference)
    if brand:
        clauses.append("l.brand = ?")
        params.append(brand)

    if fresh_only:
        sql = f"""
            SELECT l.listing_id, l.reference, l.brand, l.condition, l.has_box, l.has_papers,
                   l.seller_country, l.seller_type, l.production_year,
                   substr(MIN(s.observed_at), 1, 7) AS period,
                   s.price_cents, s.currency, MIN(s.observed_at) AS observed_at
            FROM listings l
            JOIN listing_snapshots s ON s.listing_id = l.listing_id
            WHERE {' AND '.join(clauses)}
            GROUP BY l.listing_id
        """
    else:
        sql = f"""
            SELECT l.listing_id, l.reference, l.brand, l.condition, l.has_box, l.has_papers,
                   l.seller_country, l.seller_type, l.production_year,
                   substr(s.observed_at, 1, 7) AS period,
                   s.price_cents, s.currency, MAX(s.observed_at) AS observed_at
            FROM listings l
            JOIN listing_snapshots s ON s.listing_id = l.listing_id
            WHERE {' AND '.join(clauses)}
            GROUP BY l.listing_id, period
        """

    from . import db as _db

    out = []
    for row in conn.execute(sql, params):
        price_eur = _db.to_base(row["price_cents"], row["currency"], conn, row["observed_at"]) / 100.0
        if price_eur <= 0:
            continue
        year = row["production_year"]
        age = None
        if year:
            age = max(int(row["period"][:4]) - int(year), 0)
        out.append(
            {
                "listing_id": row["listing_id"],
                "reference": row["reference"],
                "brand": row["brand"],
                "period": row["period"],
                "price": price_eur,
                "condition": row["condition"],
                "has_box": row["has_box"],
                "has_papers": row["has_papers"],
                "seller_country": row["seller_country"],
                "seller_type": row["seller_type"],
                "age_years": age,
            }
        )
    return out


def store_provider_price_series(
    conn: sqlite3.Connection,
    provider: str,
    scope_type: str,
    scope_value: str,
    payload: dict,
    fetched_at: str | None = None,
) -> int:
    """Store a provider price-history response into ``provider_price_series``.

    Kept in its own table, never merged into ``index_points``: a provider's
    series carries no information about whether it controls for a changing
    listing mix the way the hedonic index does, so conflating the two would
    let an unaudited external number masquerade as watchlab's own estimate.
    """
    fetched_at = fetched_at or _dt.date.today().isoformat()
    rows = list(price_series_rows(provider, scope_type, scope_value, payload, fetched_at))
    conn.executemany(
        """
        INSERT OR REPLACE INTO provider_price_series
            (provider, scope_type, scope_value, observed_at, price_cents, currency, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)
