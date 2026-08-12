"""SQLite storage layer.

Everything is stored in minor units (cents) as INTEGER to avoid float drift on
money, and every price carries its own currency. Conversion to the base
currency happens at read time via ``fx_rates`` so that re-running with better
FX data does not require re-scraping.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

DEFAULT_DB_PATH = os.environ.get(
    "WATCHLAB_DB", os.path.join(os.path.dirname(os.path.dirname(__file__)), "watchlab.sqlite3")
)

BASE_CURRENCY = "EUR"

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- One row per listing we have ever seen, carrying its normalised identity and
-- lifecycle. Prices here are the first/last observed ask, denormalised from
-- listing_snapshots for cheap querying.
CREATE TABLE IF NOT EXISTS listings (
    listing_id        TEXT PRIMARY KEY,   -- source-scoped, e.g. "c24:12345678"
    source            TEXT NOT NULL,
    url               TEXT,
    raw_title         TEXT NOT NULL,
    brand             TEXT,
    model             TEXT,
    reference         TEXT,               -- canonical reference number
    production_year   INTEGER,
    condition         TEXT,               -- see normalize.CONDITIONS
    has_box           INTEGER,            -- 0/1/NULL(unknown)
    has_papers        INTEGER,
    seller_country    TEXT,               -- ISO-3166 alpha-2
    seller_type       TEXT,               -- 'dealer' | 'private' | NULL
    first_seen_at     TEXT NOT NULL,      -- ISO-8601 date
    last_seen_at      TEXT NOT NULL,
    delisted_at       TEXT,               -- first date we observed it absent
    first_price_cents INTEGER,
    last_price_cents  INTEGER,
    currency          TEXT
);

CREATE INDEX IF NOT EXISTS idx_listings_reference ON listings(reference);
CREATE INDEX IF NOT EXISTS idx_listings_brand ON listings(brand);
CREATE INDEX IF NOT EXISTS idx_listings_delisted ON listings(delisted_at);

-- One row per listing per observation. This is the append-only fact table;
-- everything else can be rebuilt from it.
CREATE TABLE IF NOT EXISTS listing_snapshots (
    listing_id     TEXT NOT NULL REFERENCES listings(listing_id) ON DELETE CASCADE,
    observed_at    TEXT NOT NULL,         -- ISO-8601 date
    price_cents    INTEGER NOT NULL,
    currency       TEXT NOT NULL,
    shipping_cents INTEGER,
    PRIMARY KEY (listing_id, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_observed ON listing_snapshots(observed_at);

-- Actual clearing prices. This is the only table with transaction truth in it;
-- treat it as the ground truth for any return series.
CREATE TABLE IF NOT EXISTS auction_results (
    result_id       TEXT PRIMARY KEY,
    house           TEXT NOT NULL,
    sale_date       TEXT NOT NULL,
    lot_url         TEXT,
    lot_number      TEXT,
    raw_title       TEXT NOT NULL,
    brand           TEXT,
    model           TEXT,
    reference       TEXT,
    production_year INTEGER,
    condition       TEXT,
    has_box         INTEGER,
    has_papers      INTEGER,
    total_cents     INTEGER NOT NULL,     -- price realised: hammer + buyer's premium
    currency        TEXT NOT NULL,
    estimate_low_cents  INTEGER,          -- pre-sale low estimate, same currency
    estimate_high_cents INTEGER
);

CREATE INDEX IF NOT EXISTS idx_auction_reference ON auction_results(reference);
CREATE INDEX IF NOT EXISTS idx_auction_date ON auction_results(sale_date);

-- Canonical reference catalogue. Hand-curated; the normaliser resolves messy
-- listing titles onto these keys.
CREATE TABLE IF NOT EXISTS refs (
    reference        TEXT PRIMARY KEY,
    brand            TEXT NOT NULL,
    family           TEXT,
    model_name       TEXT,
    retail_cents     INTEGER,
    retail_currency  TEXT,
    production_start INTEGER,
    production_end   INTEGER,
    case_mm          REAL
);

-- Alternate spellings/aliases seen in the wild -> canonical reference.
CREATE TABLE IF NOT EXISTS ref_aliases (
    alias     TEXT PRIMARY KEY,
    reference TEXT NOT NULL REFERENCES refs(reference) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fx_rates (
    as_of         TEXT NOT NULL,
    currency      TEXT NOT NULL,
    rate_to_base  REAL NOT NULL,          -- multiply by this to get BASE_CURRENCY
    PRIMARY KEY (as_of, currency)
);

-- Cache of computed hedonic index points so the dashboard does not refit on
-- every page load.
CREATE TABLE IF NOT EXISTS index_points (
    scope       TEXT NOT NULL,            -- e.g. 'ref:126610LN' or 'brand:Rolex'
    period      TEXT NOT NULL,            -- 'YYYY-MM'
    index_value REAL NOT NULL,            -- base period = 100
    n_obs       INTEGER NOT NULL,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (scope, period)
);

-- Pre-aggregated price series pulled from an external data provider (e.g.
-- thewatchapi's brand/model/reference price-history endpoints).
--
-- This is deliberately NOT the same table as index_points. index_points is
-- computed by hedonic.py from watchlab's own listing data, with condition,
-- papers, box, age, country and seller type held constant, specifically to
-- avoid measuring a changing listing mix as price movement. A provider's
-- price series carries no visibility into whether it does anything similar;
-- treat it as an external cross-check, never as a substitute for the index.
CREATE TABLE IF NOT EXISTS provider_price_series (
    provider    TEXT NOT NULL,          -- e.g. 'thewatchapi'
    scope_type  TEXT NOT NULL,          -- 'brand' | 'model' | 'reference'
    scope_value TEXT NOT NULL,
    observed_at TEXT NOT NULL,          -- ISO-8601 date
    price_cents INTEGER NOT NULL,
    currency    TEXT NOT NULL,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (provider, scope_type, scope_value, observed_at)
);

-- Provenance for every fetch, so a crawl can be resumed and audited and so we
-- never re-request a page we already have.
CREATE TABLE IF NOT EXISTS fetch_log (
    url         TEXT NOT NULL,
    fetched_at  TEXT NOT NULL,
    status      INTEGER,
    bytes       INTEGER,
    cache_path  TEXT,
    PRIMARY KEY (url, fetched_at)
);
"""


def connect(path: str | None = None) -> sqlite3.Connection:
    """Open (and if needed initialise) the database."""
    conn = sqlite3.connect(path or DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def session(path: str | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def to_base(cents: int, currency: str, conn: sqlite3.Connection, as_of: str | None = None) -> int:
    """Convert a money amount into BASE_CURRENCY minor units.

    Uses the most recent rate at or before ``as_of``; falls back to the most
    recent rate overall. Unknown currencies are returned unchanged rather than
    dropped, so a missing FX row degrades accuracy instead of losing data.
    """
    if currency == BASE_CURRENCY:
        return cents
    if as_of:
        row = conn.execute(
            "SELECT rate_to_base FROM fx_rates WHERE currency = ? AND as_of <= ? "
            "ORDER BY as_of DESC LIMIT 1",
            (currency, as_of),
        ).fetchone()
        if row:
            return round(cents * row["rate_to_base"])
    row = conn.execute(
        "SELECT rate_to_base FROM fx_rates WHERE currency = ? ORDER BY as_of DESC LIMIT 1",
        (currency,),
    ).fetchone()
    return round(cents * row["rate_to_base"]) if row else cents
