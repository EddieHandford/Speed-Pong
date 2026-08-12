"""A synthetic watch market with a known latent price path.

This exists for two reasons, and the second is the important one:

  1. It seeds a usable database so the dashboard can be run and judged before
     a single page has been scraped.
  2. It is the only way to *validate the estimator*. Real listings data has no
     ground truth -- you never learn what the constant-quality price actually
     was, so you cannot tell a working index from a broken one. Here the true
     path is generated, so ``tests/test_hedonic.py`` can assert that the
     hedonic index recovers it while a naive median does not.

The generator deliberately builds in the biases that break naive analysis:
the quality mix of live listings drifts over time, sellers mark up above the
latent value by varying amounts, and overpriced listings sit on the market
longer instead of selling.
"""

from __future__ import annotations

import datetime as _dt
import math
import random
import sqlite3
from dataclasses import dataclass

from .. import db

CONDITION_MULTIPLIER = {
    "new": 1.10,
    "unworn": 1.06,
    "very_good": 1.00,
    "good": 0.93,
    "fair": 0.82,
    "poor": 0.65,
}

COUNTRY_MULTIPLIER = {  # VAT / import / local demand effects
    "DE": 1.00, "GB": 1.03, "US": 1.05, "IT": 0.97, "JP": 0.93,
    "CH": 1.02, "HK": 0.95, "AE": 0.98, "FR": 0.99, "ES": 0.96,
}

PAPERS_PREMIUM = 0.055
BOX_PREMIUM = 0.02
AGE_DECAY_PER_YEAR = -0.004  # in log space, on top of the latent path


AUCTION_HOUSES = {  # venue premium/discount vs a neutral house, a real documented effect
    "Phillips": 1.03,
    "Christie's": 1.02,
    "Sotheby's": 1.02,
    "Bonhams": 0.98,
    "Antiquorum": 0.96,
}


@dataclass
class RefSpec:
    reference: str
    brand: str
    model: str
    base_price_eur: float
    annual_drift: float
    annual_vol: float
    listings_per_day: float
    auction_sales_per_year: float = 6.0


DEFAULT_UNIVERSE = [
    # auction_sales_per_year (trailing arg) is set per reference, not left at
    # the class default: a hot steel-sports Rolex or a grail Patek trades at
    # auction far more than a quiet Grand Seiko does, and an unrealistically
    # low rate makes the transaction-based index estimation problem look
    # easier (or, with too few observations, literally under-determined)
    # than it would be on a real reference with real auction cadence.
    RefSpec("126610LN", "Rolex", "Submariner Date", 11500, 0.06, 0.13, 1.4, 22),
    RefSpec("126710BLRO", "Rolex", "GMT-Master II Pepsi", 19500, 0.03, 0.17, 0.9, 20),
    RefSpec("116500LN", "Rolex", "Daytona", 31000, 0.01, 0.20, 0.8, 24),
    RefSpec("124300", "Rolex", "Oyster Perpetual 41", 7200, -0.04, 0.15, 1.1, 8),
    RefSpec("15500ST", "Audemars Piguet", "Royal Oak 41", 52000, -0.06, 0.24, 0.5, 18),
    RefSpec("5711/1A-010", "Patek Philippe", "Nautilus", 105000, -0.09, 0.28, 0.25, 26),
    RefSpec("310.30.42.50.01.002", "Omega", "Speedmaster Moonwatch", 6100, 0.02, 0.10, 1.3, 16),
    RefSpec("M79030N", "Tudor", "Black Bay 58", 3100, 0.04, 0.09, 1.6, 9),
    RefSpec("IW371604", "IWC", "Portugieser Chronograph", 6800, -0.01, 0.11, 0.7, 8),
    RefSpec("SBGA211", "Grand Seiko", "Snowflake", 4900, 0.05, 0.08, 0.6, 5),
]


# Same generator, a different universe: everyday sub-EUR1000 watches instead of
# the luxury validation set above. auction_sales_per_year=0 throughout --
# Phillips/Christie's/Sotheby's/Bonhams/Antiquorum don't meaningfully deal in
# this segment, so simulating an auction market here would be inventing data
# no real source will ever back up.
BUDGET_UNIVERSE = [
    RefSpec("SRPD55K1", "Seiko", "5 Sports", 190, 0.02, 0.10, 3.0, 0),
    RefSpec("SRPC91K1", "Seiko", "Turtle Save The Ocean", 480, 0.05, 0.14, 1.6, 0),
    RefSpec("GA-2100-1A1", "Casio", "G-Shock CasiOak", 95, 0.01, 0.12, 4.0, 0),
    RefSpec("BN0150-28E", "Citizen", "Promaster Diver", 210, 0.00, 0.09, 2.2, 0),
    RefSpec("T137.407.11.041.00", "Tissot", "PRX Powermatic 80", 720, 0.03, 0.10, 1.8, 0),
    RefSpec("RA-AC0001S", "Orient", "Bambino V4", 160, -0.01, 0.11, 1.4, 0),
    RefSpec("RA-AA0004L", "Orient", "Kamasu", 260, 0.02, 0.10, 1.3, 0),
    RefSpec("H69439931", "Hamilton", "Khaki Field Mechanical", 480, 0.015, 0.09, 1.5, 0),
    RefSpec("TW2T22500", "Timex", "Marlin Automatic", 180, -0.02, 0.13, 1.2, 0),
    RefSpec("C032.407.11.051.00", "Certina", "DS Action Diver", 600, 0.025, 0.11, 1.0, 0),
]


def _month(day: _dt.date) -> str:
    return f"{day.year:04d}-{day.month:02d}"


def generate(
    conn: sqlite3.Connection,
    start: _dt.date,
    end: _dt.date,
    universe: list[RefSpec] | None = None,
    observation_every_days: int = 7,
    seed: int = 20260812,
) -> dict[str, dict[str, float]]:
    """Populate the database and return the true latent path per reference.

    Returns ``{reference: {'YYYY-MM': latent_price_eur}}`` -- the ground truth
    an index estimator is supposed to recover.
    """
    rng = random.Random(seed)
    universe = universe or DEFAULT_UNIVERSE
    total_days = (end - start).days
    if total_days < 30:
        raise ValueError("need at least 30 days of simulated history")

    truth: dict[str, dict[str, float]] = {}

    for spec in universe:
        conn.execute(
            "INSERT OR REPLACE INTO refs (reference, brand, family, model_name, retail_cents, "
            "retail_currency) VALUES (?, ?, ?, ?, ?, ?)",
            (spec.reference, spec.brand, spec.model.split()[0], spec.model,
             int(spec.base_price_eur * 100 * 0.75), "EUR"),
        )

        latent = _simulate_latent(spec, total_days, rng)
        truth[spec.reference] = {}
        for offset in range(0, total_days, observation_every_days):
            truth[spec.reference][_month(start + _dt.timedelta(days=offset))] = latent[offset]

        _simulate_listings(conn, spec, latent, start, total_days, observation_every_days, rng)
        _simulate_auction_sales(conn, spec, latent, start, total_days, rng)

    return truth


def _simulate_latent(spec: RefSpec, days: int, rng: random.Random) -> list[float]:
    """Geometric Brownian motion for the constant-quality price."""
    daily_drift = spec.annual_drift / 365.0
    daily_vol = spec.annual_vol / math.sqrt(365.0)
    path = [spec.base_price_eur]
    for _ in range(days):
        shock = rng.gauss(0.0, 1.0) * daily_vol
        path.append(path[-1] * math.exp(daily_drift - 0.5 * daily_vol**2 + shock))
    return path


def _simulate_listings(
    conn: sqlite3.Connection,
    spec: RefSpec,
    latent: list[float],
    start: _dt.date,
    total_days: int,
    observe_every: int,
    rng: random.Random,
) -> None:
    counter = 0
    live: list[dict] = []
    observation_days = set(range(0, total_days, observe_every))

    for day_offset in range(total_days):
        # New listings arrive as a Poisson-ish stream, with the quality mix
        # drifting over time so that naive averages pick up a fake trend.
        arrivals = _poisson(spec.listings_per_day, rng)
        drift_phase = day_offset / max(total_days, 1)
        current_year = (start + _dt.timedelta(days=day_offset)).year
        for _ in range(arrivals):
            counter += 1
            listing = _make_listing(
                spec, latent[day_offset], drift_phase, day_offset, counter, rng, current_year
            )
            live.append(listing)

        if day_offset in observation_days:
            observed_date = start + _dt.timedelta(days=day_offset)
            still_live = []
            for listing in live:
                # Sellers grind the ask down the longer it sits.
                age_days = day_offset - listing["first_offset"]
                if age_days > 0 and rng.random() < 0.12:
                    listing["ask"] *= 1.0 - rng.uniform(0.01, 0.05)

                _write_snapshot(conn, listing, observed_date)
                listing["last_offset"] = day_offset

                # Hazard of leaving the market: cheap-relative-to-fair goes
                # fast, overpriced sits. This is what makes days-on-market a
                # real signal rather than noise.
                fair = latent[day_offset] * listing["quality_mult"]
                overprice = listing["ask"] / fair - 1.0
                hazard = 0.10 * math.exp(-6.0 * max(overprice, -0.2))
                if age_days > 3 and rng.random() < hazard:
                    listing["delisted_offset"] = day_offset
                    _finalise(conn, listing, start)
                else:
                    still_live.append(listing)
            live = still_live

    for listing in live:
        _finalise(conn, listing, start)


def _sample_condition_and_completeness(
    rng: random.Random, drift_phase: float
) -> tuple[str, int, int]:
    """Shared by listings and auction sales so both draw from one distribution.

    The share of new/unworn examples rises over the simulation (``drift_phase``
    -> 1). A naive mean over listings reads that composition drift as price
    appreciation that never happened; on auction data there's no such drift
    to speak of since each sale is dated to when it happened, not smeared by
    staleness, which is exactly the contrast the auction validation test is
    built to show.
    """
    unworn_bias = 0.25 + 0.35 * drift_phase
    if rng.random() < unworn_bias:
        condition = rng.choice(["new", "unworn"])
    else:
        condition = rng.choices(
            ["very_good", "good", "fair", "poor"], weights=[50, 30, 15, 5]
        )[0]
    has_papers = 1 if rng.random() < 0.72 else 0
    has_box = 1 if rng.random() < 0.80 else 0
    return condition, has_papers, has_box


def _make_listing(
    spec: RefSpec, latent_now: float, drift_phase: float, day_offset: int,
    counter: int, rng: random.Random, current_year: int,
) -> dict:
    condition, has_papers, has_box = _sample_condition_and_completeness(rng, drift_phase)
    country = rng.choices(list(COUNTRY_MULTIPLIER), weights=[25, 12, 18, 8, 6, 7, 5, 6, 8, 5])[0]
    seller_type = "dealer" if rng.random() < 0.68 else "private"
    age_years = rng.uniform(0, 12) if condition not in ("new", "unworn") else rng.uniform(0, 2)

    quality = (
        CONDITION_MULTIPLIER[condition]
        * COUNTRY_MULTIPLIER[country]
        * (1 + PAPERS_PREMIUM * has_papers)
        * (1 + BOX_PREMIUM * has_box)
        * math.exp(AGE_DECAY_PER_YEAR * age_years)
    )
    markup = math.exp(rng.gauss(math.log(1.08), 0.06))
    ask = latent_now * quality * markup

    production_year = current_year - int(age_years)
    return {
        "listing_id": f"syn:{spec.reference}:{counter}",
        "reference": spec.reference,
        "brand": spec.brand,
        "model": spec.model,
        "title": f"{spec.brand} {spec.model} {spec.reference} {production_year} {condition}",
        "condition": condition,
        "has_papers": has_papers,
        "has_box": has_box,
        "country": country,
        "seller_type": seller_type,
        "production_year": production_year,
        "quality_mult": quality,
        "ask": ask,
        "first_ask": ask,
        "first_offset": day_offset,
        "last_offset": day_offset,
        "delisted_offset": None,
    }


def _write_snapshot(conn: sqlite3.Connection, listing: dict, observed: _dt.date) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO listings (listing_id, source, url, raw_title, brand, model, "
        "reference, production_year, condition, has_box, has_papers, seller_country, seller_type, "
        "first_seen_at, last_seen_at, first_price_cents, currency) "
        "VALUES (?, 'synthetic', NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'EUR')",
        (
            listing["listing_id"], listing["title"], listing["brand"], listing["model"],
            listing["reference"], listing["production_year"], listing["condition"],
            listing["has_box"], listing["has_papers"], listing["country"], listing["seller_type"],
            observed.isoformat(), observed.isoformat(), int(listing["first_ask"] * 100),
        ),
    )
    conn.execute(
        "INSERT OR REPLACE INTO listing_snapshots (listing_id, observed_at, price_cents, currency) "
        "VALUES (?, ?, ?, 'EUR')",
        (listing["listing_id"], observed.isoformat(), int(listing["ask"] * 100)),
    )
    conn.execute(
        "UPDATE listings SET last_seen_at = ?, last_price_cents = ? WHERE listing_id = ?",
        (observed.isoformat(), int(listing["ask"] * 100), listing["listing_id"]),
    )


def _finalise(conn: sqlite3.Connection, listing: dict, start: _dt.date) -> None:
    if listing["delisted_offset"] is None:
        return
    gone = start + _dt.timedelta(days=listing["delisted_offset"])
    conn.execute(
        "UPDATE listings SET delisted_at = ? WHERE listing_id = ?",
        (gone.isoformat(), listing["listing_id"]),
    )


def _simulate_auction_sales(
    conn: sqlite3.Connection,
    spec: RefSpec,
    latent: list[float],
    start: _dt.date,
    total_days: int,
    rng: random.Random,
) -> None:
    """Sparse, low-bias transaction events sharing the same latent price path.

    Unlike a listing's ask, a realised sale carries no seller markup and no
    staleness: it is dated to the day it happened and priced off that day's
    latent value, plus small idiosyncratic noise (the auction equivalent of
    "unique provenance", not a systematic bias). That makes the transaction
    index a much easier estimation problem than the ask-based one --
    tests/test_auctions.py checks that the hedonic index recovers the truth
    even more tightly here than it does on listings.

    Routed through ``auctions.upsert_auction_results`` rather than a direct
    INSERT, so this is also an integration test of the real import path, not
    just of the estimator. Reference/brand/condition are supplied explicitly
    (ground truth) rather than left for ``normalize.parse_title`` to infer
    from the generated title, matching how ``_write_snapshot`` already
    bypasses the normaliser for listings: this module validates the
    estimator against known truth, not the normaliser's regex accuracy
    (that's ``tests/test_normalize.py``'s job).
    """
    from .. import auctions as _auctions

    records = []
    counter = 0
    daily_rate = spec.auction_sales_per_year / 365.0
    for day_offset in range(total_days):
        if rng.random() >= daily_rate:
            continue
        counter += 1
        drift_phase = day_offset / max(total_days, 1)
        condition, has_papers, has_box = _sample_condition_and_completeness(rng, drift_phase)
        age_years = rng.uniform(0, 12) if condition not in ("new", "unworn") else rng.uniform(0, 2)
        quality = (
            CONDITION_MULTIPLIER[condition]
            * (1 + PAPERS_PREMIUM * has_papers)
            * (1 + BOX_PREMIUM * has_box)
            * math.exp(AGE_DECAY_PER_YEAR * age_years)
        )
        house = rng.choice(list(AUCTION_HOUSES))
        noise = math.exp(rng.gauss(0.0, 0.05))
        total = latent[day_offset] * quality * AUCTION_HOUSES[house] * noise

        sale_date = start + _dt.timedelta(days=day_offset)
        production_year = sale_date.year - int(age_years)

        records.append({
            "house": house,
            "sale_date": sale_date.isoformat(),
            "lot_number": str(counter),
            "raw_title": f"{spec.brand} {spec.model} {spec.reference} {production_year} {condition}",
            "brand": spec.brand,
            "reference": spec.reference,
            "production_year": production_year,
            "condition": condition,
            "has_box": has_box,
            "has_papers": has_papers,
            "total": total,
            "currency": "EUR",
            "estimate_low": total * rng.uniform(0.75, 0.9),
            "estimate_high": total * rng.uniform(0.95, 1.05),
        })

    if records:
        _auctions.upsert_auction_results(conn, records)


def _poisson(rate: float, rng: random.Random) -> int:
    """Knuth's algorithm; rates here are small so the loop is short."""
    limit = math.exp(-rate)
    k, product = 0, 1.0
    while True:
        product *= rng.random()
        if product <= limit:
            return k
        k += 1


def seed_fx(conn: sqlite3.Connection, as_of: str) -> None:
    """Placeholder FX so non-EUR sources convert. Replace with a real feed."""
    rates = {"USD": 0.92, "GBP": 1.17, "CHF": 1.04, "JPY": 0.0061, "HKD": 0.118, "AED": 0.25}
    for currency, rate in rates.items():
        conn.execute(
            "INSERT OR REPLACE INTO fx_rates (as_of, currency, rate_to_base) VALUES (?, ?, ?)",
            (as_of, currency, rate),
        )
