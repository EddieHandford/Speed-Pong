"""Load auction-transaction data into ``auction_results``.

This is the one table in the schema with transaction truth in it: every other
price in this project is an ask. Two paths in, mirroring the split in
``watchlab/catalogue.py``:

  * :func:`load_file` / :func:`upsert_auction_results` -- a documented,
    provider-agnostic JSON/CSV shape. This is the path to prefer: point it at
    structured results from wherever you can get them (a research export, a
    hand-compiled spreadsheet), and nothing about it depends on any auction
    house's HTML staying stable.
  * :func:`from_raw_lot` -- converts a ``sources.auction_houses.RawLot``
    (parsed from a saved HTML page, unverified -- see that module's
    docstring) into the same intermediate shape, so both paths land in one
    place.

Record shape (JSON: a list of these objects; CSV: these as column headers)::

    {
      "house":            "Phillips",         # required
      "sale_date":         "2024-05-11",       # required, ISO-8601
      "raw_title":         "Rolex Submariner ref. 126610LN, 2023",  # required
      "total":             18500,              # required, major units, the
                                                # REALISED price: hammer +
                                                # buyer's premium, not hammer alone
      "currency":          "USD",              # required
      "lot_url":           null,               # optional, used for a stable id
      "lot_number":        "142",              # optional, used for a stable id
      "brand":             null,               # optional, inferred from raw_title if absent
      "reference":         null,               # optional, inferred from raw_title if absent
      "production_year":   null,               # optional, inferred if absent
      "condition":         null,               # optional, inferred if absent
      "has_box":           null,               # optional, inferred if absent
      "has_papers":        null,               # optional, inferred if absent
      "estimate_low":      15000,              # optional, major units
      "estimate_high":     20000               # optional, major units
    }

Fields explicitly supplied always win over what the normaliser infers from
``raw_title`` -- an auction house's own condition report is more authoritative
than a regex guess at the lot title.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import normalize
from .sources.auction_houses import RawLot

REQUIRED_FIELDS = ("house", "sale_date", "raw_title", "total", "currency")

# A reference typically sees far fewer auction sales in a year than listings
# in a month, so the standard near-zero ridge (hedonic.DEFAULT_RIDGE) leaves
# the fit under-regularised: with few observations per parameter, coefficient
# estimates are individually noisy even though they are unbiased in
# expectation. A moderate ridge trades a small amount of bias for a real
# reduction in that estimation variance -- checked empirically across the
# synthetic universe (tests/test_auctions.py), not picked to fit one case.
DEFAULT_AUCTION_RIDGE = 0.02


@dataclass
class AuctionReport:
    seen: int = 0
    inserted: int = 0
    updated: int = 0
    unresolved_reference: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.seen} lots: {self.inserted} new, {self.updated} updated, "
            f"{self.unresolved_reference} without a resolved reference"
            + (f", {len(self.errors)} errors" if self.errors else "")
        )


def load_file(path: str) -> list[dict[str, Any]]:
    if path.endswith(".json"):
        return load_json(path)
    if path.endswith(".csv"):
        return load_csv(path)
    raise ValueError(f"unrecognised auction file type: {path} (expected .json or .csv)")


def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for key in ("lots", "results", "data", "auctions", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"{path}: expected a JSON list, or a dict wrapping one, got a plain dict")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of lot records")
    return data


def load_csv(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out.append({k: v for k, v in row.items() if v not in (None, "")})
    return out


def _validate(record: dict[str, Any], index: int) -> str | None:
    for field_name in REQUIRED_FIELDS:
        if record.get(field_name) in (None, ""):
            return f"record {index}: missing required field '{field_name}'"
    return None


def _stable_result_id(record: dict[str, Any]) -> str:
    """Deterministic id so re-importing the same source never duplicates a lot.

    Prefers a lot URL (most specific), falls back to house+sale_date+lot
    number, and finally to house+sale_date+title+total as a last resort --
    weaker, since two similar lots in one sale could collide, but still far
    better than a random id that breaks idempotency on every reimport.
    """
    house = str(record["house"])
    if record.get("lot_url"):
        basis = f"url:{house}:{record['lot_url']}"
    elif record.get("lot_number"):
        basis = f"lot:{house}:{record['sale_date']}:{record['lot_number']}"
    else:
        basis = f"title:{house}:{record['sale_date']}:{record['raw_title']}:{record['total']}"
    digest = hashlib.sha256(basis.encode()).hexdigest()[:24]
    return f"auction:{digest}"


def upsert_auction_results(
    conn: sqlite3.Connection,
    records: Iterable[dict[str, Any]],
    catalogue: dict[str, str] | None = None,
) -> AuctionReport:
    """Insert/update auction lots, filling in gaps from ``normalize.parse_title``.

    Lots whose reference could not be resolved are kept, not dropped -- same
    reasoning as ``ingest.upsert_listings``: a silently dropped row hides
    exactly the normaliser-coverage problem you need visibility into.
    """
    report = AuctionReport()

    for index, record in enumerate(records):
        report.seen += 1
        error = _validate(record, index)
        if error:
            report.errors.append(error)
            continue

        try:
            total_cents = round(float(record["total"]) * 100)
        except (TypeError, ValueError):
            report.errors.append(f"record {index}: non-numeric total")
            continue

        parsed = normalize.parse_title(record["raw_title"], catalogue=catalogue)
        brand = record.get("brand") or parsed.brand
        reference = record.get("reference") or parsed.reference
        if not reference:
            report.unresolved_reference += 1

        condition = record.get("condition") if record.get("condition") is not None else parsed.condition
        has_box = record.get("has_box") if record.get("has_box") is not None else parsed.has_box
        has_papers = (
            record.get("has_papers") if record.get("has_papers") is not None else parsed.has_papers
        )
        production_year = record.get("production_year") or parsed.production_year

        estimate_low_cents = estimate_high_cents = None
        if record.get("estimate_low") is not None:
            estimate_low_cents = round(float(record["estimate_low"]) * 100)
        if record.get("estimate_high") is not None:
            estimate_high_cents = round(float(record["estimate_high"]) * 100)

        result_id = _stable_result_id(record)
        existing = conn.execute(
            "SELECT result_id FROM auction_results WHERE result_id = ?", (result_id,)
        ).fetchone()

        conn.execute(
            """
            INSERT INTO auction_results
                (result_id, house, sale_date, lot_url, lot_number, raw_title, brand, model,
                 reference, production_year, condition, has_box, has_papers, total_cents,
                 currency, estimate_low_cents, estimate_high_cents)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(result_id) DO UPDATE SET
                house = excluded.house, sale_date = excluded.sale_date,
                lot_url = excluded.lot_url, lot_number = excluded.lot_number,
                raw_title = excluded.raw_title, brand = excluded.brand,
                reference = excluded.reference, production_year = excluded.production_year,
                condition = excluded.condition, has_box = excluded.has_box,
                has_papers = excluded.has_papers, total_cents = excluded.total_cents,
                currency = excluded.currency, estimate_low_cents = excluded.estimate_low_cents,
                estimate_high_cents = excluded.estimate_high_cents
            """,
            (
                result_id, record["house"], record["sale_date"], record.get("lot_url"),
                record.get("lot_number"), record["raw_title"], brand, parsed.model,
                reference, production_year, condition, has_box, has_papers, total_cents,
                record["currency"], estimate_low_cents, estimate_high_cents,
            ),
        )
        if existing:
            report.updated += 1
        else:
            report.inserted += 1

    return report


def from_raw_lot(lot: RawLot) -> dict[str, Any]:
    """Convert a saved-page ``RawLot`` into the generic import record shape."""
    total_cents, currency = (None, lot.currency)
    if lot.price_text:
        total_cents, parsed_currency = normalize.parse_price(lot.price_text)
        currency = lot.currency or parsed_currency

    record: dict[str, Any] = {
        "house": lot.house,
        "sale_date": lot.sale_date,
        "raw_title": lot.raw_title or "",
        "currency": currency or "USD",
        "lot_url": lot.lot_url,
        "lot_number": lot.lot_number,
    }
    if total_cents is not None:
        record["total"] = total_cents / 100.0
    if lot.estimate_low_text:
        cents, _ = normalize.parse_price(lot.estimate_low_text)
        if cents is not None:
            record["estimate_low"] = cents / 100.0
    if lot.estimate_high_text:
        cents, _ = normalize.parse_price(lot.estimate_high_text)
        if cents is not None:
            record["estimate_high"] = cents / 100.0
    return record


def _period_bin(sale_date: str, period_months: int) -> str:
    """Bin a sale date into a 'YYYY-MM' period label, coarsened to
    ``period_months`` wide (1 = monthly, 3 = quarterly, ...), labelled by the
    bin's first month. Stays a valid 'YYYY-MM' string either way, so nothing
    downstream (hedonic.annualised_return's month-diff, the dashboard) needs
    to know periods can be coarser than a month.
    """
    year, month = int(sale_date[:4]), int(sale_date[5:7])
    bin_start_month = ((month - 1) // period_months) * period_months + 1
    return f"{year:04d}-{bin_start_month:02d}"


def auction_hedonic_rows(
    conn: sqlite3.Connection,
    reference: str | None = None,
    brand: str | None = None,
    period_months: int = 3,
) -> list[dict[str, Any]]:
    """Assemble transaction rows for :func:`hedonic.time_dummy_index`.

    Unlike ``ingest.hedonic_rows``, there is no staleness or aggregation
    question here: each row already is one realised transaction on a known
    date. What auction data does need, that listings don't, is coarser time
    bins: a reference might see a few dozen sales a year across every house
    combined, nowhere near the volume a monthly index needs to be well
    estimated. ``period_months=3`` (quarterly, the default) matches realistic
    auction cadence; pass ``1`` for monthly on a reference that trades often
    enough to support it.
    """
    clauses, params = ["reference IS NOT NULL"], []
    if reference:
        clauses.append("reference = ?")
        params.append(reference)
    if brand:
        clauses.append("brand = ?")
        params.append(brand)

    sql = f"""
        SELECT reference, brand, house, condition, has_box, has_papers,
               production_year, sale_date, total_cents, currency
        FROM auction_results
        WHERE {' AND '.join(clauses)}
    """

    from . import db as _db

    out = []
    for row in conn.execute(sql, params):
        price = _db.to_base(row["total_cents"], row["currency"], conn, row["sale_date"]) / 100.0
        if price <= 0:
            continue
        period = _period_bin(str(row["sale_date"]), period_months)
        year = row["production_year"]
        age = max(int(period[:4]) - int(year), 0) if year else None
        out.append(
            {
                "reference": row["reference"],
                "brand": row["brand"],
                "house": row["house"],
                "period": period,
                "price": price,
                "condition": row["condition"],
                "has_box": row["has_box"],
                "has_papers": row["has_papers"],
                "age_years": age,
            }
        )
    return out
