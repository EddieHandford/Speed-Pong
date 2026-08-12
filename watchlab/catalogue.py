"""Load reference-catalogue data into ``refs`` / ``ref_aliases``.

This module is deliberately decoupled from any one data provider. It reads a
small, documented intermediate format (JSON or CSV) and upserts it into the
schema. Adapters for real providers (thewatchapi, WatchBase, a hand-built
spreadsheet) live separately and only need to produce records in this shape --
nothing here changes when a provider's actual field names turn out to differ
from a first guess, which is the failure mode this split is meant to avoid.

Populating this table is the highest-leverage single step in the project: it
lets ``normalize.parse_title`` match listing titles against real reference
numbers instead of falling back to brand-specific regexes, which is a large
jump in match confidence for comparatively little effort.

Record shape (JSON: a list of these objects; CSV: these as column headers)::

    {
      "reference":        "126610LN",       # required, exactly as printed on papers
      "brand":            "Rolex",          # required
      "family":           "Submariner",     # optional grouping, e.g. model line
      "model_name":       "Submariner Date",# optional display name
      "retail_price":     11500,            # optional, major units (e.g. 11500.00)
      "retail_currency":  "EUR",            # optional, ISO 4217; required if retail_price set
      "production_start": 2020,             # optional, year
      "production_end":   null,             # optional, year, null/omitted if still in production
      "case_mm":          41.0,             # optional
      "aliases":          ["126610 LN", "126610-LN"]   # optional, other spellings seen in the wild
    }

In CSV form, ``aliases`` is a single column with entries separated by ``|``.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable

REQUIRED_FIELDS = ("reference", "brand")
OPTIONAL_FIELDS = (
    "family", "model_name", "retail_price", "retail_currency",
    "production_start", "production_end", "case_mm",
)


@dataclass
class CatalogueReport:
    seen: int = 0
    refs_inserted: int = 0
    refs_updated: int = 0
    aliases_added: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.seen} records: {self.refs_inserted} new refs, "
            f"{self.refs_updated} updated, {self.aliases_added} aliases added"
            + (f", {len(self.errors)} errors" if self.errors else "")
        )


def load_file(path: str) -> list[dict[str, Any]]:
    """Read a catalogue file, dispatching on extension."""
    if path.endswith(".json"):
        return load_json(path)
    if path.endswith(".csv"):
        return load_csv(path)
    raise ValueError(f"unrecognised catalogue file type: {path} (expected .json or .csv)")


def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        # Tolerate {"references": [...]} or {"data": [...]} wrapper shapes,
        # which is the common envelope for a provider's list endpoint.
        for key in ("references", "results", "data", "watches", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"{path}: expected a JSON list, or a dict wrapping one, got a plain dict")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of records")
    return data


def load_csv(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record = {k: v for k, v in row.items() if v not in (None, "")}
            if "aliases" in record:
                record["aliases"] = [a.strip() for a in record["aliases"].split("|") if a.strip()]
            out.append(record)
    return out


def _validate(record: dict[str, Any], index: int) -> str | None:
    for field_name in REQUIRED_FIELDS:
        if not record.get(field_name):
            return f"record {index}: missing required field '{field_name}'"
    if record.get("retail_price") is not None and not record.get("retail_currency"):
        return f"record {index} ({record.get('reference')}): retail_price set without retail_currency"
    return None


def upsert_catalogue(
    conn: sqlite3.Connection, records: Iterable[dict[str, Any]]
) -> CatalogueReport:
    """Insert/update reference rows and their aliases.

    A reference is looked up case-sensitively and exactly as given -- callers
    are expected to have already run values through
    :func:`watchlab.normalize.canonical_reference` if they want the same
    normalisation the listing parser uses. This function does not silently
    reshape references, since doing so here would make it a second, divergent
    copy of the normalisation logic in ``normalize.py``.
    """
    report = CatalogueReport()

    for index, record in enumerate(records):
        report.seen += 1
        error = _validate(record, index)
        if error:
            report.errors.append(error)
            continue

        reference = str(record["reference"]).strip()
        retail_cents = None
        if record.get("retail_price") is not None:
            try:
                retail_cents = round(float(record["retail_price"]) * 100)
            except (TypeError, ValueError):
                report.errors.append(f"record {index} ({reference}): non-numeric retail_price")
                continue

        existing = conn.execute(
            "SELECT reference FROM refs WHERE reference = ?", (reference,)
        ).fetchone()

        conn.execute(
            """
            INSERT INTO refs (reference, brand, family, model_name, retail_cents,
                retail_currency, production_start, production_end, case_mm)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(reference) DO UPDATE SET
                brand = excluded.brand,
                family = COALESCE(excluded.family, refs.family),
                model_name = COALESCE(excluded.model_name, refs.model_name),
                retail_cents = COALESCE(excluded.retail_cents, refs.retail_cents),
                retail_currency = COALESCE(excluded.retail_currency, refs.retail_currency),
                production_start = COALESCE(excluded.production_start, refs.production_start),
                production_end = COALESCE(excluded.production_end, refs.production_end),
                case_mm = COALESCE(excluded.case_mm, refs.case_mm)
            """,
            (
                reference, record["brand"], record.get("family"), record.get("model_name"),
                retail_cents, record.get("retail_currency"),
                record.get("production_start"), record.get("production_end"),
                record.get("case_mm"),
            ),
        )
        if existing:
            report.refs_updated += 1
        else:
            report.refs_inserted += 1

        for alias in record.get("aliases") or []:
            alias = str(alias).strip()
            if not alias or alias == reference:
                continue
            row = conn.execute(
                "SELECT reference FROM ref_aliases WHERE alias = ?", (alias,)
            ).fetchone()
            if row and row["reference"] != reference:
                report.errors.append(
                    f"record {index} ({reference}): alias '{alias}' already points to "
                    f"'{row['reference']}' -- skipped rather than overwritten"
                )
                continue
            conn.execute(
                "INSERT OR REPLACE INTO ref_aliases (alias, reference) VALUES (?, ?)",
                (alias, reference),
            )
            report.aliases_added += 1

    return report
