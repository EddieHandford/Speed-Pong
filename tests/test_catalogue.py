"""Tests for the provider-agnostic catalogue importer.

These exercise the intermediate JSON/CSV format directly, without depending
on any real provider being reachable -- the point of the split in
catalogue.py is that these tests, and the loader they test, don't change when
a real adapter is written later.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watchlab import catalogue, db  # noqa: E402


def _write(tmpdir: str, name: str, content: str) -> str:
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


class TestLoadJson(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_plain_list(self):
        path = _write(self.tmp.name, "a.json", json.dumps([
            {"reference": "126610LN", "brand": "Rolex"},
        ]))
        records = catalogue.load_json(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["reference"], "126610LN")

    def test_wrapped_envelope(self):
        """A provider's list endpoint commonly wraps results in an envelope key."""
        for key in ("references", "results", "data", "watches", "items"):
            path = _write(self.tmp.name, f"{key}.json", json.dumps({
                key: [{"reference": "R1", "brand": "Omega"}],
                "meta": {"page": 1},
            }))
            records = catalogue.load_json(path)
            self.assertEqual(records, [{"reference": "R1", "brand": "Omega"}], key)

    def test_unwrappable_dict_raises(self):
        path = _write(self.tmp.name, "bad.json", json.dumps({"foo": "bar"}))
        with self.assertRaises(ValueError):
            catalogue.load_json(path)


class TestLoadCsv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_columns(self):
        path = _write(
            self.tmp.name, "cat.csv",
            "reference,brand,retail_price,retail_currency\n"
            "126610LN,Rolex,11500,EUR\n",
        )
        records = catalogue.load_csv(path)
        self.assertEqual(records, [
            {"reference": "126610LN", "brand": "Rolex", "retail_price": "11500",
             "retail_currency": "EUR"},
        ])

    def test_pipe_separated_aliases(self):
        path = _write(
            self.tmp.name, "cat.csv",
            "reference,brand,aliases\n"
            "126610LN,Rolex,126610 LN|126610-LN\n",
        )
        records = catalogue.load_csv(path)
        self.assertEqual(records[0]["aliases"], ["126610 LN", "126610-LN"])

    def test_empty_cells_are_dropped_not_kept_as_empty_string(self):
        path = _write(
            self.tmp.name, "cat.csv",
            "reference,brand,family\n"
            "126610LN,Rolex,\n",
        )
        records = catalogue.load_csv(path)
        self.assertNotIn("family", records[0])


class TestUpsertCatalogue(unittest.TestCase):
    def setUp(self):
        self.conn = db.connect(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_insert_then_update(self):
        report1 = catalogue.upsert_catalogue(self.conn, [
            {"reference": "126610LN", "brand": "Rolex", "case_mm": 41.0},
        ])
        self.assertEqual((report1.refs_inserted, report1.refs_updated), (1, 0))

        report2 = catalogue.upsert_catalogue(self.conn, [
            {"reference": "126610LN", "brand": "Rolex", "model_name": "Submariner Date"},
        ])
        self.assertEqual((report2.refs_inserted, report2.refs_updated), (0, 1))

        row = self.conn.execute(
            "SELECT model_name, case_mm FROM refs WHERE reference = '126610LN'"
        ).fetchone()
        self.assertEqual(row["model_name"], "Submariner Date")
        self.assertEqual(row["case_mm"], 41.0, "an update must not blank fields it didn't supply")

    def test_missing_required_field_is_reported_not_raised(self):
        report = catalogue.upsert_catalogue(self.conn, [{"brand": "Rolex"}])
        self.assertEqual(report.refs_inserted, 0)
        self.assertEqual(len(report.errors), 1)
        self.assertIn("reference", report.errors[0])

    def test_retail_price_without_currency_is_rejected(self):
        report = catalogue.upsert_catalogue(self.conn, [
            {"reference": "R1", "brand": "Omega", "retail_price": 5000},
        ])
        self.assertEqual(report.refs_inserted, 0)
        self.assertTrue(any("retail_currency" in e for e in report.errors))

    def test_retail_price_converted_to_minor_units(self):
        catalogue.upsert_catalogue(self.conn, [
            {"reference": "R1", "brand": "Omega", "retail_price": 6199.5, "retail_currency": "EUR"},
        ])
        row = self.conn.execute(
            "SELECT retail_cents, retail_currency FROM refs WHERE reference = 'R1'"
        ).fetchone()
        self.assertEqual(row["retail_cents"], 619950)
        self.assertEqual(row["retail_currency"], "EUR")

    def test_aliases_are_added(self):
        catalogue.upsert_catalogue(self.conn, [
            {"reference": "126610LN", "brand": "Rolex", "aliases": ["126610 LN", "126610-LN"]},
        ])
        rows = self.conn.execute(
            "SELECT alias FROM ref_aliases WHERE reference = '126610LN' ORDER BY alias"
        ).fetchall()
        self.assertEqual([r["alias"] for r in rows], ["126610 LN", "126610-LN"])

    def test_conflicting_alias_is_skipped_not_overwritten(self):
        """Two references claiming the same alias must not silently clobber each other."""
        catalogue.upsert_catalogue(self.conn, [
            {"reference": "R1", "brand": "Rolex", "aliases": ["Sub"]},
        ])
        report = catalogue.upsert_catalogue(self.conn, [
            {"reference": "R2", "brand": "Rolex", "aliases": ["Sub"]},
        ])
        self.assertEqual(report.aliases_added, 0)
        self.assertTrue(any("Sub" in e for e in report.errors))
        owner = self.conn.execute(
            "SELECT reference FROM ref_aliases WHERE alias = 'Sub'"
        ).fetchone()["reference"]
        self.assertEqual(owner, "R1")

    def test_alias_equal_to_its_own_reference_is_skipped_silently(self):
        report = catalogue.upsert_catalogue(self.conn, [
            {"reference": "R1", "brand": "Rolex", "aliases": ["R1"]},
        ])
        self.assertEqual(report.aliases_added, 0)
        self.assertEqual(report.errors, [])

    def test_populated_catalogue_feeds_normalize(self):
        """The point of this module: a loaded catalogue improves reference matching."""
        from watchlab import ingest, normalize

        catalogue.upsert_catalogue(self.conn, [
            {"reference": "15500ST.OO.1220ST.01", "brand": "Audemars Piguet",
             "aliases": ["15500ST"]},
        ])
        table = ingest.load_catalogue(self.conn)
        parsed = normalize.parse_title(
            "AP Royal Oak 15500ST full set 2021", catalogue=table
        )
        self.assertEqual(parsed.reference, "15500ST.OO.1220ST.01")
        # catalogue hit (0.7) + brand (0.15) + year (0.075); no condition phrase
        # in this title, so confidence falls short of a perfect score.
        self.assertGreaterEqual(parsed.confidence, 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
