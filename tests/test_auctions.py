"""Auction-results importer and transaction-based index tests.

The headline test is ``test_auction_hedonic_beats_naive_auction_median`` --
it reuses the known-truth simulator from tests/test_hedonic.py, but on
auction transactions. This is deliberately the SAME comparison
tests/test_hedonic.py makes for listings (hedonic vs naive median, same data
source, same truth), not a cross-source race between auctions and listings.

An earlier version of this file asserted that the auction-based index beats
the listings-based index outright. That did not hold up: checked empirically
across the synthetic universe and several seeds, auctions beat listings on
RMSE only about half the time. A reference sees far fewer auction sales in a
year than listings in a month, so while the auction index is genuinely less
biased (no seller markup, no staleness), it is also noisier from sample size
alone -- and in a small-sample regime, lower bias does not guarantee lower
RMSE against the truth. Asserting an outright win would have been
overclaiming, so this file doesn't. What holds reliably is the same
within-source comparison as the listings test: the hedonic correction is
still doing real work against a listing mix that drifts over time, on
auction data just as it does on listings.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import statistics
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watchlab import auctions, db, hedonic  # noqa: E402
from watchlab.sources import synthetic  # noqa: E402
from watchlab.sources.auction_houses import RawLot  # noqa: E402


def _write(tmpdir, name, content):
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


class TestLoadFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_json_plain_list(self):
        path = _write(self.tmp.name, "a.json", json.dumps([
            {"house": "Phillips", "sale_date": "2024-05-11", "raw_title": "Rolex 126610LN",
             "total": 18500, "currency": "USD"},
        ]))
        records = auctions.load_json(path)
        self.assertEqual(len(records), 1)

    def test_json_wrapped_envelope(self):
        for key in ("lots", "results", "data", "auctions", "items"):
            path = _write(self.tmp.name, f"{key}.json", json.dumps({key: [{"house": "X"}]}))
            self.assertEqual(auctions.load_json(path), [{"house": "X"}])

    def test_csv_columns(self):
        path = _write(
            self.tmp.name, "lots.csv",
            "house,sale_date,raw_title,total,currency\n"
            "Phillips,2024-05-11,Rolex 126610LN,18500,USD\n",
        )
        records = auctions.load_csv(path)
        self.assertEqual(records[0]["house"], "Phillips")


class TestUpsertAuctionResults(unittest.TestCase):
    def setUp(self):
        self.conn = db.connect(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_insert_basic_lot(self):
        report = auctions.upsert_auction_results(self.conn, [
            {"house": "Phillips", "sale_date": "2024-05-11",
             "raw_title": "Rolex Submariner 126610LN 2023 unworn full set",
             "total": 18500, "currency": "USD"},
        ])
        self.assertEqual(report.inserted, 1)
        row = self.conn.execute("SELECT * FROM auction_results").fetchone()
        self.assertEqual(row["total_cents"], 1850000)
        self.assertEqual(row["reference"], "126610LN")
        self.assertEqual(row["brand"], "Rolex")

    def test_missing_required_field_reported_not_raised(self):
        report = auctions.upsert_auction_results(self.conn, [{"house": "Phillips"}])
        self.assertEqual(report.inserted, 0)
        self.assertEqual(len(report.errors), 1)

    def test_explicit_fields_win_over_inferred(self):
        """An auction house's own condition report beats a title-regex guess."""
        report = auctions.upsert_auction_results(self.conn, [
            {"house": "Phillips", "sale_date": "2024-05-11",
             "raw_title": "Rolex Submariner 126610LN unworn", "total": 18500, "currency": "USD",
             "condition": "very_good", "reference": "OVERRIDE"},
        ])
        self.assertEqual(report.inserted, 1)
        row = self.conn.execute("SELECT condition, reference FROM auction_results").fetchone()
        self.assertEqual(row["condition"], "very_good")
        self.assertEqual(row["reference"], "OVERRIDE")

    def test_unresolved_reference_is_kept_and_counted(self):
        report = auctions.upsert_auction_results(self.conn, [
            {"house": "Phillips", "sale_date": "2024-05-11",
             "raw_title": "A lovely vintage timepiece", "total": 500, "currency": "USD"},
        ])
        self.assertEqual(report.unresolved_reference, 1)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM auction_results").fetchone()[0], 1,
        )

    def test_reimport_is_idempotent_via_lot_url(self):
        record = {"house": "Phillips", "sale_date": "2024-05-11", "raw_title": "Rolex 126610LN",
                  "total": 18500, "currency": "USD", "lot_url": "https://phillips.com/lot/1"}
        auctions.upsert_auction_results(self.conn, [record])
        record["total"] = 19000  # simulate a corrected re-publish of the same lot
        report = auctions.upsert_auction_results(self.conn, [record])
        self.assertEqual((report.inserted, report.updated), (0, 1))
        count = self.conn.execute("SELECT COUNT(*) FROM auction_results").fetchone()[0]
        self.assertEqual(count, 1, "re-importing the same lot must not duplicate it")
        total = self.conn.execute("SELECT total_cents FROM auction_results").fetchone()[0]
        self.assertEqual(total, 1900000, "the update must actually take the new value")

    def test_two_lots_same_sale_different_lot_numbers_stay_distinct(self):
        auctions.upsert_auction_results(self.conn, [
            {"house": "Phillips", "sale_date": "2024-05-11", "lot_number": "101",
             "raw_title": "Rolex 126610LN", "total": 18500, "currency": "USD"},
            {"house": "Phillips", "sale_date": "2024-05-11", "lot_number": "102",
             "raw_title": "Rolex 116500LN", "total": 32000, "currency": "USD"},
        ])
        count = self.conn.execute("SELECT COUNT(*) FROM auction_results").fetchone()[0]
        self.assertEqual(count, 2)

    def test_non_numeric_total_reported(self):
        report = auctions.upsert_auction_results(self.conn, [
            {"house": "Phillips", "sale_date": "2024-05-11", "raw_title": "Rolex 126610LN",
             "total": "POA", "currency": "USD"},
        ])
        self.assertEqual(report.inserted, 0)
        self.assertTrue(any("non-numeric" in e for e in report.errors))

    def test_estimates_stored_in_cents(self):
        auctions.upsert_auction_results(self.conn, [
            {"house": "Phillips", "sale_date": "2024-05-11", "raw_title": "Rolex 126610LN",
             "total": 18500, "currency": "USD", "estimate_low": 15000, "estimate_high": 20000},
        ])
        row = self.conn.execute(
            "SELECT estimate_low_cents, estimate_high_cents FROM auction_results"
        ).fetchone()
        self.assertEqual((row["estimate_low_cents"], row["estimate_high_cents"]), (1500000, 2000000))


class TestFromRawLot(unittest.TestCase):
    def test_price_text_parsed(self):
        lot = RawLot(
            house="Christie's", sale_date="2024-06-01", raw_title="Rolex Daytona 116520",
            price_text="$54,120", estimate_low_text="$40,000", estimate_high_text="$60,000",
        )
        record = auctions.from_raw_lot(lot)
        self.assertEqual(record["total"], 54120.0)
        self.assertEqual(record["currency"], "USD")
        self.assertEqual(record["estimate_low"], 40000.0)
        self.assertEqual(record["estimate_high"], 60000.0)

    def test_missing_price_text_omits_total(self):
        lot = RawLot(house="Christie's", sale_date="2024-06-01", raw_title="Rolex Daytona")
        record = auctions.from_raw_lot(lot)
        self.assertNotIn("total", record)


class TestAuctionHedonicRows(unittest.TestCase):
    def setUp(self):
        self.conn = db.connect(":memory:")
        auctions.upsert_auction_results(self.conn, [
            {"house": "Phillips", "sale_date": "2024-05-11", "reference": "126610LN",
             "brand": "Rolex", "raw_title": "Rolex 126610LN", "total": 18500, "currency": "USD",
             "condition": "unworn", "has_box": 1, "has_papers": 1, "production_year": 2022},
            {"house": "Christie's", "sale_date": "2024-06-01", "reference": "126610LN",
             "brand": "Rolex", "raw_title": "Rolex 126610LN", "total": 19000, "currency": "USD",
             "condition": "very_good", "has_box": 0, "has_papers": 1, "production_year": 2020},
        ])
        self.conn.execute(
            "INSERT INTO fx_rates (as_of, currency, rate_to_base) VALUES ('2024-01-01','USD',0.90)"
        )

    def tearDown(self):
        self.conn.close()

    def test_rows_shaped_for_hedonic(self):
        rows = auctions.auction_hedonic_rows(self.conn, reference="126610LN")
        self.assertEqual(len(rows), 2)
        row = rows[0]
        for key in ("reference", "brand", "house", "period", "price", "condition",
                    "has_box", "has_papers", "age_years"):
            self.assertIn(key, row)

    def test_currency_converted_to_eur(self):
        rows = auctions.auction_hedonic_rows(self.conn, reference="126610LN")
        self.assertAlmostEqual(rows[0]["price"], 18500 * 0.90, places=2)

    def test_age_computed_from_production_year(self):
        # Default period_months=3 bins June into the Q2-starting label '2024-04'.
        rows = auctions.auction_hedonic_rows(self.conn, reference="126610LN")
        by_period = {r["period"]: r for r in rows}
        self.assertEqual(by_period["2024-04"]["age_years"], 4)  # 2024 - 2020

    def test_monthly_binning_keeps_the_actual_month(self):
        rows = auctions.auction_hedonic_rows(self.conn, reference="126610LN", period_months=1)
        periods = {r["period"] for r in rows}
        self.assertEqual(periods, {"2024-05", "2024-06"})


class TestAuctionFeatures(unittest.TestCase):
    def test_house_replaces_seller_fields(self):
        names = {f.name for f in hedonic.auction_features()}
        self.assertIn("house", names)
        self.assertNotIn("seller_country", names)
        self.assertNotIn("seller_type", names)


class TestAuctionHedonicAgainstGroundTruth(unittest.TestCase):
    """Same known-truth simulator as test_hedonic.py, applied to auction sales."""

    @classmethod
    def setUpClass(cls):
        cls.conn = db.connect(":memory:")
        cls.start = _dt.date(2024, 1, 1)
        cls.end = _dt.date(2026, 1, 1)
        cls.truth = synthetic.generate(cls.conn, cls.start, cls.end, seed=11)
        cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    @staticmethod
    def _rmse_vs_truth(points, truth):
        periods = [p.period for p in points if p.period in truth]
        base_truth = truth[periods[0]]
        true_idx = {p: 100.0 * truth[p] / base_truth for p in periods}
        got = {p.period: p.value for p in points if p.period in truth}
        return math.sqrt(sum((got[p] - true_idx[p]) ** 2 for p in periods) / len(periods))

    def test_auction_sales_were_generated(self):
        n = self.conn.execute("SELECT COUNT(*) FROM auction_results").fetchone()[0]
        self.assertGreater(n, 100)

    def test_auction_hedonic_tracks_truth(self):
        reference = "126610LN"
        rows = auctions.auction_hedonic_rows(self.conn, reference=reference)
        self.assertGreater(len(rows), 20)

        points, fit = hedonic.time_dummy_index(
            rows, hedonic.auction_features(), with_ci=False, min_obs_per_period=2,
            ridge=auctions.DEFAULT_AUCTION_RIDGE,
        )
        self.assertGreater(len(points), 5)
        error = self._rmse_vs_truth(points, self.truth[reference])
        self.assertLess(error, 4.0, f"auction hedonic RMSE {error:.2f} too high")

    def test_auction_hedonic_beats_naive_auction_median(self):
        """The reliable comparison: same data source, hedonic vs naive median.

        This is the auction-data version of test_hedonic.py's headline
        assertion. It does NOT compare across auctions and listings -- see
        the module docstring for why that comparison isn't a safe one to
        assert on.
        """
        reference = "126610LN"
        truth = self.truth[reference]
        rows = auctions.auction_hedonic_rows(self.conn, reference=reference)

        points, _ = hedonic.time_dummy_index(
            rows, hedonic.auction_features(), with_ci=False, min_obs_per_period=2,
            ridge=auctions.DEFAULT_AUCTION_RIDGE,
        )
        hedonic_error = self._rmse_vs_truth(points, truth)

        by_period: dict[str, list[float]] = {}
        for row in rows:
            by_period.setdefault(row["period"], []).append(row["price"])
        periods = sorted(p for p in by_period if p in truth)
        base_median = statistics.median(by_period[periods[0]])
        naive_points = [
            hedonic.IndexPoint(period=p, value=100.0 * statistics.median(by_period[p]) / base_median,
                               n_obs=len(by_period[p]))
            for p in periods
        ]
        naive_error = self._rmse_vs_truth(naive_points, truth)

        self.assertLess(
            hedonic_error, naive_error,
            f"auction hedonic RMSE {hedonic_error:.2f} should beat "
            f"naive-median-of-auctions RMSE {naive_error:.2f}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
