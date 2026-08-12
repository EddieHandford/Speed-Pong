"""Microstructure metric tests, built on hand-constructed listing histories."""

from __future__ import annotations

import datetime as _dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watchlab import db, ingest, metrics  # noqa: E402
from watchlab.sources.chrono24 import RawListing  # noqa: E402


def _add(conn, listing_id, reference, first, prices, delisted=None, currency="EUR"):
    """Insert one listing with a weekly price series starting at ``first``."""
    start = _dt.date.fromisoformat(first)
    conn.execute(
        "INSERT INTO listings (listing_id, source, raw_title, brand, reference, condition, "
        "first_seen_at, last_seen_at, delisted_at, first_price_cents, currency) "
        "VALUES (?, 'test', ?, 'Rolex', ?, 'good', ?, ?, ?, ?, ?)",
        (listing_id, f"Rolex {reference}", reference, first,
         (start + _dt.timedelta(days=7 * (len(prices) - 1))).isoformat(),
         delisted, int(prices[0] * 100), currency),
    )
    for offset, price in enumerate(prices):
        day = (start + _dt.timedelta(days=7 * offset)).isoformat()
        conn.execute(
            "INSERT INTO listing_snapshots (listing_id, observed_at, price_cents, currency) "
            "VALUES (?, ?, ?, ?)",
            (listing_id, day, int(price * 100), currency),
        )


class TestStatisticalHelpers(unittest.TestCase):
    def test_dispersion_is_scale_free(self):
        small = [90.0, 95.0, 100.0, 105.0, 110.0]
        large = [v * 1000 for v in small]
        self.assertAlmostEqual(metrics.dispersion(small), metrics.dispersion(large), places=9)

    def test_dispersion_needs_four_points(self):
        self.assertIsNone(metrics.dispersion([1.0, 2.0, 3.0]))

    def test_quantile_interpolates(self):
        self.assertAlmostEqual(metrics._quantile([0.0, 10.0], 0.5), 5.0)
        self.assertAlmostEqual(metrics._quantile([0.0, 10.0, 20.0, 30.0], 0.25), 7.5)


class TestCostModel(unittest.TestCase):
    def test_hurdle_falls_as_price_rises(self):
        """Fixed shipping matters far more on a cheap watch than a costly one."""
        costs = metrics.CostModel()
        self.assertGreater(costs.hurdle_rate(3000, 1.0), costs.hurdle_rate(80000, 1.0))

    def test_hurdle_exceeds_round_trip_percentage(self):
        costs = metrics.CostModel()
        self.assertGreater(costs.hurdle_rate(10000, 1.0), costs.round_trip_pct())

    def test_a_ten_percent_gain_does_not_survive_costs(self):
        """The headline caveat, as an assertion."""
        costs = metrics.CostModel()
        net = metrics.net_of_costs(0.10, 10000, 1.0, costs)
        self.assertLess(net, 0.0, f"10% gross should be negative net, got {net:.1%}")

    def test_long_holds_amortise_the_friction(self):
        costs = metrics.CostModel()
        one_year = metrics.net_of_costs(0.10, 10000, 1.0, costs)
        five_years = metrics.net_of_costs(0.10, 10000, 5.0, costs)
        self.assertGreater(five_years, one_year)


class TestListingDerivedMetrics(unittest.TestCase):
    def setUp(self):
        self.conn = db.connect(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_days_on_market_measures_lifespan(self):
        _add(self.conn, "a", "R1", "2026-01-01", [100.0] * 5, delisted="2026-02-01")
        _add(self.conn, "b", "R1", "2026-01-01", [100.0] * 3, delisted="2026-01-15")
        doms = metrics.days_on_market(self.conn, "R1", as_of="2026-03-01", window_days=180)
        self.assertEqual(sorted(doms), [14.0, 31.0])

    def test_live_listings_are_excluded_from_dom(self):
        _add(self.conn, "live", "R1", "2026-01-01", [100.0] * 5)
        self.assertEqual(metrics.days_on_market(self.conn, "R1", as_of="2026-03-01"), [])

    def test_cut_rate_and_depth(self):
        _add(self.conn, "cut", "R2", "2026-01-01", [100.0, 98.0, 90.0])
        _add(self.conn, "flat", "R2", "2026-01-01", [100.0, 100.0, 100.0])
        rate, depth = metrics.price_cuts(self.conn, "R2", as_of="2026-03-01", window_days=180)
        self.assertAlmostEqual(rate, 0.5)
        self.assertAlmostEqual(depth, 0.10, places=6)

    def test_cut_measured_against_peak_not_first_price(self):
        """A seller who lists low, raises, then cuts back has still cut."""
        _add(self.conn, "wobble", "R3", "2026-01-01", [100.0, 120.0, 108.0])
        rate, depth = metrics.price_cuts(self.conn, "R3", as_of="2026-03-01", window_days=180)
        self.assertAlmostEqual(rate, 1.0)
        self.assertAlmostEqual(depth, 0.10, places=6)

    def test_live_asks_use_latest_observation(self):
        _add(self.conn, "x", "R4", "2026-01-01", [100.0, 90.0, 80.0])
        asks = metrics.live_asks_eur(self.conn, "R4", as_of="2026-03-01")
        self.assertEqual(asks, [80.0])

    def test_live_asks_respect_as_of(self):
        """Asking about the past must not leak later prices."""
        _add(self.conn, "x", "R5", "2026-01-01", [100.0, 90.0, 80.0])
        self.assertEqual(metrics.live_asks_eur(self.conn, "R5", as_of="2026-01-08"), [90.0])

    def test_currency_conversion_applied(self):
        self.conn.execute(
            "INSERT INTO fx_rates (as_of, currency, rate_to_base) VALUES ('2026-01-01','USD',0.90)"
        )
        _add(self.conn, "usd", "R6", "2026-01-05", [1000.0], currency="USD")
        self.assertEqual(metrics.live_asks_eur(self.conn, "R6", as_of="2026-02-01"), [900.0])

    def test_churn_counts_both_directions(self):
        # 'gone' is first seen before the window opens, so it contributes only
        # to delistings -- that keeps the two counts independent in this test.
        _add(self.conn, "n1", "R7", "2026-02-01", [100.0])
        _add(self.conn, "n2", "R7", "2026-02-05", [100.0])
        _add(self.conn, "gone", "R7", "2025-11-01", [100.0], delisted="2026-02-10")
        new, delisted = metrics.churn(self.conn, "R7", as_of="2026-03-01", window_days=60)
        self.assertEqual((new, delisted), (2, 1))

    def test_churn_counts_a_listing_born_and_died_in_window_on_both_sides(self):
        _add(self.conn, "quick", "R9", "2026-02-01", [100.0], delisted="2026-02-20")
        self.assertEqual(metrics.churn(self.conn, "R9", as_of="2026-03-01", window_days=60), (1, 1))

    def test_composite_score_is_bounded(self):
        _add(self.conn, "a", "R8", "2026-01-01", [100.0, 95.0], delisted="2026-01-20")
        _add(self.conn, "b", "R8", "2026-01-01", [100.0, 100.0])
        row = metrics.reference_metrics(self.conn, "R8", as_of="2026-03-01", window_days=180)
        score = metrics._composite_score(row)
        if score is not None:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestIngestLifecycle(unittest.TestCase):
    def setUp(self):
        self.conn = db.connect(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_relisting_clears_a_premature_delist(self):
        """A scrape gap must not be recorded as a sale."""
        raw = [RawListing(listing_id="c24:1", title="Rolex Submariner 126610LN 2023",
                          price_text="€11.500")]
        ingest.upsert_listings(self.conn, raw, observed_at="2026-01-01")
        ingest.mark_delisted(self.conn, [], "2026-01-08")
        gone = self.conn.execute(
            "SELECT delisted_at FROM listings WHERE listing_id='c24:1'").fetchone()[0]
        self.assertEqual(gone, "2026-01-08")

        ingest.upsert_listings(self.conn, raw, observed_at="2026-01-15")
        back = self.conn.execute(
            "SELECT delisted_at FROM listings WHERE listing_id='c24:1'").fetchone()[0]
        self.assertIsNone(back, "a reappearing listing must not stay marked delisted")

    def test_unresolved_references_are_kept_and_counted(self):
        raw = [RawListing(listing_id="c24:9", title="Vintage timepiece, lovely patina",
                          price_text="€800")]
        report = ingest.upsert_listings(self.conn, raw, observed_at="2026-01-01")
        self.assertEqual(report.unresolved_reference, 1)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0], 1,
            "unparsed rows must be retained as the normaliser's work queue",
        )

    def test_snapshot_written_per_observation(self):
        raw = [RawListing(listing_id="c24:2", title="Rolex 126610LN", price_text="€11.500")]
        ingest.upsert_listings(self.conn, raw, observed_at="2026-01-01")
        raw[0].price_text = "€11.000"
        ingest.upsert_listings(self.conn, raw, observed_at="2026-01-08")
        rows = self.conn.execute(
            "SELECT price_cents FROM listing_snapshots WHERE listing_id='c24:2' "
            "ORDER BY observed_at").fetchall()
        self.assertEqual([r[0] for r in rows], [1150000, 1100000])


if __name__ == "__main__":
    unittest.main(verbosity=2)
