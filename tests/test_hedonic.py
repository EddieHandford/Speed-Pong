"""Validate the index estimator against a market with a known true path.

The headline test is ``test_hedonic_beats_naive_median``. The synthetic market
drifts its quality mix over time (progressively more new/unworn examples), so
a naive median ask picks up appreciation that did not happen. The hedonic
index controls for that mix and should track the true latent path much more
closely. If that assertion ever fails, the index is not doing its job and no
number produced by this project should be trusted.
"""

from __future__ import annotations

import datetime as _dt
import math
import os
import statistics
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watchlab import db, hedonic, ingest  # noqa: E402
from watchlab.sources import synthetic  # noqa: E402


class TestLinearAlgebra(unittest.TestCase):
    def test_solve_matches_known_solution(self):
        matrix = [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]
        rhs = [8.0, -11.0, -3.0]
        result = hedonic.solve(matrix, rhs)
        for got, want in zip(result, [2.0, 3.0, -1.0]):
            self.assertAlmostEqual(got, want, places=9)

    def test_invert_round_trips(self):
        matrix = [[4.0, 7.0], [2.0, 6.0]]
        inverse = hedonic.invert(matrix)
        product = [
            [sum(matrix[i][k] * inverse[k][j] for k in range(2)) for j in range(2)]
            for i in range(2)
        ]
        self.assertAlmostEqual(product[0][0], 1.0, places=9)
        self.assertAlmostEqual(product[1][1], 1.0, places=9)
        self.assertAlmostEqual(product[0][1], 0.0, places=9)

    def test_singular_matrix_raises(self):
        with self.assertRaises(hedonic.SingularMatrixError):
            hedonic.solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


class TestOLSRecoversCoefficients(unittest.TestCase):
    def test_known_linear_model(self):
        rows, target = [], []
        for i in range(200):
            x1 = (i % 10) / 10.0
            x2 = 1.0 if i % 3 == 0 else 0.0
            rows.append([1.0, x1, x2])
            target.append(3.0 + 2.0 * x1 - 0.5 * x2)
        fit = hedonic.fit_ols(rows, target, ["(intercept)", "x1", "x2"], with_se=True)
        self.assertAlmostEqual(fit.coef("(intercept)"), 3.0, places=4)
        self.assertAlmostEqual(fit.coef("x1"), 2.0, places=4)
        self.assertAlmostEqual(fit.coef("x2"), -0.5, places=4)
        self.assertGreater(fit.r_squared, 0.999)

    def test_categorical_expansion_drops_base_level(self):
        rows = [{"colour": c} for c in ["red", "green", "blue", "red"]]
        matrix, columns = hedonic.build_design(rows, [hedonic.Feature("colour")])
        self.assertEqual(columns, ["(intercept)", "colour=green", "colour=red"])
        self.assertEqual(matrix[0], [1.0, 0.0, 1.0])   # red
        self.assertEqual(matrix[2], [1.0, 0.0, 0.0])   # blue is the base

    def test_missing_values_become_their_own_level(self):
        rows = [{"papers": 1}, {"papers": 0}, {"papers": None}, {"papers": 1}]
        _, columns = hedonic.build_design(rows, [hedonic.Feature("papers")])
        self.assertIn("papers=(unknown)", columns)


class TestIndexAgainstGroundTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = db.connect(":memory:")
        cls.start = _dt.date(2024, 1, 1)
        cls.end = _dt.date(2026, 1, 1)
        cls.truth = synthetic.generate(cls.conn, cls.start, cls.end, seed=7)
        cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_generator_produced_a_market(self):
        n_listings = self.conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        n_snapshots = self.conn.execute("SELECT COUNT(*) FROM listing_snapshots").fetchone()[0]
        self.assertGreater(n_listings, 2000)
        self.assertGreater(n_snapshots, n_listings)

    def test_hedonic_beats_naive_median(self):
        """The whole thesis of the project, as an assertion."""
        reference = "126610LN"
        rows = ingest.hedonic_rows(self.conn, reference=reference)
        self.assertGreater(len(rows), 300)

        points, fit = hedonic.time_dummy_index(
            rows, hedonic.standard_features(), with_ci=False
        )
        self.assertGreater(fit.r_squared, 0.5)

        truth = self.truth[reference]
        periods = [p.period for p in points if p.period in truth]
        base_period = periods[0]
        base_truth = truth[base_period]

        true_index = {p: 100.0 * truth[p] / base_truth for p in periods}

        by_period: dict[str, list[float]] = {}
        for row in rows:
            by_period.setdefault(row["period"], []).append(row["price"])
        base_median = statistics.median(by_period[base_period])
        naive_index = {
            p: 100.0 * statistics.median(by_period[p]) / base_median for p in periods
        }

        hedonic_index = {p.period: p.value for p in points if p.period in truth}

        hedonic_error = _rmse([hedonic_index[p] for p in periods],
                              [true_index[p] for p in periods])
        naive_error = _rmse([naive_index[p] for p in periods],
                            [true_index[p] for p in periods])

        self.assertLess(
            hedonic_error, naive_error,
            f"hedonic RMSE {hedonic_error:.2f} should beat naive {naive_error:.2f}",
        )
        # Not merely better -- close enough in absolute terms to be usable.
        self.assertLess(hedonic_error, 6.0, f"hedonic RMSE {hedonic_error:.2f} too high")

    def test_index_recovers_direction_of_drift(self):
        """A reference generated with negative drift must index downward."""
        rows = ingest.hedonic_rows(self.conn, reference="5711/1A-010")
        points, _ = hedonic.time_dummy_index(
            rows, hedonic.standard_features(), with_ci=False, min_obs_per_period=3
        )
        cagr = hedonic.annualised_return(points)
        self.assertIsNotNone(cagr)
        self.assertLess(cagr, 0.0)

    def test_quality_premiums_have_the_right_sign(self):
        rows = ingest.hedonic_rows(self.conn, reference="126610LN")
        matrix, columns = hedonic.build_design(rows, hedonic.standard_features())
        target = [math.log(r["price"]) for r in rows]
        fit = hedonic.fit_ols(matrix, target, columns)

        papers = fit.coef("has_papers=1")
        self.assertIsNotNone(papers)
        self.assertGreater(papers, 0.0, "papers should carry a premium")

        new = fit.coef("condition=new")
        good = fit.coef("condition=good")
        self.assertGreater(new, good, "new should price above good")

    def test_min_obs_per_period_filters_thin_periods(self):
        rows = [
            {"period": "2025-01", "price": 100.0, "condition": "good"},
            {"period": "2025-02", "price": 110.0, "condition": "good"},
        ] * 5
        rows.append({"period": "2025-03", "price": 900.0, "condition": "good"})
        points, _ = hedonic.time_dummy_index(
            rows, [hedonic.Feature("condition")], with_ci=False, min_obs_per_period=3
        )
        self.assertNotIn("2025-03", [p.period for p in points])


def _rmse(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


if __name__ == "__main__":
    unittest.main(verbosity=2)
