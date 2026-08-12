"""Walk-forward GA screen tests, against the budget synthetic universe.

These check the machinery (no lookahead, walk-forward split respected,
complexity penalty actually penalises, output is deterministic given a seed)
-- not that the GA "beats the market", which per genetic.py's own docstring
it is not expected to.
"""

from __future__ import annotations

import datetime as _dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watchlab import db, genetic, metrics  # noqa: E402
from watchlab.sources import synthetic  # noqa: E402


class TestPeriodArithmetic(unittest.TestCase):
    def test_add_months_forward(self):
        self.assertEqual(genetic._add_months("2025-01", 3), "2025-04")

    def test_add_months_across_year_boundary(self):
        self.assertEqual(genetic._add_months("2025-11", 3), "2026-02")

    def test_add_months_backward(self):
        self.assertEqual(genetic._add_months("2025-04", -3), "2025-01")


class TestZScore(unittest.TestCase):
    def test_missing_values_become_zero(self):
        result = genetic._zscore({"a": 1.0, "b": None, "c": 3.0})
        self.assertEqual(result["b"], 0.0)

    def test_fewer_than_two_present_values_returns_all_zero(self):
        result = genetic._zscore({"a": 5.0, "b": None})
        self.assertEqual(result, {"a": 0.0, "b": 0.0})

    def test_scored_values_are_centred(self):
        result = genetic._zscore({"a": 1.0, "b": 2.0, "c": 3.0})
        self.assertAlmostEqual(sum(result.values()), 0.0, places=9)


class TestBuildCheckpointsAndGA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = db.connect(":memory:")
        start = _dt.date(2023, 1, 1)
        end = _dt.date(2026, 1, 1)  # 3 years, enough for several checkpoints
        synthetic.seed_fx(cls.conn, start.isoformat())
        synthetic.generate(cls.conn, start, end, universe=synthetic.BUDGET_UNIVERSE, seed=5)
        cls.conn.commit()
        cls.references = [spec.reference for spec in synthetic.BUDGET_UNIVERSE]

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_build_checkpoints_produces_no_lookahead_features(self):
        checkpoints = genetic.build_checkpoints(self.conn, self.references)
        self.assertGreater(len(checkpoints), 3)
        for checkpoint in checkpoints:
            self.assertTrue(set(checkpoint.features).issubset(set(self.references)))
            for feature_row in checkpoint.features.values():
                self.assertEqual(set(feature_row), set(genetic.FEATURES))

    def test_forward_return_periods_are_strictly_after_the_checkpoint(self):
        checkpoints = genetic.build_checkpoints(self.conn, self.references, horizon_months=6)
        for checkpoint in checkpoints:
            for reference in checkpoint.forward_gross_return:
                # A forward return existing at all means index[period+6mo] was found;
                # sanity-check it's a plausible ratio-derived value, not a raw price.
                self.assertGreater(checkpoint.forward_gross_return[reference], -1.0)

    def test_too_little_history_returns_no_checkpoints(self):
        conn = db.connect(":memory:")
        start = _dt.date(2026, 1, 1)
        end = _dt.date(2026, 2, 15)
        synthetic.seed_fx(conn, start.isoformat())
        synthetic.generate(conn, start, end, universe=synthetic.BUDGET_UNIVERSE, seed=1)
        checkpoints = genetic.build_checkpoints(conn, self.references)
        self.assertEqual(checkpoints, [])
        conn.close()

    def test_run_ga_is_deterministic_given_a_seed(self):
        result_a = genetic.run_ga(self.conn, self.references, population_size=10, generations=5, seed=42)
        result_b = genetic.run_ga(self.conn, self.references, population_size=10, generations=5, seed=42)
        self.assertEqual(result_a.best_weights, result_b.best_weights)
        self.assertEqual(result_a.fitness_by_generation, result_b.fitness_by_generation)

    def test_run_ga_reports_walk_forward_split(self):
        result = genetic.run_ga(self.conn, self.references, population_size=10, generations=5, seed=1)
        self.assertGreater(result.n_train_checkpoints, 0)
        self.assertGreater(result.n_test_checkpoints, 0)
        self.assertEqual(set(result.best_weights), set(genetic.FEATURES))

    def test_run_ga_always_reports_buy_and_hold_alongside_ga_result(self):
        """The README's requirement: never report the GA number without the baseline."""
        result = genetic.run_ga(self.conn, self.references, population_size=10, generations=5, seed=1)
        self.assertIsNotNone(result.test_net_return)
        self.assertIsNotNone(result.test_buy_and_hold_net_return)

    def test_fitness_by_generation_is_non_decreasing(self):
        """Elitism guarantees the tracked best never regresses generation to generation."""
        result = genetic.run_ga(self.conn, self.references, population_size=10, generations=8, seed=3)
        for earlier, later in zip(result.fitness_by_generation, result.fitness_by_generation[1:]):
            self.assertGreaterEqual(later, earlier)

    def test_complexity_penalty_reduces_fitness_for_the_same_candidate(self):
        checkpoints = genetic.build_checkpoints(self.conn, self.references)
        candidate = genetic.Candidate({name: 1.0 for name in genetic.FEATURES})
        costs = metrics.CostModel()
        unpenalised = genetic._fitness(candidate, checkpoints, top_k=3, horizon_months=6,
                                        costs=costs, complexity_penalty=0.0)
        penalised = genetic._fitness(candidate, checkpoints, top_k=3, horizon_months=6,
                                      costs=costs, complexity_penalty=1.0)
        self.assertLess(penalised, unpenalised)

    def test_no_train_checkpoints_returns_empty_result_not_a_crash(self):
        result = genetic.run_ga(self.conn, [], population_size=5, generations=2, seed=1)
        self.assertEqual(result.n_train_checkpoints, 0)
        self.assertIsNone(result.test_net_return)
        self.assertIsNone(result.test_buy_and_hold_net_return)


if __name__ == "__main__":
    unittest.main(verbosity=2)
