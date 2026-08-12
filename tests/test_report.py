"""Multi-reference ranking against the budget synthetic universe."""

from __future__ import annotations

import datetime as _dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watchlab import db, report  # noqa: E402
from watchlab.sources import synthetic  # noqa: E402


class TestRankReferences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = db.connect(":memory:")
        start = _dt.date(2024, 1, 1)
        end = _dt.date(2026, 1, 1)
        synthetic.seed_fx(cls.conn, start.isoformat())
        synthetic.generate(cls.conn, start, end, universe=synthetic.BUDGET_UNIVERSE, seed=11)
        cls.conn.commit()
        cls.references = [spec.reference for spec in synthetic.BUDGET_UNIVERSE]

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_every_reference_gets_a_verdict(self):
        ranked = report.rank_references(self.conn, self.references)
        self.assertEqual({r.reference for r in ranked}, set(self.references))
        for r in ranked:
            self.assertTrue(r.skip_reason is not None or r.net_cagr is not None)

    def test_price_ceiling_skips_references_over_it(self):
        ranked = report.rank_references(self.conn, self.references, max_price_eur=1.0)
        for r in ranked:
            self.assertIsNone(r.net_cagr)
            self.assertIn("ceiling", r.skip_reason)

    def test_unknown_reference_is_skipped_not_errored(self):
        ranked = report.rank_references(self.conn, ["NOT-A-REAL-REF"])
        self.assertEqual(len(ranked), 1)
        self.assertIsNone(ranked[0].net_cagr)
        self.assertIsNotNone(ranked[0].skip_reason)

    def test_top_bottom_are_sorted_and_disjoint_ends(self):
        ranked = report.rank_references(self.conn, self.references, min_obs=5)
        top, bottom = report.top_bottom(ranked, n=3)
        top_cagrs = [r.net_cagr for r in top]
        self.assertEqual(top_cagrs, sorted(top_cagrs, reverse=True))
        bottom_cagrs = [r.net_cagr for r in bottom]
        self.assertEqual(bottom_cagrs, sorted(bottom_cagrs))
        if len(top) == len(bottom) == 3 and len({r.reference for r in ranked if r.net_cagr is not None}) >= 6:
            self.assertFalse(set(r.reference for r in top) & set(r.reference for r in bottom))


if __name__ == "__main__":
    unittest.main(verbosity=2)
