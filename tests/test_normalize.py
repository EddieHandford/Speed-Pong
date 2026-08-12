"""Entity-resolution tests.

Every one of these titles is the kind of thing a real marketplace listing
looks like. The failure mode this guards against is silent: a title that
resolves to the wrong reference does not raise, it just quietly pollutes a
price series with a different watch.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watchlab import normalize  # noqa: E402

CATALOGUE = {
    "126610LN": "126610LN",
    "116610LV": "116610LV",
    "5711/1A-010": "5711/1A-010",
    "310.30.42.50.01.002": "310.30.42.50.01.002",
    "15500ST.OO.1220ST.01": "15500ST.OO.1220ST.01",
    "IW371604": "IW371604",
}


class TestBrandDetection(unittest.TestCase):
    def test_common_brands(self):
        cases = {
            "Rolex Submariner": "Rolex",
            "OMEGA Speedmaster": "Omega",
            "AP Royal Oak 15500ST": "Audemars Piguet",
            "PP Nautilus": "Patek Philippe",
            "JLC Reverso": "Jaeger-LeCoultre",
        }
        for title, want in cases.items():
            self.assertEqual(normalize.detect_brand(normalize._fold(title)), want, title)

    def test_longest_alias_wins(self):
        """'Grand Seiko' must not resolve to 'Seiko'."""
        folded = normalize._fold("Grand Seiko Snowflake SBGA211")
        self.assertEqual(normalize.detect_brand(folded), "Grand Seiko")

    def test_unknown_brand_is_none(self):
        self.assertIsNone(normalize.detect_brand(normalize._fold("Casio F-91W")))


class TestReferenceExtraction(unittest.TestCase):
    def test_catalogue_hit_scores_highest(self):
        parsed = normalize.parse_title(
            "Rolex Submariner Date 126610LN 2023 Box & Papers", catalogue=CATALOGUE
        )
        self.assertEqual(parsed.reference, "126610LN")
        self.assertEqual(parsed.brand, "Rolex")
        self.assertGreater(parsed.confidence, 0.85)

    def test_separator_variants_collapse(self):
        """The same watch written three ways must produce one key."""
        titles = [
            "Rolex Submariner 126610LN unworn",
            "Rolex Submariner 126610 LN unworn",
            "ROLEX SUBMARINER DATE 126610-LN UNWORN",
        ]
        keys = {normalize.parse_title(t).key for t in titles}
        self.assertEqual(len(keys), 1, f"expected one key, got {keys}")

    def test_dotted_references_keep_their_dots(self):
        parsed = normalize.parse_title(
            "Omega Speedmaster Moonwatch 310.30.42.50.01.002 new", catalogue=CATALOGUE
        )
        self.assertEqual(parsed.reference, "310.30.42.50.01.002")

    def test_patek_slashed_reference(self):
        parsed = normalize.parse_title(
            "Patek Philippe Nautilus 5711/1A-010 mint", catalogue=CATALOGUE
        )
        self.assertEqual(parsed.reference, "5711/1A-010")

    def test_iwc_prefixed_reference(self):
        parsed = normalize.parse_title("IWC Portugieser Chronograph IW371604")
        self.assertEqual(parsed.reference, "IW371604")

    def test_bare_four_digits_is_not_a_reference(self):
        """A stray year must not become a reference for non-Patek brands."""
        parsed = normalize.parse_title("Omega Seamaster from 1998 serviced")
        self.assertIsNone(parsed.reference)
        self.assertEqual(parsed.production_year, 1998)


class TestYearExtraction(unittest.TestCase):
    def test_year_not_taken_from_reference_digits(self):
        """'116610LV' contains no year; '2015' in the title is the year."""
        parsed = normalize.parse_title(
            "Rolex Submariner Hulk 116610LV 2015 full set", catalogue=CATALOGUE
        )
        self.assertEqual(parsed.reference, "116610LV")
        self.assertEqual(parsed.production_year, 2015)

    def test_no_year_returns_none(self):
        parsed = normalize.parse_title("Rolex Submariner 126610LN", catalogue=CATALOGUE)
        self.assertIsNone(parsed.production_year)


class TestConditionAndCompleteness(unittest.TestCase):
    def test_condition_phrases(self):
        cases = {
            "Rolex 126610LN brand new": "new",
            "Rolex 126610LN unworn 2024": "unworn",
            "Rolex 126610LN mint condition": "very_good",
            "Rolex 126610LN very good": "very_good",
            "Rolex 126610LN fair": "fair",
        }
        for title, want in cases.items():
            self.assertEqual(normalize.parse_title(title).condition, want, title)

    def test_full_set_sets_both(self):
        parsed = normalize.parse_title("Rolex 126610LN full set")
        self.assertEqual((parsed.has_box, parsed.has_papers), (1, 1))

    def test_bare_watch_clears_both(self):
        parsed = normalize.parse_title("Rolex 126610LN watch only no box no papers")
        self.assertEqual((parsed.has_box, parsed.has_papers), (0, 0))

    def test_unknown_completeness_stays_none(self):
        """Silence about papers is not evidence of absence."""
        parsed = normalize.parse_title("Rolex Submariner 126610LN 2023")
        self.assertIsNone(parsed.has_papers)
        self.assertIsNone(parsed.has_box)

    def test_papers_only(self):
        parsed = normalize.parse_title("Rolex 126610LN papers only")
        self.assertEqual(parsed.has_papers, 1)


class TestPriceParsing(unittest.TestCase):
    def test_european_and_anglo_grouping(self):
        cases = {
            "€11.500": (1150000, "EUR"),
            "€11.500,50": (1150050, "EUR"),
            "$12,750": (1275000, "USD"),
            "$12,750.25": (1275025, "USD"),
            "£9,999": (999900, "GBP"),
            "CHF 14'000": (1400000, "CHF"),
        }
        for text, (want_cents, want_ccy) in cases.items():
            cents, currency = normalize.parse_price(text)
            self.assertEqual(currency, want_ccy, text)
            self.assertEqual(cents, want_cents, text)

    def test_non_breaking_space_grouping(self):
        cents, currency = normalize.parse_price("11 500 €")
        self.assertEqual(cents, 1150000)
        self.assertEqual(currency, "EUR")

    def test_empty_and_garbage(self):
        self.assertEqual(normalize.parse_price(""), (None, None))
        self.assertEqual(normalize.parse_price("Price on request")[0], None)


class TestAccentAndCaseFolding(unittest.TestCase):
    def test_accents_do_not_split_a_reference(self):
        a = normalize.parse_title("A. Lange & Söhne Lange 1 191.032")
        b = normalize.parse_title("A. Lange & Sohne Lange 1 191.032")
        self.assertEqual(a.brand, b.brand)
        self.assertEqual(a.reference, b.reference)


if __name__ == "__main__":
    unittest.main(verbosity=2)
