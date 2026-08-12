"""Tests for the (unverified, no live access) auction-house HTML parser.

These only prove the JSON-LD extraction logic works on well-formed schema.org
markup -- they cannot prove any real auction house's results pages actually
carry that markup, since this sandbox has no network access to check. See
sources/auction_houses.py's module docstring for the honest version of that
caveat: schema.org's Offer type is a poor semantic fit for a lot that has
already sold, more so than it was for Chrono24's live listings, so treat this
path as worth trying, not as something to trust without running `calibrate`
against a real saved page first.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watchlab.sources import auction_houses  # noqa: E402

JSONLD_PAGE = """
<html><head>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "Rolex Daytona ref. 116520, Lot 142",
  "sku": "142",
  "url": "https://example-house.com/lots/142",
  "offers": {
    "@type": "Offer",
    "price": "54120",
    "priceCurrency": "USD"
  }
}
</script>
</head><body>
<h1>Lot 142: Rolex Daytona</h1>
<p>Price Realised: $54,120</p>
</body></html>
"""

MULTI_LOT_PAGE = """
<html><body>
<script type="application/ld+json">
[
  {"@type": "Product", "name": "Lot 1: Patek 5711", "sku": "1",
   "offers": {"@type": "Offer", "price": "180000", "priceCurrency": "USD"}},
  {"@type": "Product", "name": "Lot 2: AP 15500ST", "sku": "2",
   "offers": {"@type": "Offer", "price": "42000", "priceCurrency": "USD"}}
]
</script>
</body></html>
"""

NO_STRUCTURED_DATA_PAGE = """
<html><body>
<div class="lot-result">
  <h2>Lot 3: Omega Speedmaster</h2>
  <span class="price-realised">$8,400</span>
</div>
</body></html>
"""


class TestParseJsonLd(unittest.TestCase):
    def test_single_lot(self):
        lots = auction_houses.parse_jsonld(JSONLD_PAGE, house="Example House")
        self.assertEqual(len(lots), 1)
        lot = lots[0]
        self.assertEqual(lot.house, "Example House")
        self.assertEqual(lot.lot_number, "142")
        self.assertEqual(lot.price_text, "54120")
        self.assertEqual(lot.currency, "USD")
        self.assertEqual(lot.lot_url, "https://example-house.com/lots/142")
        self.assertIn("Daytona", lot.raw_title)

    def test_multiple_lots_in_one_array(self):
        lots = auction_houses.parse_jsonld(MULTI_LOT_PAGE, house="Example House")
        self.assertEqual(len(lots), 2)
        self.assertEqual({lot.lot_number for lot in lots}, {"1", "2"})

    def test_no_structured_data_returns_empty_not_an_error(self):
        """The honest failure mode: no JSON-LD present, nothing crashes."""
        lots = auction_houses.parse_jsonld(NO_STRUCTURED_DATA_PAGE, house="Example House")
        self.assertEqual(lots, [])

    def test_house_is_tagged_on_every_lot(self):
        lots = auction_houses.parse_jsonld(JSONLD_PAGE, house="Christie's")
        self.assertTrue(all(lot.house == "Christie's" for lot in lots))


if __name__ == "__main__":
    unittest.main(verbosity=2)
