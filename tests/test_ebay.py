"""eBay Browse API adapter tests.

The network layer is faked -- this sandbox cannot reach eBay's live API --
but the shapes exercised here match eBay's documented conventions: the
client-credentials OAuth2 token response (RFC 6749 standard fields:
access_token/expires_in/token_type), the item_summary/search resource's
ItemSummary fields (itemId, price.value/currency, itemLocation.country,
seller.sellerAccountType, itemWebUrl), and eBay's standard REST error
envelope (an ``errors`` array with errorId/domain/category/message), all as
documented at developer.ebay.com. Nothing here is a guess the way Chrono24's
first SELECTORS pass was.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watchlab import db, ingest  # noqa: E402
from watchlab.sources import ebay  # noqa: E402

TOKEN_RESPONSE = {
    "access_token": "test-app-token",
    "expires_in": 7200,
    "token_type": "Application Access Token",
}

OAUTH_ERROR = {"error": "invalid_client", "error_description": "client authentication failed"}

ITEM_SUMMARY_SEARCH = {
    "href": "https://api.ebay.com/buy/browse/v1/item_summary/search?q=Seiko+SRPD55K1",
    "total": 2,
    "itemSummaries": [
        {
            "itemId": "v1|123456789012|0",
            "title": "Seiko 5 Sports SRPD55K1 Automatic Men's Watch Used",
            "price": {"value": "179.99", "currency": "GBP"},
            "condition": "Used",
            "itemLocation": {"country": "GB"},
            "seller": {"username": "watchseller123", "sellerAccountType": "INDIVIDUAL"},
            "itemWebUrl": "https://www.ebay.co.uk/itm/123456789012",
        },
        {
            "itemId": "v1|987654321098|0",
            "title": "Seiko 5 Sports SRPD55K1 New with tags",
            "price": {"value": "219.00", "currency": "GBP"},
            "condition": "New with tags",
            "itemLocation": {"country": "DE"},
            "seller": {"username": "germanwatches", "sellerAccountType": "BUSINESS"},
            "itemWebUrl": "https://www.ebay.co.uk/itm/987654321098",
        },
    ],
    "limit": 50,
    "offset": 0,
}

ERROR_ENVELOPE = {
    "errors": [
        {"errorId": 12023, "domain": "API_BROWSE", "category": "REQUEST",
         "message": "Invalid marketplace id."}
    ]
}


class FakeTokenFetcher:
    def __init__(self, responses):
        self.queue = list(responses)
        self.calls = 0

    def __call__(self, client_id, client_secret, scope, token_url, timeout):
        self.calls += 1
        if not self.queue:
            raise AssertionError("FakeTokenFetcher queue exhausted")
        return self.queue.pop(0)


class FakeOpener:
    def __init__(self, responses):
        self.queue = list(responses)
        self.calls = []

    def __call__(self, url, token, marketplace_id, timeout):
        self.calls.append((url, token, marketplace_id))
        if not self.queue:
            raise AssertionError("FakeOpener queue exhausted")
        status, payload = self.queue.pop(0)
        return status, {}, json.dumps(payload).encode("utf-8")


def _client(token_responses, opener_responses, cache_dir=None, max_retries=2):
    token_fetcher = FakeTokenFetcher(token_responses)
    opener = FakeOpener(opener_responses)
    client = ebay.Client(
        client_id="id", client_secret="secret", cache_dir=cache_dir,
        token_fetcher=token_fetcher, opener=opener, max_retries=max_retries,
    )
    return client, token_fetcher, opener


class TestClientRequiresCredentials(unittest.TestCase):
    def test_empty_client_id_raises(self):
        with self.assertRaises(ValueError):
            ebay.Client(client_id="", client_secret="secret")

    def test_empty_client_secret_raises(self):
        with self.assertRaises(ValueError):
            ebay.Client(client_id="id", client_secret="")


class TestTokenLifecycle(unittest.TestCase):
    def test_token_fetched_once_and_reused_across_calls(self):
        client, token_fetcher, opener = _client(
            [(200, TOKEN_RESPONSE)], [(200, ITEM_SUMMARY_SEARCH), (200, ITEM_SUMMARY_SEARCH)],
        )
        ebay.search_items(client, "Seiko SRPD55K1")
        ebay.search_items(client, "Seiko SRPD55K1 full set")
        self.assertEqual(token_fetcher.calls, 1, "one token should serve multiple search calls")
        self.assertEqual(opener.calls[0][1], "test-app-token")

    def test_oauth_failure_raises_typed_error(self):
        client, _, _ = _client([(400, OAUTH_ERROR)], [])
        with self.assertRaises(ebay.EbayError) as ctx:
            ebay.search_items(client, "Seiko SRPD55K1")
        self.assertEqual(ctx.exception.code, "oauth_failed")
        self.assertEqual(ctx.exception.http_status, 400)

    def test_401_triggers_one_token_refetch_and_retry(self):
        client, token_fetcher, opener = _client(
            [(200, TOKEN_RESPONSE), (200, TOKEN_RESPONSE)],
            [(401, {"errors": [{"errorId": 1001, "message": "Invalid access token"}]}),
             (200, ITEM_SUMMARY_SEARCH)],
        )
        result = ebay.search_items(client, "Seiko SRPD55K1")
        self.assertEqual(len(result), 2)
        self.assertEqual(token_fetcher.calls, 2, "a 401 should force exactly one refetch")


class TestSearchItems(unittest.TestCase):
    def test_returns_item_summaries(self):
        client, _, _ = _client([(200, TOKEN_RESPONSE)], [(200, ITEM_SUMMARY_SEARCH)])
        result = ebay.search_items(client, "Seiko SRPD55K1")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["itemId"], "v1|123456789012|0")

    def test_query_reaches_the_request_url(self):
        client, _, opener = _client([(200, TOKEN_RESPONSE)], [(200, ITEM_SUMMARY_SEARCH)])
        ebay.search_items(client, "Seiko SRPD55K1")
        self.assertIn("q=Seiko", opener.calls[0][0])

    def test_no_results_returns_empty_list(self):
        client, _, _ = _client([(200, TOKEN_RESPONSE)], [(200, {"itemSummaries": []})])
        self.assertEqual(ebay.search_items(client, "nonexistent watch"), [])


class TestRawListingConversion(unittest.TestCase):
    def test_maps_documented_fields(self):
        listing = ebay.to_raw_listing(ITEM_SUMMARY_SEARCH["itemSummaries"][0])
        self.assertEqual(listing.listing_id, "ebay:v1|123456789012|0")
        self.assertEqual(listing.price_text, "179.99")
        self.assertEqual(listing.currency, "GBP")
        self.assertEqual(listing.seller_country, "GB")
        self.assertEqual(listing.seller_type, "private")
        self.assertEqual(listing.url, "https://www.ebay.co.uk/itm/123456789012")

    def test_business_seller_maps_to_dealer(self):
        listing = ebay.to_raw_listing(ITEM_SUMMARY_SEARCH["itemSummaries"][1])
        self.assertEqual(listing.seller_type, "dealer")

    def test_known_condition_strings_map(self):
        self.assertEqual(ebay._map_condition("Used"), "good")
        self.assertEqual(ebay._map_condition("New with tags"), "new")
        self.assertEqual(ebay._map_condition("For parts or not working"), "poor")

    def test_unrecognised_condition_stays_none(self):
        """Same rule normalize.py applies to box/papers: don't guess."""
        self.assertIsNone(ebay._map_condition("Some future eBay condition string"))
        self.assertIsNone(ebay._map_condition(None))

    def test_missing_item_id_returns_none(self):
        self.assertIsNone(ebay.to_raw_listing({"price": {"value": "100"}}))

    def test_missing_price_returns_none(self):
        self.assertIsNone(ebay.to_raw_listing({"itemId": "v1|1|0"}))


class TestSearchWatchlist(unittest.TestCase):
    def test_dedupes_across_queries(self):
        client, _, _ = _client(
            [(200, TOKEN_RESPONSE)], [(200, ITEM_SUMMARY_SEARCH), (200, ITEM_SUMMARY_SEARCH)],
        )
        listings = ebay.search_watchlist(client, ["Seiko SRPD55K1", "Seiko SRPD55K1 full set"])
        self.assertEqual(len(listings), 2, "the same two items appearing twice should dedupe")

    def test_feeds_upsert_listings(self):
        client, _, _ = _client([(200, TOKEN_RESPONSE)], [(200, ITEM_SUMMARY_SEARCH)])
        listings = ebay.search_watchlist(client, ["Seiko SRPD55K1"])
        conn = db.connect(":memory:")
        report = ingest.upsert_listings(conn, listings, observed_at="2026-08-19", source="ebay")
        self.assertEqual(report.new_listings, 2)
        stored = conn.execute("SELECT COUNT(*) FROM listings WHERE source='ebay'").fetchone()[0]
        self.assertEqual(stored, 2)
        conn.close()


class TestErrorHandling(unittest.TestCase):
    def test_error_envelope_parses_into_typed_error(self):
        client, _, _ = _client([(200, TOKEN_RESPONSE)], [(400, ERROR_ENVELOPE)])
        with self.assertRaises(ebay.EbayError) as ctx:
            ebay.search_items(client, "Seiko SRPD55K1")
        self.assertEqual(ctx.exception.code, "12023")
        self.assertEqual(ctx.exception.http_status, 400)

    def test_rate_limit_retries_then_succeeds(self):
        client, _, opener = _client(
            [(200, TOKEN_RESPONSE)],
            [(429, {"errors": []}), (429, {"errors": []}), (200, ITEM_SUMMARY_SEARCH)],
        )
        import watchlab.sources.ebay as ebay_module
        original_sleep = ebay_module.time.sleep
        ebay_module.time.sleep = lambda *_: None
        try:
            result = ebay.search_items(client, "Seiko SRPD55K1")
        finally:
            ebay_module.time.sleep = original_sleep
        self.assertEqual(len(result), 2)
        self.assertEqual(len(opener.calls), 3)

    def test_retries_exhausted_raises(self):
        client, _, opener = _client(
            [(200, TOKEN_RESPONSE)], [(500, {"errors": []})] * 10,
        )
        import watchlab.sources.ebay as ebay_module
        original_sleep = ebay_module.time.sleep
        ebay_module.time.sleep = lambda *_: None
        try:
            with self.assertRaises(ebay.EbayError):
                ebay.search_items(client, "Seiko SRPD55K1")
        finally:
            ebay_module.time.sleep = original_sleep
        self.assertEqual(len(opener.calls), client.max_retries + 1)


class TestCaching(unittest.TestCase):
    def test_second_call_hits_cache_not_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, _, opener = _client(
                [(200, TOKEN_RESPONSE)], [(200, ITEM_SUMMARY_SEARCH)], cache_dir=tmp,
            )
            ebay.search_items(client, "Seiko SRPD55K1")
            ebay.search_items(client, "Seiko SRPD55K1")
            self.assertEqual(len(opener.calls), 1, "second identical call should be served from cache")

    def test_cache_key_excludes_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            opener = FakeOpener([(200, ITEM_SUMMARY_SEARCH), (200, ITEM_SUMMARY_SEARCH)])
            client_a = ebay.Client(
                client_id="id-a", client_secret="secret-a", cache_dir=tmp,
                token_fetcher=FakeTokenFetcher([(200, {**TOKEN_RESPONSE, "access_token": "token-a"})]),
                opener=opener,
            )
            ebay.search_items(client_a, "Seiko SRPD55K1")
            client_b = ebay.Client(
                client_id="id-b", client_secret="secret-b", cache_dir=tmp,
                token_fetcher=FakeTokenFetcher([(200, {**TOKEN_RESPONSE, "access_token": "token-b"})]),
                opener=opener,
            )
            ebay.search_items(client_b, "Seiko SRPD55K1")
            self.assertEqual(len(opener.calls), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
