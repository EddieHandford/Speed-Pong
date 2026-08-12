"""thewatchapi.com adapter tests.

The network layer is faked -- this sandbox cannot reach the live API -- but
every payload used here is copied verbatim from the real documentation at
thewatchapi.com/documentation (brand list, Rolex Daytona model/search
results, reference/search, and brand/model/reference price history), so the
parsing logic is validated against real documented shapes, not invented ones.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watchlab import catalogue, db, ingest  # noqa: E402
from watchlab.sources import thewatchapi as twa  # noqa: E402

BRAND_LIST = {"data": ["A. Lange & Söhne", "Alpina", "Rolex", "Zenith", "ZRC"]}

MODEL_LIST = {"data": ["Rolex Air King", "Rolex Cellini", "Rolex Datejust"]}

REFERENCE_LIST = {"data": ["226570", "14000M", "126900", "76080", "116520"]}

BRAND_SEARCH = {"data": ["Rolex"]}

REFERENCE_SEARCH = {"data": [{"brand": "Rolex", "reference_number": "116520"}]}

# Trimmed from the real two-record example in the docs (descriptions shortened).
MODEL_SEARCH = {
    "data": [
        {
            "brand": "Rolex", "reference_number": "116520", "model": "Rolex Daytona",
            "movement": "Automatic", "year_of_production": "1989 - 2018",
            "case_material": "Steel", "case_diameter": "40 mm",
            "description": "Introducing the pinnacle of luxury...",
            "last_updated": "2023-10-12 12:12:33",
        },
        {
            "brand": "Rolex", "reference_number": "116500LN", "model": "Rolex Daytona",
            "movement": "Automatic", "year_of_production": "2010 - 2023",
            "case_material": "Steel", "case_diameter": "40 mm",
            "description": "The Rolex Daytona reference number 116500LN...",
            "last_updated": "2023-10-12 12:29:54",
        },
    ]
}

REFERENCE_PRICE_HISTORY = {
    "meta": {"brand": "Rolex", "reference_number": "116520"},
    "data": [
        {"date": "2023-10-13T00:00:00.000Z", "price": 24653.12},
        {"date": "2023-10-12T00:00:00.000Z", "price": 24640.94},
    ],
}

ERROR_INVALID_TOKEN = {"error": {"code": "invalid_api_token", "message": "Invalid API token."}}
ERROR_USAGE_LIMIT = {"error": {"code": "usage_limit_reached", "message": "Plan limit reached."}}
ERROR_RATE_LIMIT = {"error": {"code": "rate_limit_reached", "message": "Too many requests."}}


class FakeTransport:
    """Replays a queue of (status, payload) pairs, one per call, in order."""

    def __init__(self, responses):
        self.queue = list(responses)
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append(url)
        if not self.queue:
            raise AssertionError("FakeTransport queue exhausted")
        status, payload = self.queue.pop(0)
        return status, {}, json.dumps(payload).encode("utf-8")


def _client(responses, cache_dir=None):
    transport = FakeTransport(responses)
    client = twa.Client(token="test-token", cache_dir=cache_dir, opener=transport, max_retries=2)
    return client, transport


class TestClientRequiresToken(unittest.TestCase):
    def test_empty_token_raises(self):
        with self.assertRaises(ValueError):
            twa.Client(token="")


class TestListEndpoints(unittest.TestCase):
    def test_list_brands(self):
        client, _ = _client([(200, BRAND_LIST)])
        self.assertIn("Rolex", twa.list_brands(client))

    def test_list_models(self):
        client, transport = _client([(200, MODEL_LIST)])
        result = twa.list_models(client, "rolex")
        self.assertEqual(result, MODEL_LIST["data"])
        self.assertIn("brand=rolex", transport.calls[0])

    def test_list_references(self):
        client, _ = _client([(200, REFERENCE_LIST)])
        self.assertEqual(twa.list_references(client, "rolex"), REFERENCE_LIST["data"])


class TestSearchEndpoints(unittest.TestCase):
    def test_search_brands(self):
        client, _ = _client([(200, BRAND_SEARCH)])
        self.assertEqual(twa.search_brands(client, "rolex"), ["Rolex"])

    def test_search_references(self):
        client, _ = _client([(200, REFERENCE_SEARCH)])
        result = twa.search_references(client, "116520")
        self.assertEqual(result, [{"brand": "Rolex", "reference_number": "116520"}])

    def test_search_models_returns_full_records(self):
        client, _ = _client([(200, MODEL_SEARCH)])
        result = twa.search_models(client, "rolex daytona", search_attributes="model")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["case_diameter"], "40 mm")

    def test_search_models_omits_none_params_from_request(self):
        client, transport = _client([(200, MODEL_SEARCH)])
        twa.search_models(client, "daytona")
        self.assertNotIn("brand=", transport.calls[0])
        self.assertNotIn("case_material=", transport.calls[0])

    def test_enrich_reference_returns_first_match(self):
        client, _ = _client([(200, MODEL_SEARCH)])
        record = twa.enrich_reference(client, "Rolex", "116520")
        self.assertEqual(record["reference_number"], "116520")

    def test_enrich_reference_none_when_no_match(self):
        client, _ = _client([(200, {"data": []})])
        self.assertIsNone(twa.enrich_reference(client, "Rolex", "nonexistent"))


class TestYearAndCaseParsing(unittest.TestCase):
    def test_range(self):
        self.assertEqual(twa._parse_year_range("1989 - 2018"), (1989, 2018))

    def test_ongoing_production(self):
        self.assertEqual(twa._parse_year_range("2020 - Present"), (2020, None))

    def test_single_year(self):
        self.assertEqual(twa._parse_year_range("2020"), (2020, 2020))

    def test_none_input(self):
        self.assertEqual(twa._parse_year_range(None), (None, None))

    def test_case_diameter(self):
        self.assertEqual(twa._parse_case_mm("40 mm"), 40.0)
        self.assertEqual(twa._parse_case_mm("41.5mm"), 41.5)
        self.assertIsNone(twa._parse_case_mm(None))


class TestCatalogueConversion(unittest.TestCase):
    def test_to_catalogue_record_maps_documented_fields(self):
        record = twa.to_catalogue_record(MODEL_SEARCH["data"][0])
        self.assertEqual(record["reference"], "116520")
        self.assertEqual(record["brand"], "Rolex")
        self.assertEqual(record["model_name"], "Rolex Daytona")
        self.assertEqual(record["production_start"], 1989)
        self.assertEqual(record["production_end"], 2018)
        self.assertEqual(record["case_mm"], 40.0)

    def test_no_retail_price_field_is_ever_invented(self):
        """thewatchapi has no retail-price data; the adapter must not fabricate one."""
        record = twa.to_catalogue_record(MODEL_SEARCH["data"][0])
        self.assertNotIn("retail_price", record)

    def test_minimal_record_from_bare_reference(self):
        self.assertEqual(
            twa.minimal_catalogue_record("Rolex", "116520"),
            {"reference": "116520", "brand": "Rolex"},
        )

    def test_sync_brand_references_uses_the_cheap_endpoint_only(self):
        client, transport = _client([(200, REFERENCE_LIST)])
        records = twa.sync_brand_references(client, "rolex")
        self.assertEqual(len(records), len(REFERENCE_LIST["data"]))
        self.assertEqual(len(transport.calls), 1, "must be exactly one call, not one per reference")
        self.assertIn("/reference/list", transport.calls[0])

    def test_sync_result_feeds_the_catalogue_importer(self):
        client, _ = _client([(200, REFERENCE_LIST)])
        records = twa.sync_brand_references(client, "Rolex")
        conn = db.connect(":memory:")
        report = catalogue.upsert_catalogue(conn, records)
        self.assertEqual(report.refs_inserted, len(REFERENCE_LIST["data"]))
        conn.close()

    def test_enriched_record_feeds_the_catalogue_importer_with_richer_fields(self):
        conn = db.connect(":memory:")
        catalogue.upsert_catalogue(conn, [twa.to_catalogue_record(MODEL_SEARCH["data"][0])])
        row = conn.execute("SELECT model_name, case_mm FROM refs WHERE reference='116520'").fetchone()
        self.assertEqual(row["model_name"], "Rolex Daytona")
        self.assertEqual(row["case_mm"], 40.0)
        conn.close()


class TestHighUsageMarkers(unittest.TestCase):
    def test_flagged_endpoints_match_the_docs(self):
        self.assertEqual(
            twa.HIGH_USAGE_ENDPOINTS,
            {"/model/search", "/brand/price/history", "/model/price/history"},
        )

    def test_reference_price_history_is_not_flagged(self):
        """The docs mark reference/price/history as NOT high usage, unlike its siblings."""
        self.assertNotIn("/reference/price/history", twa.HIGH_USAGE_ENDPOINTS)


class TestPriceHistory(unittest.TestCase):
    def test_reference_price_history_parses(self):
        client, _ = _client([(200, REFERENCE_PRICE_HISTORY)])
        payload = twa.reference_price_history(client, "116520")
        self.assertEqual(payload["meta"]["reference_number"], "116520")
        self.assertEqual(len(payload["data"]), 2)

    def test_price_series_rows_rounds_to_cents_and_truncates_date(self):
        rows = list(twa.price_series_rows(
            "thewatchapi", "reference", "116520", REFERENCE_PRICE_HISTORY, "2026-08-12"
        ))
        self.assertEqual(rows[0], ("thewatchapi", "reference", "116520", "2023-10-13", 2465312, "USD", "2026-08-12"))

    def test_store_provider_price_series_round_trip(self):
        conn = db.connect(":memory:")
        n = ingest.store_provider_price_series(
            conn, "thewatchapi", "reference", "116520", REFERENCE_PRICE_HISTORY
        )
        self.assertEqual(n, 2)
        stored = conn.execute(
            "SELECT COUNT(*) FROM provider_price_series WHERE scope_value='116520'"
        ).fetchone()[0]
        self.assertEqual(stored, 2)
        conn.close()

    def test_price_series_never_touches_index_points(self):
        """Provider series and the hedonic index must stay in separate tables."""
        conn = db.connect(":memory:")
        ingest.store_provider_price_series(
            conn, "thewatchapi", "reference", "116520", REFERENCE_PRICE_HISTORY
        )
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM index_points").fetchone()[0], 0)
        conn.close()


class TestErrorHandling(unittest.TestCase):
    def test_invalid_token_raises_typed_error(self):
        client, _ = _client([(401, ERROR_INVALID_TOKEN)])
        with self.assertRaises(twa.ThewatchapiError) as ctx:
            twa.list_brands(client)
        self.assertEqual(ctx.exception.code, "invalid_api_token")
        self.assertEqual(ctx.exception.http_status, 401)

    def test_usage_limit_does_not_retry(self):
        client, transport = _client([(402, ERROR_USAGE_LIMIT), (200, BRAND_LIST)])
        with self.assertRaises(twa.ThewatchapiError):
            twa.list_brands(client)
        self.assertEqual(len(transport.calls), 1, "a 402 must not be retried")

    def test_rate_limit_retries_then_succeeds(self):
        client, transport = _client([
            (429, ERROR_RATE_LIMIT), (429, ERROR_RATE_LIMIT), (200, BRAND_LIST),
        ])
        # avoid real sleeping in the test
        import watchlab.sources.thewatchapi as twa_module
        original_sleep = twa_module.time.sleep
        twa_module.time.sleep = lambda *_: None
        try:
            result = twa.list_brands(client)
        finally:
            twa_module.time.sleep = original_sleep
        self.assertEqual(result, BRAND_LIST["data"])
        self.assertEqual(len(transport.calls), 3)

    def test_retries_exhausted_raises(self):
        client, transport = _client([(500, {"error": {}})] * 10)
        import watchlab.sources.thewatchapi as twa_module
        original_sleep = twa_module.time.sleep
        twa_module.time.sleep = lambda *_: None
        try:
            with self.assertRaises(twa.ThewatchapiError):
                twa.list_brands(client)
        finally:
            twa_module.time.sleep = original_sleep
        self.assertEqual(len(transport.calls), client.max_retries + 1)


class TestCaching(unittest.TestCase):
    def test_second_call_hits_cache_not_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, transport = _client([(200, BRAND_LIST)], cache_dir=tmp)
            twa.list_brands(client)
            twa.list_brands(client)
            self.assertEqual(len(transport.calls), 1, "second call should be served from cache")

    def test_token_is_not_part_of_the_cache_key(self):
        """Two clients with different tokens hitting the same query should share a cache file."""
        with tempfile.TemporaryDirectory() as tmp:
            transport = FakeTransport([(200, BRAND_LIST), (200, BRAND_LIST)])
            client_a = twa.Client(token="token-a", cache_dir=tmp, opener=transport)
            twa.list_brands(client_a)
            client_b = twa.Client(token="token-b", cache_dir=tmp, opener=transport)
            twa.list_brands(client_b)
            self.assertEqual(len(transport.calls), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
