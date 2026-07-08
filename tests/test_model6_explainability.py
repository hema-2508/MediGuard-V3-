import os
import tempfile
import unittest

from model6.cache import SQLiteCache
from model6.explainability import ExplainabilityService


class DummyBrandMapper:
    def __init__(self, mapping):
        self.mapping = mapping

    def resolve_brand(self, input_name):
        return self.mapping.get(input_name, {"generic_name": input_name})


class DummyDailyMedClient:
    def __init__(self, payloads=None):
        self.payloads = payloads or {}
        self.calls = []

    def search_label(self, medicine_name):
        self.calls.append(medicine_name)
        return self.payloads.get(medicine_name, {"results": []})


class ExplainabilityServiceTests(unittest.TestCase):
    def test_explain_uses_normalization_and_caches_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SQLiteCache(os.path.join(tmp_dir, "explainability.sqlite3"))
            brand_mapper = DummyBrandMapper({"Tylenol": {"generic_name": "Acetaminophen"}})
            dailymed_client = DummyDailyMedClient(
                {
                    "Acetaminophen": {
                        "results": [
                            {
                                "title": "Acetaminophen Label",
                                "setid": "abc",
                                "spl_version": "1",
                                "adverse_reactions": ["Nausea", "Dizziness"],
                            }
                        ]
                    }
                }
            )
            service = ExplainabilityService(cache=cache, brand_mapper=brand_mapper, dailymed_client=dailymed_client)

            first = service.explain("Tylenol", {"symptom": "nausea"})
            second = service.explain("tylenol", {"symptom": "nausea"})

            self.assertIn("nausea", first["explanation"].lower())
            self.assertEqual(first["source"], "DailyMed")
            self.assertTrue(first["cache_status"]["miss"])
            self.assertTrue(second["cache_status"]["hit"])
            self.assertEqual(len(dailymed_client.calls), 1)

    def test_explain_returns_fallback_when_no_evidence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SQLiteCache(os.path.join(tmp_dir, "explainability.sqlite3"))
            service = ExplainabilityService(cache=cache, brand_mapper=DummyBrandMapper({}), dailymed_client=DummyDailyMedClient({}))
            result = service.explain("Unknown Drug", {"symptom": "headache"})
            self.assertIn("No direct DailyMed evidence", result["explanation"])
            self.assertEqual(result["confidence"], 0.3)

    def test_invalid_input_raises_clear_error(self):
        service = ExplainabilityService(brand_mapper=DummyBrandMapper({}), dailymed_client=DummyDailyMedClient({}))
        with self.assertRaisesRegex(ValueError, "Invalid medicine"):
            service.explain("   ", {"symptom": "nausea"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
