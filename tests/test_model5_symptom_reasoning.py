import os
import tempfile
import unittest

from model5.cache import SQLiteCache
from model5.symptom_reasoner import SymptomReasoner


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


class SymptomReasonerTests(unittest.TestCase):
    def test_analyze_uses_normalization_and_caches_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SQLiteCache(os.path.join(tmp_dir, "dailymed.sqlite3"))
            brand_mapper = DummyBrandMapper({"Tylenol": {"generic_name": "Acetaminophen"}})
            dailymed_client = DummyDailyMedClient(
                {
                    "Acetaminophen": {
                        "results": [
                            {
                                "setid": "abc",
                                "title": "Acetaminophen Label",
                                "spl_version": "1",
                                "adverse_reactions": ["Nausea", "Dizziness"],
                            }
                        ]
                    }
                }
            )
            reasoner = SymptomReasoner(cache=cache, brand_mapper=brand_mapper, dailymed_client=dailymed_client)

            first = reasoner.analyze("Tylenol", "nausea")
            second = reasoner.analyze("tylenol", "Nausea")

            self.assertTrue(first["known_adverse_effect"])
            self.assertEqual(first["severity"], "unknown")
            self.assertEqual(first["source"], "DailyMed")
            self.assertTrue(first["cache_status"]["miss"])
            self.assertTrue(second["cache_status"]["hit"])
            self.assertEqual(len(dailymed_client.calls), 1)

    def test_analyze_returns_fallback_when_no_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SQLiteCache(os.path.join(tmp_dir, "dailymed.sqlite3"))
            reasoner = SymptomReasoner(cache=cache, brand_mapper=DummyBrandMapper({}), dailymed_client=DummyDailyMedClient({}))
            result = reasoner.analyze("Unknown Drug", "headache")
            self.assertFalse(result["known_adverse_effect"])
            self.assertEqual(result["severity"], "unknown")
            self.assertEqual(result["evidence"], [])

    def test_invalid_input_raises_clear_error(self):
        reasoner = SymptomReasoner(brand_mapper=DummyBrandMapper({}), dailymed_client=DummyDailyMedClient({}))
        with self.assertRaisesRegex(ValueError, "Invalid medicine"):
            reasoner.analyze("   ", "nausea")


if __name__ == "__main__":
    unittest.main(verbosity=2)
