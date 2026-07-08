import os
import tempfile
import unittest

from model4.cache import SQLiteCache
from model4.risk_assessor import RiskAssessor


class DummyBrandMapper:
    def __init__(self, mapping):
        self.mapping = mapping

    def resolve_brand(self, input_name):
        return self.mapping.get(input_name, {"generic_name": input_name})


class DummyFAERSClient:
    def __init__(self, payloads=None):
        self.payloads = payloads or {}
        self.calls = []

    def search_reports(self, medicine_name, limit=5):
        self.calls.append(medicine_name)
        return self.payloads.get(medicine_name, {"results": []})


class RiskAssessorTests(unittest.TestCase):
    def test_assess_uses_normalization_and_caches_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SQLiteCache(os.path.join(tmp_dir, "faers.sqlite3"))
            brand_mapper = DummyBrandMapper({"Tylenol": {"generic_name": "Acetaminophen"}})
            faers_client = DummyFAERSClient(
                {
                    "Acetaminophen": {
                        "results": [
                            {
                                "patient": {
                                    "reaction": [{"reactionmeddrapt": "Nausea"}, {"reactionmeddrapt": "Dizziness"}],
                                    "patientsex": "2",
                                    "patientonsetage": 35,
                                },
                                "serious": "1",
                                "seriousnessdeath": "1",
                            }
                        ]
                    }
                }
            )
            assessor = RiskAssessor(cache=cache, brand_mapper=brand_mapper, faers_client=faers_client)

            first = assessor.assess("Tylenol", age=35, gender="female")
            second = assessor.assess("tylenol", age=35, gender="female")

            self.assertEqual(first["normalized_medicine"], ["acetaminophen"])
            self.assertEqual(first["common_adverse_reactions"], ["Nausea", "Dizziness"])
            self.assertEqual(first["serious_adverse_reactions"], ["Nausea", "Dizziness"])
            self.assertTrue(first["cache_status"]["miss"])
            self.assertTrue(second["cache_status"]["hit"])
            self.assertEqual(len(faers_client.calls), 1)

    def test_assess_returns_empty_fallback_when_no_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SQLiteCache(os.path.join(tmp_dir, "faers.sqlite3"))
            assessor = RiskAssessor(
                cache=cache,
                brand_mapper=DummyBrandMapper({}),
                faers_client=DummyFAERSClient({}),
            )

            result = assessor.assess("Unknown Drug", age=40, gender="male")

            self.assertEqual(result["normalized_medicine"], ["unknown drug"])
            self.assertEqual(result["common_adverse_reactions"], [])
            self.assertEqual(result["serious_adverse_reactions"], [])
            self.assertEqual(result["reporting_frequency"], {"total_reports": 0, "sample_size": 0})

    def test_invalid_input_raises_clear_error(self):
        assessor = RiskAssessor(brand_mapper=DummyBrandMapper({}), faers_client=DummyFAERSClient({}))
        with self.assertRaisesRegex(ValueError, "Invalid medicine"):
            assessor.assess("   ", age=20, gender="male")


if __name__ == "__main__":
    unittest.main(verbosity=2)
