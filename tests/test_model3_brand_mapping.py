import os
import tempfile
import unittest

from model3.brand_mapper import BrandMapper
from model3.cache import SQLiteCache


class DummyOpenFDAClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def search_brand(self, brand_name):
        self.calls.append(brand_name)
        if brand_name not in self.payloads:
            return None
        return self.payloads[brand_name]


class BrandMapperTests(unittest.TestCase):
    def test_resolve_brand_returns_expected_schema_and_caches_result(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SQLiteCache(os.path.join(tmp_dir, "cache.sqlite3"))
            api_client = DummyOpenFDAClient(
                {
                    "Tylenol": {
                        "brand_name": "Tylenol",
                        "generic_name": "Acetaminophen",
                        "active_ingredients": ["Acetaminophen"],
                        "strength": "500 mg",
                        "dosage_form": "tablet",
                        "manufacturer_name": "Johnson & Johnson",
                        "smiles": None,
                        "product_ndc": "12345-678-90",
                    }
                }
            )

            mapper = BrandMapper(cache=cache, api_client=api_client)
            first = mapper.lookup("Tylenol")
            second = mapper.lookup("tylenol")

            self.assertEqual(first["input_name"], "Tylenol")
            self.assertEqual(first["brand_name"], "Tylenol")
            self.assertEqual(first["generic_name"], "Acetaminophen")
            self.assertEqual(first["active_ingredients"], [{"name": "Acetaminophen", "strength": ""}])
            self.assertEqual(first["source"], "openFDA")
            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            self.assertEqual(len(api_client.calls), 1)

    def test_resolve_brand_returns_fallback_when_no_result(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SQLiteCache(os.path.join(tmp_dir, "cache.sqlite3"))
            api_client = DummyOpenFDAClient({})
            mapper = BrandMapper(cache=cache, api_client=api_client)

            result = mapper.lookup("Unknown Brand")

            self.assertEqual(result["input_name"], "Unknown Brand")
            self.assertEqual(result["brand_name"], "")
            self.assertEqual(result["generic_name"], "")
            self.assertEqual(result["active_ingredients"], [])
            self.assertIsNone(result["smiles"])
            self.assertFalse(result["cached"])

    def test_empty_input_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = SQLiteCache(os.path.join(tmp_dir, "cache.sqlite3"))
            mapper = BrandMapper(cache=cache, api_client=DummyOpenFDAClient({}))
            with self.assertRaisesRegex(ValueError, "Invalid drug name"):
                mapper.lookup("   ")


if __name__ == "__main__":
    unittest.main(verbosity=2)
