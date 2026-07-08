import tempfile
import unittest

from model7.orchestrator import Model7Orchestrator


class DummyExtractor:
    def __init__(self, payloads=None):
        self.payloads = payloads or []
        self.calls = []

    def extract(self, text):
        self.calls.append(text)
        return self.payloads.pop(0) if self.payloads else []


class DummyBrandMapper:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def resolve_brand(self, input_name):
        return self.mapping.get(input_name, {"generic_name": input_name})


class DummyRiskAssessor:
    def __init__(self, payloads=None):
        self.payloads = payloads or []
        self.calls = []

    def assess(self, medicine_name, **kwargs):
        self.calls.append((medicine_name, kwargs))
        return self.payloads.pop(0) if self.payloads else {"common_adverse_reactions": []}


class DummySymptomReasoner:
    def __init__(self, payloads=None):
        self.payloads = payloads or []
        self.calls = []

    def analyze(self, medicine_name, symptom):
        self.calls.append((medicine_name, symptom))
        return self.payloads.pop(0) if self.payloads else {"known_adverse_effect": False}


class DummyExplainabilityService:
    def __init__(self, payloads=None):
        self.payloads = payloads or []
        self.calls = []

    def explain(self, medicine_name, result_payload):
        self.calls.append((medicine_name, result_payload))
        return self.payloads.pop(0) if self.payloads else {"explanation": "fallback explanation"}


class Model7OrchestratorTests(unittest.TestCase):
    def test_orchestrate_builds_unified_response(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            extractor = DummyExtractor(
                [
                    [
                        {"name": "Amoxicillin", "strength": "500mg", "confidence": 0.9},
                        {"name": "Ibuprofen", "strength": "200mg", "confidence": 0.8},
                    ]
                ]
            )
            brand_mapper = DummyBrandMapper(
                {
                    "Amoxicillin": {"generic_name": "Amoxicillin"},
                    "Ibuprofen": {"generic_name": "Ibuprofen"},
                }
            )
            risk_assessor = DummyRiskAssessor(
                [
                    {"common_adverse_reactions": ["Nausea"], "confidence": {"confidence_score": 0.7}},
                    {"common_adverse_reactions": ["Headache"], "confidence": {"confidence_score": 0.6}},
                ]
            )
            symptom_reasoner = DummySymptomReasoner([
                {"known_adverse_effect": True, "severity": "moderate"},
                {"known_adverse_effect": False, "severity": "unknown"},
            ])
            explainability_service = DummyExplainabilityService(
                [
                    {"explanation": "Amoxicillin explanation", "confidence": 0.8},
                    {"explanation": "Ibuprofen explanation", "confidence": 0.7},
                ]
            )

            orchestrator = Model7Orchestrator(
                extractor=extractor,
                brand_mapper=brand_mapper,
                risk_assessor=risk_assessor,
                symptom_reasoner=symptom_reasoner,
                explainability_service=explainability_service,
                cache_path=tmp_dir + "/model7.sqlite3",
            )

            response = orchestrator.orchestrate("Take Amoxicillin and Ibuprofen for pain")

            self.assertEqual(response["input_text"], "Take Amoxicillin and Ibuprofen for pain")
            self.assertEqual(len(response["medicines"]), 2)
            self.assertEqual(response["medicines"][0]["name"], "Amoxicillin")
            self.assertEqual(response["medicines"][0]["risk_summary"]["common_adverse_reactions"], ["Nausea"])
            self.assertTrue(response["medicines"][0]["symptom_reasoning"]["known_adverse_effect"])
            self.assertIn("Amoxicillin explanation", response["medicines"][0]["explanation"]["explanation"])
            self.assertIn("chat_response", response)
            self.assertIn("summary", response)

    def test_orchestrate_handles_empty_input(self):
        orchestrator = Model7Orchestrator()
        with self.assertRaisesRegex(ValueError, "Invalid input"):
            orchestrator.orchestrate("   ")

    def test_orchestrate_handles_dependency_failure(self):
        class FailingExtractor:
            def extract(self, text):
                raise RuntimeError("boom")

        orchestrator = Model7Orchestrator(extractor=FailingExtractor())
        response = orchestrator.orchestrate("Take aspirin")

        self.assertEqual(response["medicines"], [])
        self.assertIn("error", response["chat_response"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
