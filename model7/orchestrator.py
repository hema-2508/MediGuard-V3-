from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from typing import Any


class Model7Orchestrator:
    """Coordinate Model 2–6 into a single conversational response pipeline."""

    def __init__(
        self,
        extractor: Any = None,
        brand_mapper: Any = None,
        risk_assessor: Any = None,
        symptom_reasoner: Any = None,
        explainability_service: Any = None,
        cache_path: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.extractor = extractor
        self.brand_mapper = brand_mapper
        self.risk_assessor = risk_assessor
        self.symptom_reasoner = symptom_reasoner
        self.explainability_service = explainability_service
        self.logger = logger or logging.getLogger(__name__)
        self.cache_path = cache_path
        self._cache_connection = None
        self._last_error: str | None = None

    def orchestrate(self, input_text: str) -> dict[str, Any]:
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("Invalid input: input text is empty.")

        normalized_input = input_text.strip()
        cached_response = self._read_cache(normalized_input)
        if cached_response is not None:
            return cached_response

        try:
            extraction_result = self._extract_medicines(normalized_input)
            medicines = self._build_medicine_results(extraction_result, normalized_input)
            chat_response = self._build_chat_response(medicines)
            if not medicines and self._last_error:
                chat_response = f"I couldn't complete the analysis due to an error: {self._last_error}"
            response = {
                "input_text": normalized_input,
                "medicines": medicines,
                "chat_response": chat_response,
                "summary": self._build_summary(medicines),
            }
            self._write_cache(normalized_input, response)
            return response
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.exception("Model 7 orchestration failed")
            response = {
                "input_text": normalized_input,
                "medicines": [],
                "chat_response": f"I couldn't complete the analysis due to an error: {exc}",
                "summary": {"medicine_count": 0, "risk_count": 0},
            }
            self._write_cache(normalized_input, response)
            return response

    def _extract_medicines(self, input_text: str) -> list[dict[str, Any]]:
        if self.extractor is not None:
            try:
                payload = self.extractor.extract(input_text)
            except TypeError:
                payload = self.extractor.extract([input_text])
            except Exception as exc:
                self._last_error = str(exc)
                self.logger.warning("Medicine extraction failed: %s", exc)
                return []
            return self._normalize_extraction_payload(payload)

        try:
            from model2.inference import MedicineExtractor

            extractor = MedicineExtractor()
            self.extractor = extractor
            payload = extractor.extract(input_text)
            return self._normalize_extraction_payload(payload)
        except Exception as exc:  # pragma: no cover - import or runtime dependency failure
            self._last_error = str(exc)
            self.logger.warning("Falling back to empty extraction because Model 2 is unavailable: %s", exc)
            return []

    def _normalize_extraction_payload(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            medicines = payload.get("medicines") or payload.get("results") or []
            if isinstance(medicines, list):
                return [item for item in medicines if isinstance(item, dict)]
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _build_medicine_results(self, extraction_result: list[dict[str, Any]], input_text: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for entry in extraction_result:
            medicine_name = self._coerce_text(entry.get("name") or entry.get("medicine") or "")
            if not medicine_name:
                continue

            try:
                brand_mapping = self._resolve_brand_mapping(medicine_name)
            except Exception as exc:
                self.logger.warning("Brand mapping failed for %s: %s", medicine_name, exc)
                brand_mapping = {"generic_name": medicine_name, "cached": False}

            try:
                risk_summary = self._assess_risk(medicine_name)
            except Exception as exc:
                self.logger.warning("Risk assessment failed for %s: %s", medicine_name, exc)
                risk_summary = {"common_adverse_reactions": [], "confidence": {"confidence_score": 0.0}}

            try:
                symptom = self._infer_symptom(input_text, medicine_name)
                symptom_reasoning = self._reason_about_symptoms(medicine_name, symptom)
            except Exception as exc:
                self.logger.warning("Symptom reasoning failed for %s: %s", medicine_name, exc)
                symptom_reasoning = {"known_adverse_effect": False, "severity": "unknown"}

            try:
                explanation = self._explain_result(medicine_name, risk_summary)
            except Exception as exc:
                self.logger.warning("Explanation generation failed for %s: %s", medicine_name, exc)
                explanation = {"explanation": f"Unable to explain {medicine_name} due to an error.", "confidence": 0.0}

            results.append(
                {
                    "name": medicine_name,
                    "strength": self._coerce_text(entry.get("strength") or entry.get("dosage") or ""),
                    "confidence": self._coerce_float(entry.get("confidence") or 0.0),
                    "brand_mapping": brand_mapping,
                    "risk_summary": risk_summary,
                    "symptom_reasoning": symptom_reasoning,
                    "explanation": explanation,
                }
            )

        return results

    def _resolve_brand_mapping(self, medicine_name: str) -> dict[str, Any]:
        if self.brand_mapper is not None:
            return self.brand_mapper.resolve_brand(medicine_name)

        try:
            from model3.brand_mapper import BrandMapper

            self.brand_mapper = BrandMapper()
            return self.brand_mapper.resolve_brand(medicine_name)
        except Exception as exc:  # pragma: no cover - import or runtime dependency failure
            self.logger.warning("Model 3 is unavailable: %s", exc)
            return {"generic_name": medicine_name, "cached": False}

    def _assess_risk(self, medicine_name: str) -> dict[str, Any]:
        if self.risk_assessor is not None:
            return self.risk_assessor.assess(medicine_name)

        try:
            from model4.risk_assessor import RiskAssessor

            self.risk_assessor = RiskAssessor()
            return self.risk_assessor.assess(medicine_name)
        except Exception as exc:  # pragma: no cover - import or runtime dependency failure
            self.logger.warning("Model 4 is unavailable: %s", exc)
            return {"common_adverse_reactions": [], "confidence": {"confidence_score": 0.0}}

    def _reason_about_symptoms(self, medicine_name: str, symptom: str) -> dict[str, Any]:
        if self.symptom_reasoner is not None:
            return self.symptom_reasoner.analyze(medicine_name, symptom)

        try:
            from model5.symptom_reasoner import SymptomReasoner

            self.symptom_reasoner = SymptomReasoner()
            return self.symptom_reasoner.analyze(medicine_name, symptom)
        except Exception as exc:  # pragma: no cover - import or runtime dependency failure
            self.logger.warning("Model 5 is unavailable: %s", exc)
            return {"known_adverse_effect": False, "severity": "unknown"}

    def _explain_result(self, medicine_name: str, risk_summary: dict[str, Any]) -> dict[str, Any]:
        if self.explainability_service is not None:
            return self.explainability_service.explain(medicine_name, risk_summary)

        try:
            from model6.explainability import ExplainabilityService

            self.explainability_service = ExplainabilityService()
            return self.explainability_service.explain(medicine_name, risk_summary)
        except Exception as exc:  # pragma: no cover - import or runtime dependency failure
            self.logger.warning("Model 6 is unavailable: %s", exc)
            return {"explanation": f"No additional explanation available for {medicine_name}.", "confidence": 0.0}

    def _infer_symptom(self, input_text: str, medicine_name: str) -> str:
        text = (input_text or "").lower()
        if "nausea" in text or "vomit" in text:
            return "nausea"
        if "pain" in text:
            return "pain"
        if "rash" in text:
            return "rash"
        if "headache" in text:
            return "headache"
        return f"side effect for {medicine_name}".strip()

    def _build_chat_response(self, medicines: list[dict[str, Any]]) -> str:
        if not medicines:
            return "I couldn't identify any medicines to analyze from the provided text."

        parts = []
        for medicine in medicines:
            risk = medicine.get("risk_summary", {}).get("common_adverse_reactions") or []
            risk_text = ", ".join(risk) if risk else "no obvious adverse reactions"
            parts.append(f"{medicine['name']} may involve {risk_text}.")
        return " ".join(parts)

    def _build_summary(self, medicines: list[dict[str, Any]]) -> dict[str, Any]:
        risk_count = sum(1 for medicine in medicines if (medicine.get("risk_summary", {}).get("common_adverse_reactions") or []))
        return {"medicine_count": len(medicines), "risk_count": risk_count}

    def _read_cache(self, input_text: str) -> dict[str, Any] | None:
        if not self.cache_path:
            return None
        try:
            connection = self._get_cache_connection()
            row = connection.execute(
                "SELECT payload FROM orchestrations WHERE input_text = ?",
                (input_text,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])
        except Exception as exc:  # pragma: no cover - safety fallback
            self.logger.warning("Cache read failed: %s", exc)
            return None

    def _write_cache(self, input_text: str, payload: dict[str, Any]) -> None:
        if not self.cache_path:
            return
        try:
            connection = self._get_cache_connection()
            connection.execute(
                "CREATE TABLE IF NOT EXISTS orchestrations (input_text TEXT PRIMARY KEY, payload TEXT NOT NULL)",
            )
            connection.execute(
                "INSERT OR REPLACE INTO orchestrations (input_text, payload) VALUES (?, ?)",
                (input_text, json.dumps(payload, sort_keys=True)),
            )
            connection.commit()
        except Exception as exc:  # pragma: no cover - safety fallback
            self.logger.warning("Cache write failed: %s", exc)

    def _get_cache_connection(self) -> sqlite3.Connection:
        if self._cache_connection is None:
            directory = os.path.dirname(self.cache_path) or "."
            os.makedirs(directory, exist_ok=True)
            self._cache_connection = sqlite3.connect(self.cache_path)
        return self._cache_connection

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
