import json
import logging
from typing import Any

from model3.brand_mapper import BrandMapper
from model3.utils import normalize_name

from .api import DailyMedClient
from .cache import SQLiteCache


class ExplainabilityService:
    """Generate human-readable explanations from medication risk results."""

    def __init__(
        self,
        cache: SQLiteCache | None = None,
        brand_mapper: BrandMapper | None = None,
        dailymed_client: DailyMedClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cache = cache or SQLiteCache()
        self.brand_mapper = brand_mapper or BrandMapper()
        self.dailymed_client = dailymed_client or DailyMedClient()
        self.logger = logger or logging.getLogger(__name__)

    def explain(self, medicine_name: str, result_payload: dict[str, Any]) -> dict[str, Any]:
        """Create a human-readable explanation backed by DailyMed evidence."""
        if not isinstance(medicine_name, str) or not medicine_name.strip():
            raise ValueError("Invalid medicine name: input is empty.")
        if not isinstance(result_payload, dict):
            raise ValueError("Invalid result payload: expected a dictionary.")

        normalized_input = normalize_name(medicine_name)
        brand_result = self.brand_mapper.resolve_brand(medicine_name)
        normalized_name = self._normalize_medicine_name(brand_result, medicine_name)
        query_name = self._select_query_name(brand_result, medicine_name)
        cache_key = self._cache_key(normalized_input, result_payload)
        cached_payload = self.cache.get(cache_key)
        if cached_payload is not None:
            payload = json.loads(cached_payload)
            payload["cache_status"] = {"hit": True, "miss": False}
            return payload

        label_payload = self.dailymed_client.search_label(query_name)
        evidence = self._extract_evidence(label_payload)
        explanation = self._build_explanation(medicine_name, normalized_name, result_payload, evidence)
        response = {
            "medicine": medicine_name,
            "normalized_medicine": normalized_name,
            "explanation": explanation,
            "evidence": evidence,
            "confidence": 0.7 if evidence else 0.3,
            "source": "DailyMed",
            "cache_status": {"hit": False, "miss": True},
        }
        self.cache.set(cache_key, json.dumps(response))
        return response

    def _normalize_medicine_name(self, brand_result: dict[str, Any], medicine_name: str) -> str:
        generic_name = brand_result.get("generic_name") or ""
        if isinstance(generic_name, str) and generic_name.strip():
            return normalize_name(generic_name)
        return normalize_name(medicine_name)

    def _cache_key(self, normalized_input: str, result_payload: dict[str, Any]) -> str:
        payload_text = json.dumps(result_payload, sort_keys=True)
        return f"{normalize_name(normalized_input)}:{payload_text}"

    def _select_query_name(self, brand_result: dict[str, Any], medicine_name: str) -> str:
        generic_name = brand_result.get("generic_name") or ""
        if isinstance(generic_name, str) and generic_name.strip():
            return generic_name.strip()
        return medicine_name.strip()

    def _extract_evidence(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        records = payload.get("results") or []
        evidence: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            adverse_reactions = record.get("adverse_reactions") or []
            if isinstance(adverse_reactions, list):
                evidence.append(
                    {
                        "title": record.get("title") or "DailyMed label",
                        "setid": record.get("setid") or "",
                        "spl_version": record.get("spl_version") or "",
                        "adverse_reactions": [str(item) for item in adverse_reactions if str(item).strip()],
                    }
                )
        return evidence

    def _build_explanation(
        self,
        medicine_name: str,
        normalized_name: str,
        result_payload: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> str:
        if evidence:
            summary = ", ".join(evidence[0].get("adverse_reactions", [])[:3])
            return (
                f"{medicine_name} was normalized to {normalized_name}. "
                f"The available evidence suggests these labeled concerns: {summary}."
            )
        return (
            f"No direct DailyMed evidence was found for {medicine_name}. "
            f"The explanation is based on the supplied result payload only."
        )
