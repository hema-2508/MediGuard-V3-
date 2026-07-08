import json
import logging
from typing import Any

from model3.brand_mapper import BrandMapper
from model3.utils import normalize_name

from .api import DailyMedClient
from .cache import SQLiteCache


class SymptomReasoner:
    """Reason about whether a symptom is a known medication adverse effect."""

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

    def analyze(self, medicine_name: str, symptom: str) -> dict[str, Any]:
        """Return an adverse-effect assessment for a medicine and symptom."""
        if not isinstance(medicine_name, str) or not medicine_name.strip():
            raise ValueError("Invalid medicine name: input is empty.")
        if not isinstance(symptom, str) or not symptom.strip():
            raise ValueError("Invalid symptom: input is empty.")

        normalized_input = normalize_name(medicine_name)
        brand_result = self.brand_mapper.resolve_brand(medicine_name)
        normalized_name = self._normalize_medicine_name(brand_result, medicine_name)
        query_name = self._select_query_name(brand_result, medicine_name)
        cache_key = self._cache_key(normalized_input, symptom)
        cached_payload = self.cache.get(cache_key)
        if cached_payload is not None:
            payload = json.loads(cached_payload)
            payload["cache_status"] = {"hit": True, "miss": False}
            return payload

        label_payload = self.dailymed_client.search_label(query_name)
        evidence = self._extract_evidence(label_payload)
        symptom_match = self._match_symptom(symptom, evidence)
        response = {
            "medicine": medicine_name,
            "normalized_medicine": normalized_name,
            "reported_symptom": symptom.strip(),
            "known_adverse_effect": symptom_match["known"],
            "severity": symptom_match["severity"],
            "evidence": evidence,
            "source": "DailyMed",
            "confidence": 0.7 if evidence else 0.3,
            "cache_status": {"hit": False, "miss": True},
        }
        self.cache.set(cache_key, json.dumps(response))
        return response

    def _normalize_medicine_name(self, brand_result: dict[str, Any], medicine_name: str) -> str:
        generic_name = brand_result.get("generic_name") or ""
        if isinstance(generic_name, str) and generic_name.strip():
            return normalize_name(generic_name)
        return normalize_name(medicine_name)

    def _cache_key(self, normalized_input: str, symptom: str) -> str:
        return f"{normalize_name(normalized_input)}:{normalize_name(symptom)}"

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

    def _match_symptom(self, symptom: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        normalized_symptom = normalize_name(symptom)
        for item in evidence:
            for reaction in item.get("adverse_reactions", []) or []:
                normalized_reaction = normalize_name(reaction)
                if normalized_reaction == normalized_symptom or normalized_symptom in normalized_reaction or normalized_reaction in normalized_symptom:
                    return {"known": True, "severity": "unknown"}
        return {"known": False, "severity": "unknown"}
