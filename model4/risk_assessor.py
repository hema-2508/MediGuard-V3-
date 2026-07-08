import json
import logging
from collections import Counter
from typing import Any

from model3.brand_mapper import BrandMapper
from model3.utils import normalize_name

from .api import FAERSClient
from .cache import SQLiteCache


class RiskAssessor:
    """Assess medication-related adverse event risk using FAERS data."""

    def __init__(
        self,
        cache: SQLiteCache | None = None,
        brand_mapper: BrandMapper | None = None,
        faers_client: FAERSClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cache = cache or SQLiteCache()
        self.brand_mapper = brand_mapper or BrandMapper()
        self.faers_client = faers_client or FAERSClient()
        self.logger = logger or logging.getLogger(__name__)

    def assess(
        self,
        medicine_name: str,
        *,
        age: int | None = None,
        gender: str | None = None,
        conditions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a risk summary for a medicine using normalized names and FAERS reports."""
        if not isinstance(medicine_name, str) or not medicine_name.strip():
            raise ValueError("Invalid medicine name: input is empty.")

        normalized_input = normalize_name(medicine_name)
        brand_result = self.brand_mapper.resolve_brand(medicine_name)
        normalized_medicines = self._normalize_medicines(brand_result, medicine_name)
        query_name = self._select_query_name(brand_result, medicine_name)

        cache_key = self._cache_key(normalized_input, age, gender, conditions)
        cached_payload = self.cache.get(cache_key)
        if cached_payload is not None:
            payload = json.loads(cached_payload)
            payload["cache_status"] = {"hit": True, "miss": False}
            return payload

        faers_payload = self.faers_client.search_reports(query_name, limit=5)
        results = faers_payload.get("results") or []
        reactions = self._collect_reactions(results)
        serious_reactions = self._collect_serious_reactions(results)
        reporting_frequency = self._compute_reporting_frequency(results)
        observations = self._collect_observations(results, age=age, gender=gender)

        response = {
            "input_medicine": medicine_name,
            "normalized_medicine": normalized_medicines,
            "common_adverse_reactions": reactions,
            "serious_adverse_reactions": serious_reactions,
            "reporting_frequency": reporting_frequency,
            "age_gender_observations": observations,
            "confidence": {
                "source": "openFDA FAERS",
                "normalized_via": "Model 3 BrandMapper",
                "confidence_score": 0.7 if results else 0.3,
            },
            "cache_status": {"hit": False, "miss": True},
        }
        self.cache.set(cache_key, json.dumps(response))
        return response

    def _normalize_medicines(self, brand_result: dict[str, Any], medicine_name: str) -> list[str]:
        generic_name = brand_result.get("generic_name") or ""
        if isinstance(generic_name, str) and generic_name.strip():
            return [normalize_name(generic_name)]
        normalized_input = normalize_name(medicine_name)
        return [normalized_input] if normalized_input else []

    def _cache_key(
        self,
        normalized_input: str,
        age: int | None,
        gender: str | None,
        conditions: list[str] | None,
    ) -> str:
        conditions_text = "|".join(sorted((conditions or [])))
        return f"{normalize_name(normalized_input)}:{age or ''}:{(gender or '').lower()}:{conditions_text}"
    def _select_query_name(self, brand_result: dict[str, Any], medicine_name: str) -> str:
        generic_name = brand_result.get("generic_name") or ""
        if isinstance(generic_name, str) and generic_name.strip():
            return generic_name.strip()
        return medicine_name.strip()

    def _collect_reactions(self, results: list[dict[str, Any]]) -> list[str]:
        reaction_counter: Counter[str] = Counter()
        for result in results:
            patient = result.get("patient") or {}
            reactions = patient.get("reaction") or []
            for item in reactions:
                if isinstance(item, dict):
                    name = item.get("reactionmeddrapt") or item.get("reactionmeddraversionpt") or ""
                    if name:
                        reaction_counter[str(name)] += 1
        return [name for name, _ in reaction_counter.most_common(5)]

    def _collect_serious_reactions(self, results: list[dict[str, Any]]) -> list[str]:
        serious_names: list[str] = []
        for result in results:
            patient = result.get("patient") or {}
            reactions = patient.get("reaction") or []
            if result.get("serious") == "1" or result.get("seriousnessdeath") == "1":
                for item in reactions:
                    if isinstance(item, dict):
                        name = item.get("reactionmeddrapt") or item.get("reactionmeddraversionpt") or ""
                        if name and name not in serious_names:
                            serious_names.append(str(name))
        return serious_names[:5]

    def _compute_reporting_frequency(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        return {"total_reports": len(results), "sample_size": len(results)}

    def _collect_observations(self, results: list[dict[str, Any]], *, age: int | None, gender: str | None) -> list[str]:
        observations: list[str] = []
        if age is not None:
            observations.append(f"age {age} reported in FAERS sample")
        if gender:
            observations.append(f"gender {gender.lower()} considered")
        if not observations and results:
            observations.append("demographic observations unavailable")
        return observations
