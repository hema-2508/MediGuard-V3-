import ast
import json
import logging
from typing import Any

from .api import OpenFDAClient
from .cache import SQLiteCache
from .models import BrandMappingResult
from .utils import as_string, as_string_list, normalize_name, safe_get


class BrandMapper:
    def __init__(self, cache: SQLiteCache | None = None, api_client: OpenFDAClient | None = None) -> None:
        self.cache = cache or SQLiteCache()
        self.api_client = api_client or OpenFDAClient()
        self.logger = logging.getLogger(__name__)

    def lookup(self, drug_name: str) -> dict[str, Any]:
        return self.resolve_brand(drug_name)

    def resolve_brand(self, input_name: str) -> dict[str, Any]:
        if not isinstance(input_name, str) or not input_name.strip():
            raise ValueError("Invalid drug name: input is empty.")

        normalized = normalize_name(input_name)
        cached_value = self.cache.get(normalized)
        if cached_value is not None:
            payload = json.loads(cached_value)
            payload["active_ingredients"] = self._normalize_active_ingredients(payload.get("active_ingredients", []))
            payload["cached"] = True
            return payload

        record = self.api_client.search_brand(input_name)
        if record is None:
            result = self._empty_result(input_name)
            result["cached"] = False
            return result

        result = self._build_result(input_name, record)
        self.cache.set(normalized, json.dumps(result))
        result["cached"] = False
        return result

    def _build_result(self, input_name: str, record: dict[str, Any]) -> dict[str, Any]:
        openfda = record.get("openfda", {}) if isinstance(record.get("openfda"), dict) else {}
        generic_name = self._extract_generic_name(record, openfda)
        active_ingredients = self._extract_active_ingredients(record, openfda)
        result = BrandMappingResult(
            input_name=input_name,
            brand_name=as_string(safe_get(record, "brand_name") or safe_get(openfda, "brand_name")),
            generic_name=generic_name,
            active_ingredients=active_ingredients,
            strength=as_string(safe_get(record, "strength") or safe_get(openfda, "strength")),
            dosage_form=as_string(safe_get(record, "dosage_form") or safe_get(openfda, "dosage_form")),
            manufacturer=as_string(safe_get(record, "manufacturer_name") or safe_get(openfda, "manufacturer_name")),
            smiles=as_string(safe_get(record, "smiles")) or None,
            ndc=as_string(safe_get(record, "product_ndc") or safe_get(record, "ndc") or safe_get(openfda, "product_ndc")),
            source="openFDA",
            cached=False,
        )
        return result.to_dict()

    def _extract_generic_name(self, record: dict[str, Any], openfda: dict[str, Any]) -> str:
        generic_name = as_string(safe_get(record, "generic_name") or safe_get(openfda, "generic_name"))
        if generic_name:
            return generic_name

        active_ingredients = self._extract_active_ingredients(record, openfda)
        if active_ingredients:
            return ", ".join(item.get("name", "") for item in active_ingredients if item.get("name"))
        return ""

    def _extract_active_ingredients(self, record: dict[str, Any], openfda: dict[str, Any]) -> list[dict[str, str]]:
        raw_values = safe_get(record, "active_ingredients") or safe_get(openfda, "active_ingredient")
        if isinstance(raw_values, list):
            items: list[dict[str, str]] = []
            for item in raw_values:
                if isinstance(item, dict):
                    name = as_string(item.get("name") or item.get("ingredient") or item.get("active_ingredient"))
                    strength = as_string(item.get("strength") or item.get("strength_value"))
                    if name or strength:
                        items.append({"name": name, "strength": strength})
                elif item:
                    items.append({"name": as_string(item), "strength": ""})
            return items
        if isinstance(raw_values, str):
            return [{"name": raw_values, "strength": ""}]
        return []

    def _normalize_active_ingredients(self, active_ingredients: Any) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        if not isinstance(active_ingredients, list):
            return normalized
        for item in active_ingredients:
            if isinstance(item, dict):
                normalized.append({"name": as_string(item.get("name")), "strength": as_string(item.get("strength"))})
                continue
            if isinstance(item, str):
                text = item.strip()
                if text.startswith("{") and text.endswith("}"):
                    try:
                        parsed = ast.literal_eval(text)
                    except (ValueError, SyntaxError):
                        parsed = None
                    if isinstance(parsed, dict):
                        normalized.append({"name": as_string(parsed.get("name")), "strength": as_string(parsed.get("strength"))})
                        continue
                normalized.append({"name": text, "strength": ""})
        return normalized

    def _empty_result(self, input_name: str) -> dict[str, Any]:
        return BrandMappingResult(
            input_name=input_name,
            brand_name="",
            generic_name="",
            active_ingredients=[],
            strength="",
            dosage_form="",
            manufacturer="",
            smiles=None,
            ndc="",
            source="openFDA",
            cached=False,
        ).to_dict()
