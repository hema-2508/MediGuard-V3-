import json
import logging
from typing import Any

import requests

from .utils import get_openfda_api_key, logger


class OpenFDAClient:
    def __init__(self, api_key: str | None = None, timeout: int = 10) -> None:
        self.api_key = api_key or get_openfda_api_key()
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)

    def search_brand(self, brand_name: str) -> dict[str, Any] | None:
        if not isinstance(brand_name, str) or not brand_name.strip():
            raise ValueError("Invalid drug name: input is empty.")

        endpoint = "https://api.fda.gov/drug/ndc.json"
        params: dict[str, Any] = {"search": f"brand_name:{brand_name.strip()}", "limit": 1}
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            response = requests.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results") or []
            if not results:
                return None
            return results[0]
        except requests.RequestException as exc:
            self.logger.warning("openFDA lookup failed for %s: %s", brand_name, exc)
            return None
        except ValueError as exc:
            self.logger.warning("openFDA returned invalid JSON for %s: %s", brand_name, exc)
            return None
