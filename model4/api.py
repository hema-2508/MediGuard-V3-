import logging
from typing import Any

import requests


class FAERSClient:
    def __init__(self, api_key: str | None = None, timeout: int = 10) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def search_reports(self, medicine_name: str, limit: int = 5) -> dict[str, Any]:
        if not isinstance(medicine_name, str) or not medicine_name.strip():
            raise ValueError("Invalid medicine name: input is empty.")

        endpoint = "https://api.fda.gov/drug/event.json"
        params: dict[str, Any] = {"search": f"patient.drug.medicinalproduct:{medicine_name.strip()}", "limit": limit}
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            response = requests.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            self.logger.warning("FAERS lookup failed for %s: %s", medicine_name, exc)
            return {"results": []}
        except ValueError as exc:
            self.logger.warning("FAERS returned invalid JSON for %s: %s", medicine_name, exc)
            return {"results": []}
