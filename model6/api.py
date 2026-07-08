import logging
from typing import Any

import requests


class DailyMedClient:
    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def search_label(self, medicine_name: str) -> dict[str, Any]:
        if not isinstance(medicine_name, str) or not medicine_name.strip():
            raise ValueError("Invalid medicine name: input is empty.")

        endpoint = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
        params = {"search": medicine_name.strip(), "limit": 5}
        try:
            response = requests.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {"results": []}
        except requests.RequestException as exc:
            self.logger.warning("DailyMed lookup failed for %s: %s", medicine_name, exc)
            return {"results": []}
        except ValueError as exc:
            self.logger.warning("DailyMed returned invalid JSON for %s: %s", medicine_name, exc)
            return {"results": []}
