import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


def get_openfda_api_key() -> str:
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "OPENFDA_API_KEY":
                    return value.strip().strip('"').strip("'")
    return os.getenv("OPENFDA_API_KEY", "")


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip().lower()


def safe_get(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return ""
        current = current[key]
    return current


def as_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value)


def as_string_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]
