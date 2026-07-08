import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any


class SQLiteCache:
    def __init__(self, db_path: str | None = None, ttl_seconds: int = 86400) -> None:
        self.db_path = db_path or os.path.join(os.getcwd(), "model5", "cache.sqlite3")
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._initialize()

    def _initialize(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS symptom_cache (
                    lookup_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, key: str) -> Any | None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT payload, created_at FROM symptom_cache WHERE lookup_key = ?",
                    (key,),
                ).fetchone()
            if not row:
                return None
            payload, created_at = row
            if datetime.now() - datetime.fromisoformat(created_at) > timedelta(seconds=self.ttl_seconds):
                self.delete(key)
                return None
            return payload

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO symptom_cache (lookup_key, payload, created_at) VALUES (?, ?, ?)",
                    (key, value, datetime.now().isoformat()),
                )
                conn.commit()

    def delete(self, key: str) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM symptom_cache WHERE lookup_key = ?", (key,))
                conn.commit()
