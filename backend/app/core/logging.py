from __future__ import annotations

import logging
import sys
from typing import Any, Dict

try:  # Optional dependency: fall back to plain logging if unavailable.
    from pythonjsonlogger import jsonlogger

    _HAS_JSON_LOGGER = True
except Exception:  # pragma: no cover - import guard
    jsonlogger = None  # type: ignore[assignment]
    _HAS_JSON_LOGGER = False


if _HAS_JSON_LOGGER:

    class _JsonFormatter(jsonlogger.JsonFormatter):  # type: ignore[name-defined]
        def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
            super().add_fields(log_record, record, message_dict)
            if not log_record.get("level"):
                log_record["level"] = record.levelname
            if record.exc_info and not log_record.get("exc_info"):
                log_record["exc_info"] = self.formatException(record.exc_info)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    if _HAS_JSON_LOGGER:
        formatter: logging.Formatter = _JsonFormatter("%(asctime)s %(level)s %(name)s %(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy dependencies
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("faiss").setLevel(logging.WARNING)
