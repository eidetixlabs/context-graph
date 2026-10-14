"""Read standard library logging level from LOG_LEVEL (or another env var)."""

from __future__ import annotations

import logging
import os


def log_level_from_env(var: str = "LOG_LEVEL", default: str = "INFO") -> int:
    name = (os.environ.get(var) or default).upper()
    return getattr(logging, name, logging.INFO)
