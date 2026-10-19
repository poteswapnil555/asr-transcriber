"""Settings come from environment variables only. Nothing is read from files here. No keys are needed."""
from __future__ import annotations

import os


class ConfigError(Exception):
    """An environment variable has an unusable value. The message says which one."""


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return value if value else default


def env_float(name: str, default: float) -> float:
    raw = env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got '{raw}'") from None


def env_bool(name: str, default: bool = False) -> bool:
    raw = env(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}
