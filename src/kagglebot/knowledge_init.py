"""Backward-compatible import shim for the consolidated knowledge module."""

from __future__ import annotations

from kagglebot import knowledge as _knowledge
from kagglebot.knowledge import *  # noqa: F401,F403

__all__ = [name for name in dir(_knowledge) if not name.startswith("_")]


def __getattr__(name: str) -> object:
    return getattr(_knowledge, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_knowledge)))
