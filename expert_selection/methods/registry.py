from __future__ import annotations

from collections.abc import Callable
from typing import Any


_REGISTRY: dict[str, Callable[[], Any]] = {}


def register(name: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    normalized = name.lower()

    def decorator(factory: Callable[[], Any]) -> Callable[[], Any]:
        if normalized in _REGISTRY:
            raise RuntimeError(f"Selection method {normalized!r} is already registered")
        _REGISTRY[normalized] = factory
        return factory

    return decorator


def _load_builtins() -> None:
    from . import gca, gmm, oia, slu  # noqa: F401


def registered_methods() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(_REGISTRY))


def build_methods(names: tuple[str, ...] | list[str]) -> list[Any]:
    _load_builtins()
    missing = [name for name in names if name not in _REGISTRY]
    if missing:
        raise ValueError(f"Unregistered selection methods: {missing}")
    return [_REGISTRY[name]() for name in names]

