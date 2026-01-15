"""Method registry."""

from typing import Dict, Type

from addm.methods.base import Method


class MethodRegistry:
    def __init__(self) -> None:
        self._methods: Dict[str, Type[Method]] = {}

    def register(self, method: Type[Method]) -> None:
        self._methods[method.name] = method

    def get(self, name: str) -> Type[Method]:
        if name not in self._methods:
            raise KeyError(f"Unknown method: {name}")
        return self._methods[name]

    def names(self) -> list[str]:
        return sorted(self._methods.keys())
