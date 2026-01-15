"""Methods exports."""

from addm.methods.registry import MethodRegistry
from addm.methods.direct import DirectMethod


def build_method_registry() -> MethodRegistry:
    registry = MethodRegistry()
    registry.register(DirectMethod)
    return registry
