"""Methods exports."""

from addm.methods.registry import MethodRegistry
from addm.methods.direct import DirectMethod
from addm.methods.rlm import RLMMethod


def build_method_registry() -> MethodRegistry:
    registry = MethodRegistry()
    registry.register(DirectMethod)
    registry.register(RLMMethod)
    return registry
