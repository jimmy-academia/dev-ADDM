"""Methods exports."""

from addm.methods.registry import MethodRegistry
from addm.methods.direct import DirectMethod
from addm.methods.rlm import RLMMethod
from addm.methods.rag import RAGMethod


def build_method_registry() -> MethodRegistry:
    registry = MethodRegistry()
    registry.register(DirectMethod)
    registry.register(RLMMethod)
    registry.register(RAGMethod)
    return registry
