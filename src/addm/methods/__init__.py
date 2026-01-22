"""Methods exports."""

from addm.methods.registry import MethodRegistry
from addm.methods.direct import DirectMethod
from addm.methods.cot import CoTMethod
from addm.methods.react import ReACTMethod
from addm.methods.rlm import RLMMethod
from addm.methods.rag import RAGMethod
from addm.methods.amos import AMOSMethod


def build_method_registry() -> MethodRegistry:
    registry = MethodRegistry()
    registry.register(DirectMethod)
    registry.register(CoTMethod)
    registry.register(ReACTMethod)
    registry.register(RLMMethod)
    registry.register(RAGMethod)
    registry.register(AMOSMethod)
    return registry
