"""AMOS - Agenda-Driven Mining with Observable Steps.

Two-phase method for evaluating restaurants based on LLM-generated specifications:
- Phase 1: LLM "compiles" agenda into Formula Seed (cached per policy)
- Phase 2: Interpreter executes Formula Seed with optional adaptive mode

Search Strategy Features:
- Priority-based review ordering
- Early stopping when verdict is determinable
- Hybrid embedding retrieval when keywords are insufficient
"""

from addm.methods.amos.config import AMOSConfig
from addm.methods.amos.phase1 import generate_formula_seed
from addm.methods.amos.phase2 import FormulaSeedInterpreter

__all__ = ["AMOSConfig", "generate_formula_seed", "FormulaSeedInterpreter"]
