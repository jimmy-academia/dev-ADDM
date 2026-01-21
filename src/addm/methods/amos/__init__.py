"""AMOS - Agenda-Driven Mining with Observable Steps.

Two-phase method for evaluating restaurants based on LLM-generated specifications:
- Phase 1: LLM "compiles" agenda into Formula Seed (cached per policy)
- Phase 2: Interpreter executes Formula Seed with optional adaptive mode

Phase 1 Approaches (for dynamic keyword discovery):
- PLAN_AND_ACT: Fixed 3-step pipeline (OBSERVE → PLAN → ACT)
- REACT: Iterative loop with actions (self-correcting)
- REFLEXION: Initial generation + quality analysis + revision

Search Strategy Features:
- Priority-based review ordering
- Early stopping when verdict is determinable
- Hybrid embedding retrieval when keywords are insufficient
"""

from addm.methods.amos.config import AMOSConfig, Phase1Approach
from addm.methods.amos.phase1 import generate_formula_seed, generate_formula_seed_with_config
from addm.methods.amos.phase2 import FormulaSeedInterpreter

__all__ = [
    "AMOSConfig",
    "Phase1Approach",
    "generate_formula_seed",
    "generate_formula_seed_with_config",
    "FormulaSeedInterpreter",
]
