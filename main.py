from addm.tasks.constants import ALL_TOPICS, ALL_POLICIES, GROUP_TOPICS

usage = """
ADDM - Adaptive Decision-Making with LLMs
==========================================

Run experiment:
  .venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 --dev --sample

Options:
  --policy POLICY   Policy ID (e.g., G1_allergy_V2) or comma-separated list
  --topic TOPIC     Run all V0-V3 for a topic (e.g., G1_allergy)
  --group GROUP     Run all policies in a group (e.g., G1)
  --dev             Dev mode (results/dev/)
  --sample          Use 3 verdict samples per policy
  -n N              Number of restaurants (default: 100)
  --k K             Reviews per restaurant (25/50/100/200)
"""

print(usage)

print("TOPICS (18):")
print("-" * 40)
for group, topics in GROUP_TOPICS.items():
    print(f"  {group}: {', '.join(t.split('_', 1)[1] for t in topics)}")

print(f"\nPOLICIES: {len(ALL_POLICIES)} total (18 topics x 4 variants V0-V3)")

# =============================================================================
# AMOS Phase 1 Hybrid Plan (2025-01-25)
# =============================================================================

AMOS_HYBRID_PLAN = """
================================================================================
AMOS PHASE 1: HYBRID APPROACH PLAN
================================================================================
Date: 2025-01-25
Decision: Pending - review and decide tomorrow

CONTEXT
-------
Compared two Phase 1 approaches:
1. Current (Parts): Query sections → 3 focused extractions → PolicyYAML → compile to Seed
2. Backup (OBSERVE→PLAN→ACT): Full analysis → strategy → direct seed generation

Paper's AMOS design emphasizes:
- Compilation metaphor (agenda → executable Formula Seed)
- Explicit, inspectable intermediate representations
- Deterministic transformation
- NOT opaque end-to-end generation

VERDICT: Current (Parts) approach is more aligned with the paper.
BUT: Backup's OBSERVE step has valuable semantic analysis we should adopt.

================================================================================
PROPOSED HYBRID: Add OBSERVE as Step 0
================================================================================

CURRENT FLOW:
    Query
      ↓ _parse_query_sections()     ← Deterministic regex parsing (brittle)
    [sections dict]
      ↓
    EXTRACT_TERMS(section)          ← Each LLM sees only its section
    EXTRACT_SCORING(section)
    EXTRACT_VERDICTS(section)
      ↓
    PolicyYAML → compile → Seed


HYBRID FLOW:
    Query
      ↓ OBSERVE(full query)         ← LLM analyzes entire agenda (NEW)
    [observations: policy_type, fields, signals, verdict_rules, ...]
      ↓
    EXTRACT_TERMS(section, observations)      ← Guided extraction
    EXTRACT_SCORING(section, observations)    ← Can skip if not scoring policy
    EXTRACT_VERDICTS(section, observations)   ← Knows rule order, preconditions
      ↓
    PolicyYAML → compile → Seed

================================================================================
WHAT OBSERVE REPLACES/ENHANCES
================================================================================

1. REPLACE regex section parsing (_parse_query_sections)
   - Current: Brittle regex splitting by "## " headers
   - OBSERVE: Semantic understanding of section boundaries
   - Benefit: Works even if markdown format varies

2. PRE-DETERMINE policy_type
   - Current: Discovered during extraction
   - OBSERVE: Known upfront (scoring / severity_rule_based / signal_rule_based)
   - Benefit: Skip scoring extraction for rule-based policies, adjust prompts

3. PROVIDE field inventory to extraction steps
   - Current: Extraction steps work blind
   - OBSERVE: Complete list of extraction_fields from agenda
   - Benefit: EXTRACT_TERMS knows what to look for, EXTRACT_VERDICTS knows valid fields
   - Reduces "field discovery" hacks in _discover_fields_from_rules()

4. GUIDE verdict rule interpretation
   - Current: Implicit ordering
   - OBSERVE: Explicit evaluation_order, preconditions, check_order
   - Benefit: Correct rule ordering in compiled seed

================================================================================
IMPLEMENTATION SKETCH
================================================================================

async def generate_formula_seed(agenda, policy_id, llm, ...):
    # Step 0: OBSERVE - Semantic analysis of full agenda (NEW)
    observations, obs_usage = await _observe(agenda, llm, policy_id)

    policy_type = observations.get("policy_type", "count_rule_based")
    known_fields = {f["name"] for f in observations.get("extraction_fields", [])}

    # Step 1: Extract terms (guided by observations)
    terms, usage1 = await _extract_terms_from_section(
        section=_get_terms_section(agenda, observations),  # smarter section finding
        llm=llm,
        policy_id=policy_id,
        expected_fields=known_fields,  # NEW: know what to look for
    )

    # Step 2: Extract scoring (skip if not scoring policy)
    if policy_type == "scoring":
        scoring, usage2 = await _extract_scoring_from_section(
            section=_get_scoring_section(agenda, observations),
            terms=terms,
            llm=llm,
            policy_id=policy_id,
            scoring_hints=observations.get("scoring_system"),  # NEW
        )
    else:
        scoring = {"policy_type": policy_type}
        usage2 = {}

    # Step 3: Extract verdicts (guided by observations)
    verdicts_data, usage3 = await _extract_verdicts_from_section(
        section=_get_verdicts_section(agenda, observations),
        terms=terms,
        scoring=scoring,
        llm=llm,
        policy_id=policy_id,
        verdict_hints=observations.get("verdict_rules"),  # NEW
    )

    # Rest unchanged: combine → validate → compile
    yaml_data = _combine_parts_to_yaml(terms, scoring, verdicts_data, ...)
    seed = compile_yaml_to_seed(yaml_data)
    return seed, total_usage

================================================================================
PROMPT MODIFICATIONS
================================================================================

EXTRACT_TERMS prompt addition:
```
## EXPECTED FIELDS (from agenda analysis)
The agenda defines these fields - make sure to extract all of them:
{expected_fields_list}

If you find additional fields not in this list, include them too.
```

EXTRACT_VERDICTS prompt addition:
```
## VERDICT STRUCTURE (from agenda analysis)
Policy type: {policy_type}
Evaluation order: {evaluation_order}
Expected verdicts: {verdict_labels}

Use this information to correctly order your rules.
```

================================================================================
TRADE-OFFS
================================================================================

| Aspect            | Without OBSERVE      | With OBSERVE            |
|-------------------|----------------------|-------------------------|
| LLM calls         | 3                    | 4 (one more)            |
| Token cost        | Lower                | +500-1000 tokens        |
| Section parsing   | Regex (brittle)      | Semantic (robust)       |
| Field coverage    | May miss fields      | Knows all fields upfront|
| Policy type       | Inferred             | Known upfront           |
| Verdict ordering  | Implicit             | Explicit guidance       |

================================================================================
FILES TO MODIFY
================================================================================

1. src/addm/methods/amos/phase1.py
   - Add _observe() function (copy from backup/phase1_backup.py)
   - Modify generate_formula_seed() to call OBSERVE first
   - Pass observations to extraction functions

2. src/addm/methods/amos/phase1_prompts.py
   - Add OBSERVE_PROMPT (copy from backup/phase1_prompts.py)
   - Modify EXTRACT_TERMS_PROMPT to accept expected_fields
   - Modify EXTRACT_VERDICTS_PROMPT to accept verdict_hints

3. src/addm/methods/amos/phase1_helpers.py
   - Add helper to extract sections using observations
   - _get_terms_section(agenda, observations)
   - _get_scoring_section(agenda, observations)
   - _get_verdicts_section(agenda, observations)

================================================================================
DECISION NEEDED
================================================================================

Options:
A) Implement full hybrid (OBSERVE + guided extraction)
B) Just add OBSERVE for policy_type detection, keep rest as-is
C) Keep current approach, don't add OBSERVE
D) Other refinements?

Run: python main.py to review this plan
================================================================================
"""

print("\n" + "=" * 80)
print(AMOS_HYBRID_PLAN)
