"""Phase 1 Prompts: OBSERVE, PLAN, ACT prompts for Formula Seed generation.

Prompts for the Plan-and-Act approach to Formula Seed generation.
Separated from logic for maintainability and easier prompt iteration.
"""

# =============================================================================
# Step 1: OBSERVE - Query Analysis (NO domain hints)
# =============================================================================

OBSERVE_PROMPT = '''Analyze this task agenda and extract EXACTLY what it specifies.

## TASK AGENDA

{agenda}

## YOUR JOB

Read the agenda carefully. There are THREE types of verdict logic - identify which one this agenda uses:

### TYPE A: SCORING-BASED (uses point values)
The agenda specifies point values for categories (e.g., "Severe: 15 points, Moderate: 5 points").
Verdict is determined by total score vs thresholds (e.g., "Critical if score >= 8").

### TYPE B: SEVERITY-RULE-BASED (uses severity count thresholds)
The agenda specifies count thresholds by SEVERITY LEVEL (e.g., "Critical if 2+ severe incidents").
NO point values. Categories are severity levels (mild/moderate/severe or similar).

### TYPE C: SIGNAL-RULE-BASED (uses signal detection or quality assessment)
The agenda specifies verdict rules based on detecting SPECIFIC SIGNALS or QUALITY LEVELS in reviews.

**Pattern 1 - Signal phrases:**
- "Recommended if <positive_signal> mentioned" (signal names vary by policy type!)
- "Not Recommended if <negative_signal> described" (read actual signals from agenda!)

**Pattern 2 - Quality assessment (IMPORTANT - this is NOT severity-rule-based!):**
When the outcome field represents SERVICE QUALITY on a scale from negative to positive:
- Values like: Absent, Poor, Adequate, Good, Excellent
- OR: Rude, Cold, Neutral, Friendly, Warm
- Verdict rules say things like "Excellent if multiple reviews PRAISE service" or "Needs Improvement if servers were INATTENTIVE"

**KEY DISTINCTION:**
- SEVERITY-rule-based: "2+ severe incidents → Critical" (counting bad things, more = worse)
- SIGNAL/QUALITY-based: "Multiple positive observations → Excellent" (good observations lead to good verdict)

If the agenda says "Excellent verdict when reviews praise X" → this is SIGNAL-RULE-BASED!
The quality values (Excellent, Good, Poor, Absent) ARE the signals to detect.

## WHAT TO EXTRACT

### 1. Policy Type
Determine:
- "scoring" if point values are given
- "severity_rule_based" if counting by severity levels (mild/moderate/severe)
- "signal_rule_based" if detecting specific signals/phrases (READ SIGNAL NAMES FROM AGENDA!)

### 2. Categories OR Signals
- For SCORING/SEVERITY: What severity categories exist? (mild/moderate/severe or similar)
- For SIGNAL-BASED: What signals to detect? (READ FROM AGENDA's "## Definitions of Terms" - signals are policy-specific!)

### 3. For SCORING-BASED policies:
- Extract EXACT point values for each category (can be positive or negative)
- Extract modifier point adjustments

### 4. For SEVERITY-RULE-BASED policies:
- Extract count threshold rules by severity (e.g., "2+ severe incidents" → "Critical")

### 5. For SIGNAL-RULE-BASED policies (includes quality assessment!):
- List ALL positive signals FROM THE AGENDA that lead to favorable verdict
- List ALL negative signals FROM THE AGENDA that lead to unfavorable verdict
- Signals become enum values for extraction - use ONLY signal names defined in the agenda!

**FOR QUALITY ASSESSMENT patterns:**
If the outcome field has quality values (Absent/Poor/Adequate/Good/Excellent), map them to signals:
- Positive signals: values indicating good quality (Excellent, Good, Warm, Friendly)
- Negative signals: values indicating poor quality (Absent, Poor, Rude, Cold)
- Neutral signals: middle values (Adequate, Neutral)

### 6. Verdict Rules
- For SCORING: thresholds on total score
- For SEVERITY-RULE: count conditions by severity
- For SIGNAL-RULE: presence of positive/negative signals

### 7. ALL Definition Fields (CRITICAL!)
The agenda has a "## Definitions of Terms" section with multiple "### <Term>" subsections.
You MUST extract EVERY "### <Term>" as a separate extraction field:
- Each "### <Term>" becomes a field (e.g., "### Staff Response" → STAFF_RESPONSE field)
- Each bullet point under the term becomes an enum value
- Include the description for each value

IMPORTANT: Only create enum fields from "### <Term>" sections that have this pattern:
  - **ValueName**: Description of what this value means

Do NOT create enum fields from "### <Term>" sections that just list examples without the **bold** value names.
For example, "### Incident" sections that say "considered relevant if it discusses: - bullet1 - bullet2"
are EXPLANATORY (describing what counts), NOT enum field definitions.

Example: If agenda has "### Assurance of Safety" with bullets for "Assurance given", "No assurance", "Unknown",
then you MUST include an ASSURANCE_CLAIM field with those 3 values.

DO NOT skip any "### <Term>" sections that define enum values! Each one is needed for proper extraction.
But DO skip sections that are just explanatory text without **bold** value definitions.

## OUTPUT FORMAT

```json
{{
  "core_concepts": {{
    "primary_topic": "<main subject>",
    "explicit_terms": ["<terms from agenda>"]
  }},

  "policy_type": "<scoring OR severity_rule_based OR signal_rule_based>",

  "evaluation_criteria": {{
    "what_counts": "<from agenda>",
    "what_does_not_count": "<from agenda>"
  }},

  "categories": {{
    "field_name": "<what to call this: INCIDENT_SEVERITY, OUTCOME, QUALITY_LEVEL, SIGNAL_TYPE, etc.>",
    "values": [
      {{"name": "<category name>", "description": "<from agenda>"}}
    ]
  }},

  "signals": {{
    "// ONLY FOR signal_rule_based policies - list signals to detect FROM AGENDA": "",
    "positive_signals": [
      {{"signal": "<signal name from agenda's Definitions>", "description": "<description from agenda>", "detection_phrases": ["<phrases that indicate this signal>"]}}
    ],
    "negative_signals": [
      {{"signal": "<signal name from agenda's Definitions>", "description": "<description from agenda>", "detection_phrases": ["<phrases that indicate this signal>"]}}
    ],
    "neutral_signal": "<signal name for no strong signal, e.g., neutral>"
  }},

  "scoring_system": {{
    "has_scoring": <true if point values are specified, false otherwise>,
    "description": "<how scoring works, or 'N/A - rule-based' if no scoring>",
    "base_point_categories": [
      {{"category": "<exact name>", "points": <exact number, can be negative>}}
    ],
    "modifiers": [
      {{"name": "<modifier name>", "condition": "<when it applies>", "points": <exact number>}}
    ]
  }},

  "verdict_rules": {{
    "type": "<scoring OR severity_rule_based OR signal_rule_based>",
    "verdicts": ["<verdict1>", "<verdict2>", "<verdict3>"],
    "evaluation_order": "<CRITICAL: Which verdict is checked FIRST? Extract from agenda - e.g., 'Excellent first, then Needs Improvement if Excellent does not apply'>",
    "rules": [
      {{
        "verdict": "<verdict label>",
        "check_order": <1, 2, 3 - the order this rule is evaluated>,
        "precondition": "<any precondition from agenda, e.g., 'Excellent does not apply', or null if none>",
        "condition_type": "<score_threshold OR count_threshold OR signal_presence>",
        "condition": "<EXACT from agenda: e.g., 'min_count: 2' becomes '>= 2'>",
        "description": "<human readable rule from agenda>"
      }}
    ],
    "default_verdict": "<verdict when no rules match>"
  }},

  "account_handling": {{
    "field_name": "<what to call the account type field - e.g., ACCOUNT_TYPE, EVIDENCE_SOURCE, REPORTER_TYPE>",
    "types": [
      // Extract ALL account types defined in the agenda - do NOT assume firsthand/secondhand/general
      // For each type, copy the ENTIRE definition text that appears after the colon/bullet marker
      // Example: "- **Firsthand**: The reviewer or their dining party experienced..." → description = "The reviewer or their dining party experienced..."
      // {{"type": "<name>", "description": "<FULL definition text after the colon - copy the ENTIRE sentence, not just the type name>", "counts_for_verdict": <from agenda>}}
    ]
  }},

  "description_field": {{
    "name": "<what to call the incident description field - e.g., SPECIFIC_INCIDENT, DESCRIPTION, DETAILS>",
    "purpose": "<purpose from agenda context>"
  }},

  "extraction_fields": [
    // CRITICAL: List EVERY "### <Term>" section from the Definitions as a field!
    // Each field the agenda defines (Staff Response, Assurance of Safety, etc.) MUST be here.
    // These fields are used by modifiers in the scoring system.
    // For each value, copy the ENTIRE definition text after the colon
    // Example: "- **Mild incident**: Discomfort or minor symptoms..." → description = "Discomfort or minor symptoms..."
    {{"name": "<FIELD_NAME from ### section>", "type": "enum", "values": [
      {{"value": "<value1>", "description": "<FULL definition text after the colon - the ENTIRE sentence>"}},
      {{"value": "<value2>", "description": "<FULL definition text after the colon - the ENTIRE sentence>"}}
    ]}}
  ]
}}
```

CRITICAL:
1. First determine the policy_type:
   - "scoring" if point values are given
   - "severity_rule_based" if counting by severity levels (mild/moderate/severe)
   - "signal_rule_based" if detecting specific signals (signal names vary by policy - READ FROM AGENDA!)
2. Extract EXACT numbers - don't use template values
3. For severity_rule_based: verdict rules use severity counts (e.g., "severe_count >= 2")
4. For signal_rule_based: verdict rules use signal counts (e.g., "N_NEGATIVE >= 1")
5. For signal_rule_based: MUST fill in the "signals" section with all positive and negative signals FROM THE AGENDA

Output ONLY the JSON:

```json
'''


# =============================================================================
# Step 2: PLAN - Keyword and Field Strategy (NO domain hints)
# =============================================================================

PLAN_PROMPT = '''Create a detailed plan for generating a Formula Seed based on these observations.

## OBSERVATIONS FROM QUERY ANALYSIS

{observations}

## YOUR JOB

Think carefully about what keywords, fields, and rules will be needed.
Base your reasoning ONLY on what was observed in the query - do NOT add external domain knowledge.

IMPORTANT: Check observations.policy_type to know if this is "scoring" or "rule_based":
- scoring: Need BASE_POINTS, MODIFIER_POINTS, SCORE computations
- rule_based: Need severity counts (N_SEVERE, N_MODERATE, etc.) and count-based verdict rules

For keywords, reason about:
- What words would reviewers use when discussing the concepts from the observations?
- What terms indicate each category level according to the agenda's definitions?
- What phrases distinguish firsthand from secondhand accounts?

KEYWORD QUALITY CRITERIA (CRITICAL):
- High-signal keywords should appear in <30% of reviews
- AVOID generic words that appear in most reviews regardless of topic:
  * Generic review language: review, customer, experience, mentioned, overall, definitely
  * Generic restaurant language: restaurant, food, service, staff, waiter, server, menu, order
  * Abstract terms: quality, incident, evidence, assessment, issue, problem, concern
  * Sentiment words: good, bad, great, terrible, amazing, awful (too broad)
- FOCUS on DOMAIN-SPECIFIC terms that distinguish relevant from irrelevant reviews
- Test mentally: "Would this keyword mostly match reviews about MY specific topic, or would it match almost any restaurant review?"

For each keyword, briefly note WHY it's specific to this task in derivation_notes.

Examples of BAD keywords (too generic): "mentioned", "experience", "restaurant", "food", "service"
Examples of GOOD keywords for allergy: "allergic", "nut-free", "gluten", "epipen", "anaphylaxis"
Examples of GOOD keywords for romance: "anniversary", "proposal", "candlelit", "intimate", "date night"

Output your plan:

```json
{{
  "policy_type": "<copy from observations.policy_type: 'scoring' or 'rule_based'>",

  "keyword_strategy": {{
    "reasoning": "<explain how you derived keywords from the observed concepts>",

    "from_core_concepts": {{
      "primary_terms": ["<words derived from primary_topic>"],
      "related_terms": ["<words derived from related_concepts>"],
      "derivation_notes": "<how these connect to the agenda>"
    }},

    "from_category_definitions": {{
      "category_indicators": {{
        "<category1>": ["<words that match this category's definition>"],
        "<category2>": ["<words that match this category's definition>"]
      }},
      "derivation_notes": "<how these connect to category descriptions>"
    }},

    "from_evaluation_criteria": {{
      "positive_indicators": ["<words indicating something counts>"],
      "negative_indicators": ["<words indicating something doesn't count>"],
      "derivation_notes": "<how these connect to criteria>"
    }},

    "account_type_indicators": {{
      "firsthand_phrases": ["<phrases indicating personal experience>"],
      "secondhand_phrases": ["<phrases indicating heard from others>"],
      "general_phrases": ["<phrases indicating general statements>"]
    }},

    "priority_terms": ["<highest-signal keywords for early stopping>"]
  }},

  "field_strategy": {{
    "reasoning": "<explain what extraction fields are needed>",

    "required_fields": [
      {{
        "name": "<from observations.account_handling.field_name>",
        "purpose": "Distinguish account types per observations.account_handling.types",
        "values": "<from observations.account_handling.types>",
        "based_on": "observations.account_handling"
      }},
      {{
        "name": "<from observations.categories.field_name>",
        "purpose": "Classify category level",
        "values": "<from observations.categories.values>",
        "based_on": "observations.categories"
      }},
      {{
        "name": "<from observations.description_field.name>",
        "purpose": "<from observations.description_field.purpose>",
        "based_on": "observations.description_field"
      }}
    ],

    "additional_fields": [
      {{
        "name": "<field_name>",
        "purpose": "<why needed>",
        "based_on": "<which modifier/criteria requires this>"
      }}
    ]
  }},

  "compute_strategy": {{
    "reasoning": "<explain the approach based on policy_type>",

    "for_scoring_type": {{
      "base_scoring": "<how to compute base points using observations.scoring_system.base_point_categories>",
      "modifier_scoring": "<how to apply modifiers using observations.scoring_system.modifiers>",
      "aggregation": "SCORE = BASE_POINTS + MODIFIER_POINTS"
    }},

    "for_rule_based_type": {{
      "counts_needed": ["<N_SEVERE>", "<N_MODERATE>", "<etc based on verdict rules>"],
      "count_logic": "<how to count each category>"
    }},

    "verdict_rules": {{
      "type": "<scoring or rule_based>",
      "rules_from_observations": "<copy observations.verdict_rules.rules>",
      "edge_cases": "<any special handling needed>"
    }}
  }}
}}
```

REMEMBER: Derive everything from the observations. Do NOT inject external domain knowledge.

Output ONLY the JSON:

```json
'''


# =============================================================================
# Step 3: ACT - Generate Formula Seed (Following the Plan)
# =============================================================================

ACT_PROMPT = '''Generate the complete Formula Seed based on observations.

## ORIGINAL OBSERVATIONS

{observations}

## GENERATION PLAN

{plan}

## CRITICAL: CHECK POLICY TYPE FIRST

Look at observations.policy_type to determine the structure:
- If "scoring": Use point-based BASE_POINTS, MODIFIER_POINTS, SCORE, and score thresholds for VERDICT
- If "rule_based": Use severity counts and count thresholds for VERDICT (NO points)

---

## FOR SCORING-BASED POLICIES (observations.policy_type == "scoring")

### EXTRACTION FIELDS
- <observations.account_handling.field_name>: enum (values from observations.account_handling.types)
- <CATEGORY_FIELD> (from observations.categories.field_name): enum with category values
- <observations.description_field.name>: string for <observations.description_field.purpose>
- Modifier fields: one per modifier in observations.scoring_system.modifiers

### COMPUTE OPERATIONS
1. N_INCIDENTS: count where <account_field> = <counting_account_type> AND <outcome_field> NOT IN <none_values>
   - CRITICAL: Only count extractions with actual incidents (outcome != none_value)
   - The outcome_field and none_values come from your extract section (defined below)
   - Example: {{"name": "N_INCIDENTS", "op": "count", "where": {{"field": "<account_field>", "equals": "<counting_type>"}}, "and": [{{"field": "<outcome_field>", "not_equals": <none_values>}}]}}
2. BASE_POINTS: sum using EXACT point values from observations.scoring_system.base_point_categories
   Example: "CASE WHEN OUTCOME = 'severe' THEN 15 WHEN OUTCOME = 'moderate' THEN 5 ... ELSE 0 END"
3. MODIFIER_POINTS: sum using EXACT point values from observations.scoring_system.modifiers
   Example: "(CASE WHEN MOD1 = 'yes' THEN 3 ELSE 0 END) + (CASE WHEN MOD2 = 'yes' THEN -5 ELSE 0 END)"
4. SCORE: BASE_POINTS + MODIFIER_POINTS
5. VERDICT: case on SCORE using observations.verdict_rules.rules conditions

### VERDICT FORMAT (SCORING)

**CRITICAL - EXACT VERDICT LABELS**: Copy verdict labels EXACTLY from observations.verdict_rules.verdicts.
Do NOT simplify or modify labels. If observations say "Critical Risk", use "Critical Risk" - NOT "Critical".
The verdicts array contains the EXACT strings to use in your rules.

Use observations.verdict_rules.rules to build:
```json
{{"name": "VERDICT", "op": "case", "source": "SCORE", "rules": [
  {{"when": ">= 8", "then": "<EXACT label from observations.verdict_rules.verdicts[0]>"}},
  {{"when": ">= 4", "then": "<EXACT label from observations.verdict_rules.verdicts[1]>"}},
  {{"else": "<EXACT label from observations.verdict_rules.verdicts[2]>"}}
]}}
```

---

## FOR RULE-BASED POLICIES (observations.policy_type == "rule_based")

### EXTRACTION FIELDS
- <observations.account_handling.field_name>: enum (values from observations.account_handling.types)
- <CATEGORY_FIELD> (from observations.categories.field_name): enum with category values
- <observations.description_field.name>: string for <observations.description_field.purpose>
- Any additional fields needed for compound conditions

**CRITICAL - Signal-Based Policies (READ CAREFULLY):**

**WARNING: The examples below use PLACEHOLDER signal names. You MUST replace them with ACTUAL signal names
from YOUR agenda's "## Definitions of Terms" section. Each policy type has DIFFERENT signals:
- Romance policies: ambiance_quality, noise_level, lighting, privacy, intimacy, etc.
- Business policies: client_impression, conversation_quality, professionalism, etc.
- Group policies: accommodation, space_suitability, noise_tolerance, etc.
- Server policies: attentiveness, friendliness, knowledge, responsiveness, etc.
DO NOT copy placeholder signals - read YOUR agenda and extract ITS signal definitions!**

If the policy verdict depends on detecting specific SIGNAL TYPES
rather than severity levels (e.g., "mild", "moderate", "severe"), you MUST follow this EXACT pattern:

**STEP 1 - Add SIGNAL_TYPE to extract.fields (NOT in compute!):**
```json
"extract": {{
  "fields": [
    {{"name": "ACCOUNT_TYPE", "type": "enum", "values": {{...}}}},
    {{"name": "SIGNAL_TYPE", "type": "enum", "values": {{
      "<positive_signal_1_from_agenda>": "<description from agenda's Definitions section>",
      "<positive_signal_2_from_agenda>": "<description from agenda's Definitions section>",
      "<negative_signal_1_from_agenda>": "<description from agenda's Definitions section>",
      "<negative_signal_2_from_agenda>": "<description from agenda's Definitions section>",
      "neutral": "No strong positive/negative signal detected"
    }}}},
    {{"name": "DESCRIPTION", "type": "string", "description": "Details of incident"}}
  ],
  "outcome_field": "SIGNAL_TYPE",
  "none_values": ["neutral"]
}}
```

**STEP 2 - Count by the ENUM field in compute:**
```json
"compute": [
  {{"name": "N_POSITIVE", "op": "count", "where": {{"SIGNAL_TYPE": ["<positive_signals_from_agenda>"]}}}},
  {{"name": "N_NEGATIVE", "op": "count", "where": {{"SIGNAL_TYPE": ["<negative_signals_from_agenda>"]}}}},
  {{"name": "VERDICT", "op": "case", "rules": [
    {{"when": "N_NEGATIVE >= 1", "then": "<unfavorable_verdict_from_agenda>"}},
    {{"when": "N_POSITIVE >= 1", "then": "<favorable_verdict_from_agenda>"}},
    {{"else": "<default_verdict_from_agenda>"}}
  ]}}
]
```

**KEY INSIGHT:** The LLM doing extraction will choose the appropriate SIGNAL_TYPE enum value for each review.
Then count operations aggregate by that enum value. This is how signal detection works!

**WRONG PATTERNS - NEVER DO THIS:**
```json
// WRONG: SIGNAL_TYPE in compute section (must be in extract.fields!)
"compute": [{{"name": "SIGNAL_TYPE", "type": "enum", ...}}]

// WRONG: Counting by string field (string fields contain free text, not matchable values)
{{"op": "count", "where": {{"DESCRIPTION": "impress_client"}}}}  // WILL ALWAYS BE 0!

// WRONG: Counting by string field with array of phrases
{{"op": "count", "where": {{"DESCRIPTION": ["private rooms", "secluded"]}}}}  // WILL ALWAYS BE 0!
```

### COMPUTE OPERATIONS
1. N_INCIDENTS: count where <account_field> = <counting_account_type> AND <outcome_field> NOT IN <none_values>
   - CRITICAL: Only count extractions with actual incidents (outcome != none_value)
   - The outcome_field and none_values come from your extract section (defined below)
2. For SEVERITY-based policies: N_SEVERE, N_MODERATE counts by severity enum
3. For SIGNAL-based policies: N_POSITIVE, N_NEGATIVE counts by signal_type enum
4. (Add more counts as needed for verdict rules)
5. VERDICT: case using COUNT conditions (NOT score)

**CRITICAL: Count operations can ONLY filter by ENUM fields, never by string fields!**

### VERDICT FORMAT (RULE-BASED)

**CRITICAL - RULE ORDER**: The order of rules in the CASE statement matters!
- Check observations.verdict_rules.evaluation_order to determine which verdict is checked FIRST
- If agenda says "Excellent first, then Needs Improvement if Excellent does not apply", put Excellent rule BEFORE Needs Improvement
- Rules with preconditions (e.g., "Excellent does not apply") come AFTER the rules they depend on

**CRITICAL - EXACT CONDITIONS**: Extract thresholds EXACTLY from agenda.
- If agenda says "min_count: 2", use ">= 2" - do NOT invent your own thresholds
- If agenda says "Multiple reviews" without a number, use ">= 2" as the standard interpretation

**CRITICAL - EXACT VERDICT LABELS**: Copy verdict labels EXACTLY from observations.verdict_rules.verdicts.
Do NOT simplify or modify labels. If observations say "Critical Risk", use "Critical Risk" - NOT "Critical".

Build verdict from observations.verdict_rules.rules using count conditions:
```json
{{"name": "VERDICT", "op": "case", "source": "N_SEVERE", "rules": [
  {{"when": ">= 2", "then": "<EXACT from observations.verdict_rules.verdicts>"}},
  {{"else": null}}
], "fallback_source": "N_MODERATE", "fallback_rules": [
  {{"when": ">= 2", "then": "<EXACT from observations.verdict_rules.verdicts>"}},
  {{"else": "<EXACT from observations.verdict_rules.verdicts>"}}
]}}
```

OR use compound conditions in a single case:
```json
{{"name": "VERDICT", "op": "case", "rules": [
  {{"when": "N_SEVERE >= 2", "then": "<EXACT from observations.verdict_rules.verdicts[0]>"}},
  {{"when": "N_MODERATE >= 2", "then": "<EXACT from observations.verdict_rules.verdicts[1]>"}},
  {{"else": "<EXACT from observations.verdict_rules.verdicts[2]>"}}
]}}
```

---

## FOR SIGNAL-RULE-BASED POLICIES (observations.policy_type == "signal_rule_based")

This is for policies where verdict depends on detecting specific SIGNALS or QUALITY levels (not incident severity).

**IMPORTANT:** Signal names are POLICY-SPECIFIC. You MUST read the agenda's "## Definitions of Terms" section
to extract the actual signal names. Examples vary by policy type:
- Romance: ambiance, intimacy, noise_level, privacy, romantic_elements
- Business: client_impression, conversation_quality, professionalism
- Group: accommodation, space, noise_tolerance, party_friendliness
- Service: attentiveness, friendliness, knowledge, responsiveness

### EXTRACTION FIELDS
1. ACCOUNT_TYPE: enum (Firsthand, Secondhand, Hypothetical)
2. **SIGNAL_TYPE or SERVICE_QUALITY: enum with ALL signals/quality levels from observations.signals**
   - Include all positive signals FROM OBSERVATIONS.SIGNALS.POSITIVE_SIGNALS
   - Include all negative signals FROM OBSERVATIONS.SIGNALS.NEGATIVE_SIGNALS
   - Include a neutral value (e.g., "neutral", "adequate")
3. DESCRIPTION: string for incident details

### COMPUTE OPERATIONS (REQUIRED!)

**Example - Signal detection (use signals from YOUR agenda):**
```json
"compute": [
  {{"name": "N_POSITIVE", "op": "count", "where": {{"SIGNAL_TYPE": ["<positive_signals_from_observations>"]}}}},
  {{"name": "N_NEGATIVE", "op": "count", "where": {{"SIGNAL_TYPE": ["<negative_signals_from_observations>"]}}}},
  {{"name": "VERDICT", "op": "case", "rules": [
    {{"when": "N_NEGATIVE >= <threshold_from_agenda>", "then": "<unfavorable_verdict>"}},
    {{"when": "N_POSITIVE >= <threshold_from_agenda>", "then": "<favorable_verdict>"}},
    {{"else": "<default_verdict>"}}
  ]}}
]
```

**CRITICAL:** Verdict rules and signal names MUST come from the agenda (observations), not invented.
Extract the EXACT conditions and signal names specified in the agenda.

---

## COMPUTE SECTION IS MANDATORY

**CRITICAL**: The compute section MUST NOT be empty! At minimum, it must contain:
1. At least one count operation (N_INCIDENTS, N_POSITIVE, N_NEGATIVE, etc.)
2. A VERDICT operation

If compute is empty, the seed is INVALID and will fail to produce any verdicts.

---

## OUTPUT FORMAT - FOLLOW EXACTLY

**IMPORTANT: This is a SIMPLIFIED schema. Do NOT include filter, search_strategy, expansion_hints, or verdict_metadata.**

```json
{{
  "task_name": "<from observations.core_concepts.primary_topic>",

  "extraction_guidelines": "<GENERATE THIS dynamically from observations - see instructions below>",

  "extract": {{
    "fields": [
      // REQUIRED: Account type field with values as DICT (not array!)
      // Key = type name (e.g., "Firsthand"), Value = FULL description from observations.account_handling.types[].description
      {{"name": "<observations.account_handling.field_name>", "type": "enum", "values": {{
        "<type_name>": "<FULL description sentence from observations.account_handling.types - e.g., 'The reviewer or their dining party experienced the event directly'>",
        "<type_name>": "<FULL description sentence - copy the ENTIRE text, not just the name>"
      }}}},
      // REQUIRED: Category/outcome field with values as DICT
      // Key = category name, Value = FULL description from observations.categories.values[].description
      {{"name": "<observations.categories.field_name>", "type": "enum", "values": {{
        "<cat_name>": "<FULL description from observations.categories.values - copy entire sentence>",
        "<cat_name>": "<FULL description - preserve all details like 'hives or swelling', 'medication', etc.>"
      }}}},
      // REQUIRED: Description field
      {{"name": "<observations.description_field.name>", "type": "string", "description": "<observations.description_field.purpose>"}},
      // REQUIRED: Include ALL fields from observations.extraction_fields
      // EVERY "### <Term>" field from the agenda MUST be here (Staff Response, Assurance, etc.)
      // Each field referenced in MODIFIER_POINTS scoring MUST be defined here!
    ],
    // REQUIRED METADATA - tells phase2 which field represents the outcome and what means "no incident"
    "outcome_field": "<observations.categories.field_name>",  // e.g., "INCIDENT_SEVERITY", "QUALITY_LEVEL", "OUTCOME"
    "none_values": ["<values that mean no incident>"]  // e.g., ["none", "no incident", "n/a"] - from observations.categories.values
  }},

  // CRITICAL: All enum value descriptions MUST be FULL SENTENCES copied from observations.
  // WRONG: "Firsthand": "Firsthand" (just repeating the name)
  // CORRECT: "Firsthand": "The reviewer or their dining party experienced the event directly"
  // Do NOT shorten. Preserve ALL qualifiers (e.g., "or their dining party", "hives or swelling").

  "compute": [
    // EXACT OPERATION FORMATS (use these, NOT SQL expressions):

    // COUNT operation - counts extractions matching conditions:
    {{"name": "N_INCIDENTS", "op": "count", "where": {{
      "ACCOUNT_TYPE": ["firsthand"],
      "INCIDENT_SEVERITY": ["mild", "moderate", "severe"]
    }}}},
    // NOTE: where conditions use ENUM VALUES from extract.fields. Use list for multiple values.

    // SUM operation - use SQL CASE WHEN syntax in expr:
    {{"name": "BASE_POINTS", "op": "sum", "expr": "CASE WHEN INCIDENT_SEVERITY = 'Severe' THEN 15 WHEN INCIDENT_SEVERITY = 'Moderate' THEN 5 WHEN INCIDENT_SEVERITY = 'Mild' THEN 2 ELSE 0 END", "where": {{"ACCOUNT_TYPE": ["Firsthand"]}}}},

    // FOR SCORING POLICIES: Modifiers can be separate sum operations (cleaner):
    {{"name": "MOD_ASSURANCE", "op": "sum", "expr": "CASE WHEN ASSURANCE_OF_SAFETY = 'Assurance given' AND INCIDENT_SEVERITY IN ('Mild','Moderate','Severe') THEN 5 ELSE 0 END", "where": {{"ACCOUNT_TYPE": ["Firsthand"]}}}},

    // EXPR operation - combines computed values (supports CASE WHEN if needed):
    {{"name": "SCORE", "op": "expr", "expr": "BASE_POINTS + MODIFIER_POINTS"}},

    // CASE operation for verdict - applies threshold rules:
    {{"name": "VERDICT", "op": "case", "source": "SCORE", "rules": [
      {{"when": ">= 12", "then": "Critical Risk"}},
      {{"when": ">= 2", "then": "High Risk"}},
      {{"else": "Low Risk"}}
    ]}}
  ],

  "output": ["VERDICT", "SCORE", "N_INCIDENTS"]
}}
```

## CRITICAL RULES - MUST FOLLOW

1. **Values format**: extract.fields[].values MUST be a DICT (not array!): {{"value1": "desc1", "value2": "desc2"}}
2. **Compute format**: Each compute operation MUST have "op" key (count/sum/expr/case), NOT "expression"
3. **Field references**: ALL fields used in compute MUST exist in extract.fields
   - If MODIFIER_POINTS uses ASSURANCE_OF_SAFETY, add it to extract.fields!
   - If compute references STAFF_RESPONSE, add it to extract.fields!
   - CRITICAL: Each field name must appear EXACTLY ONCE in extract.fields - do NOT define the same field multiple times!
4. **sum operations**: Use {{"op": "sum", "expr": "CASE WHEN ... THEN ... END", "where": {{...}}}} - Phase 2 only reads the "expr" key
5. **case operations**: Use {{"op": "case", "source": "...", "rules": [...]}} format
6. **expr operations**: Use {{"op": "expr", "expr": "..."}} for combining computed values
7. **COUNT OPERATIONS - ENUM ONLY**: Count "where" clauses can ONLY reference ENUM fields!
   - CORRECT: {{"where": {{"SIGNAL_TYPE": ["positive"]}}}} where SIGNAL_TYPE is type: "enum"
   - WRONG: {{"where": {{"DESCRIPTION": "some text"}}}} where DESCRIPTION is type: "string"
   - If you need to count by signal types, define those signals as an ENUM field first!
8. **VERDICT LABELS**: Use EXACT verdict strings from observations.verdict_rules.verdicts - copy character-for-character!
   - If observations say "Critical Risk", use "Critical Risk" NOT "Critical"
   - If observations say "Low Risk", use "Low Risk" NOT "Low"
9. **outcome_field (REQUIRED)**: MUST set extract.outcome_field to the category field name that represents the outcome
   - This is the field phase2 uses to determine if an extraction represents an actual incident
   - Example: for allergy → "INCIDENT_SEVERITY", for romance → "AMBIANCE_QUALITY", for service → "SERVICE_OUTCOME"
10. **none_values (REQUIRED)**: MUST set extract.none_values to the list of values that mean "no incident"
   - Derive from observations.categories.values - find values meaning absence/none/not applicable
   - Example: ["none", "not applicable", "no incident"]
11. **ENUM CONSISTENCY (CRITICAL)**: The `where` conditions in compute operations MUST use values that match
    what the LLM extraction will return. Use SHORT LABELS that match the extract.fields enum keys:
    - WRONG: extract.fields uses {{"Moderate incident": "..."}}, compute.where uses {{"SEVERITY": ["Moderate"]}}
    - CORRECT: extract.fields uses {{"Moderate": "..."}}, compute.where uses {{"SEVERITY": ["Moderate"]}}
    - Keep enum value keys SHORT and SIMPLE (e.g., "Moderate" not "Moderate incident")
12. **NO EXTRA SECTIONS**: Do NOT include filter, search_strategy, expansion_hints, or verdict_metadata sections!
    The simplified schema only needs: task_name, extraction_guidelines, extract, compute, output.
13. **VERDICT REQUIRED**: The compute section MUST include a VERDICT operation with op="case".
    This is required for the seed to produce deterministic verdicts.

14. **extraction_guidelines**: MUST be generated from observations to guide precise extraction. Format as a multi-line string:

   Generate guidelines covering these aspects (derive ALL from observations, never hard-code):

   a) WHAT COUNTS AS AN INCIDENT:
      - Derive from observations.evaluation_criteria.what_counts
      - Example: "Reports of actual allergic reactions, near-misses, or explicit ingredient mislabeling"

   b) WHAT DOES NOT COUNT:
      - Derive from observations.evaluation_criteria.what_does_not_count
      - Example: "General mentions of food without allergy context, hypothetical concerns, positive safety mentions"

   c) ACCOUNT TYPES THAT CONTRIBUTE TO VERDICT:
      - Derive from observations.account_handling.types (only those with counts_for_verdict=true)
      - Example: "Only firsthand accounts (personal experience) count. Secondhand (heard from others) and general statements do NOT count."

   d) SEVERITY CLASSIFICATION:
      - Derive from observations.categories.values
      - Example: "Severe: life-threatening reactions, ER visits. Moderate: significant symptoms, medication needed. Mild: minor discomfort."

   e) CRITICAL VALIDATION:
      - "You MUST include a supporting_quote - the exact text from the review that supports your extraction."
      - "If you cannot quote specific evidence from the review, set is_relevant: false."

   The extraction_guidelines field should be a SINGLE STRING with these sections separated by newlines.

Output ONLY the JSON:

```json
'''


# =============================================================================
# Fix Prompts (for validation loop)
# =============================================================================

FIX_PROMPT = '''The Formula Seed you generated has validation errors:

{errors}

Here is the problematic Formula Seed:
```json
{seed_json}
```

Please fix these errors and output the COMPLETE corrected Formula Seed JSON.

Common fixes for expression errors:
- Use consistent quote style: 'Critical Risk' not "Critical Risk' (mismatched quotes)
- Ensure parentheses are balanced
- Use valid Python operators: >= not =>
- String literals inside expressions should use single quotes: 'value' not "value"

Output ONLY the corrected JSON (no explanation):

```json
'''

SEMANTIC_FIX_PROMPT = '''The Formula Seed you generated has semantic validation errors against the agenda:

## ERRORS

{errors}

## ORIGINAL AGENDA (for reference)

{agenda_snippet}

## PROBLEMATIC FORMULA SEED

```json
{seed_json}
```

## FIX REQUIREMENTS

Please fix these semantic errors:

1. **Verdict Labels**: Use EXACT verdict labels from the agenda (e.g., "Critical Risk" not "Critical")
2. **Evaluation Order**: The first case rule should check the first verdict in the precedence ladder
3. **Default Verdict**: The else clause should return the default/"otherwise" verdict
4. **Count Thresholds**: Use the EXACT thresholds from the agenda (e.g., ">= 2" if agenda says "2 or more")

Output the COMPLETE corrected Formula Seed JSON:

```json
'''


# =============================================================================
# Constants for Extraction Fields
# =============================================================================

REQUIRED_EXTRACTION_FIELDS = """
You MUST use these EXACT field names in extraction schema:
- ACCOUNT_TYPE: "firsthand" | "secondhand" | "general"
- INCIDENT_SEVERITY: "none" | "mild" | "moderate" | "severe"
- SPECIFIC_INCIDENT: string describing what happened (or null if none)

Additional fields may be added based on task requirements (e.g., STAFF_RESPONSE, RECURRENCE).
"""

COMPUTE_TEMPLATE = """
Use this EXACT compute structure (adjust values based on task):

1. Count incidents:
   {{"name": "N_INCIDENTS", "op": "count", "where": {{"ACCOUNT_TYPE": "firsthand", "INCIDENT_SEVERITY": ["mild", "moderate", "severe"]}}}}

2. Sum base points (use CASE WHEN for severity scoring):
   {{"name": "BASE_POINTS", "op": "sum", "expr": "CASE WHEN INCIDENT_SEVERITY = 'severe' THEN <severe_pts> WHEN INCIDENT_SEVERITY = 'moderate' THEN <mod_pts> WHEN INCIDENT_SEVERITY = 'mild' THEN <mild_pts> ELSE 0 END", "where": {{"ACCOUNT_TYPE": "firsthand"}}}}

3. Sum modifier points (if task has modifiers):
   {{"name": "MODIFIER_POINTS", "op": "sum", "expr": "CASE WHEN MODIFIER_FIELD = 'trigger_value' THEN <mod_points> ELSE 0 END", "where": {{"ACCOUNT_TYPE": "firsthand"}}}}

4. Total score:
   {{"name": "SCORE", "op": "expr", "expr": "BASE_POINTS + MODIFIER_POINTS"}}

5. Verdict (use threshold values from task):
   {{"name": "VERDICT", "op": "case", "source": "SCORE", "rules": [
     {{"when": ">= <critical_threshold>", "then": "<critical_label>"}},
     {{"when": ">= <high_threshold>", "then": "<high_label>"}},
     {{"else": "<low_label>"}}
   ]}}
"""
