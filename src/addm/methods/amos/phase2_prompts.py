"""Phase 2 ATKD prompts for gate init, gate discovery, and verifier."""

from __future__ import annotations

from typing import Any, Dict, List


def build_gate_init_prompt(
    primitives: List[Dict[str, Any]],
    term_defs: Dict[str, Any],
    agenda_text: str,
    num_gates_suggest: int,
) -> str:
    """Build GateInit prompt for initial cheap gates per primitive."""
    primitives_lines: List[str] = []
    for p in primitives:
        primitives_lines.append(f"- primitive_id: {p.get('primitive_id')}")
        if p.get("conditions") is not None:
            primitives_lines.append(f"  conditions: {p.get('conditions')}")
    primitives_text = "\n".join(primitives_lines) if primitives_lines else "(none)"

    return (
        "You are initializing cheap gates for Active Test-time Knowledge Discovery (ATKD).\n\n"
        "Input:\n"
        "- The full policy agenda (verbatim)\n"
        "- A list of primitives (each primitive_id corresponds to one condition)\n"
        "- Term schema (fields + value_ids + descriptions)\n\n"
        "Goal:\n"
        "- First, extract a short list of topic anchor terms from the agenda.\n"
        "- Then, for EACH primitive, propose cheap gates that are anchored to the agenda topic.\n"
        f"For EACH primitive, propose up to {num_gates_suggest} gate instances of EACH kind:\n"
        "- pos_bm25_gates: keyword/phrase groups that tend to appear when the primitive signal is truly present\n"
        "- neg_bm25_gates: keyword/phrase groups that tend to appear in benign/confuser contexts\n"
        "- pos_emb_gates: natural-language prototype sentences (write like a real review) that express the primitive signal\n"
        "- neg_emb_gates: prototype sentences for benign/confuser contexts\n\n"
        "Rules:\n"
        "- Output JSON only.\n"
        "- Topic-first: use the agenda to understand what signals matter and how reviewers write them.\n"
        "- All gates MUST be specific to the agenda topic. Do NOT propose generic restaurant-quality gates\n"
        "  (e.g., service speed, rudeness, price, cleanliness) unless the agenda explicitly makes them evidence.\n"
        "- Avoid generic sentiment-only keywords (e.g., \"terrible\", \"awful\", \"worst\", \"disappointed\") unless\n"
        "  they are directly tied to a topic-specific event described in the agenda.\n"
        "- Do NOT compute verdicts or apply thresholds.\n"
        "- Do NOT use verdict/rule/threshold/rubric language (e.g., \"1 or more\", \"2 or more\",\n"
        "  \"Critical Risk\", \"High Risk\", \"Low Risk\", \"min_count\", \"clause\", \"default\").\n"
        "- Use the primitive conditions (term values) + term schema descriptions to identify concrete review-language\n"
        "  signals. Gates must target single-review evidence, not number words.\n"
        "- Do NOT create gates about repetition (e.g., \"again\", \"repeat\", \"multiple visits\"). Counts are handled\n"
        "  in code; the gate should target the per-review evidence signal.\n"
        "- Encourage diversity: cover different aspects of the topic (events/symptoms, staff actions,\n"
        "  safety assurances, cross-contact language, outcomes, abbreviations/misspellings, etc.).\n"
        "- If the agenda includes examples (e.g., a specific allergen/product/brand), treat them as examples only.\n"
        "  Include broader synonyms and related terms that appear in real reviews.\n"
        "- topic_anchors must be agenda-specific. Avoid sentiment-only words (\"terrible\", \"awful\") and generic\n"
        "  restaurant words (\"service\", \"wait\", \"price\") unless the agenda explicitly makes them evidence.\n"
        "- BM25 keyword groups are AND-groups: ALL keywords in the group must appear in a review for that gate to match.\n"
        "  Prefer single discriminative keywords, or 2-word fixed phrases. Avoid generic words like \"incident\",\n"
        "  \"episode\", \"reviewer\", \"dining\", \"visit\", \"room\", \"check\", \"staff\".\n"
        "- For positive BM25 groups, prefer 2-token AND-groups that combine:\n"
        "  (a) a topic marker from the agenda (topic_anchors), and\n"
        "  (b) a concrete marker from the relevant term/value descriptions for that primitive.\n"
        "  This reduces false positives from generic words that appear outside the topic.\n"
        "- Negative BM25 gates must be specific to benign/confuser contexts (e.g., explicit \"no reaction\" / \"no issues\"\n"
        "  style language), not just generic topic words (e.g., \"allergy\" alone).\n"
        "- Embedding prototype sentences should read like real reviews and MUST include concrete agenda markers:\n"
        "  an allergen/symptom/medication/safety phrase (or analogous concrete marker) mentioned in the agenda/term schema.\n"
        "  Do NOT write generic negative-review sentences that could match any topic.\n"
        "- Positive embedding sentences should include at least one topic anchor AND at least one concrete marker.\n"
        "- Negative embedding sentences should still be topic-relevant (mention anchors/markers) but clearly indicate a\n"
        "  benign/confuser context.\n"
        f"- If you cannot think of {num_gates_suggest} good gates for a list, return fewer; do not pad.\n\n"
        "OUTPUT JSON SCHEMA:\n"
        "{\n"
        "  \"topic_anchors\": [\"anchor1\", \"anchor2\"],\n"
        "  \"primitives\": [\n"
        "    {\n"
        "      \"primitive_id\": \"...\",\n"
        "      \"pos_bm25_gates\": [[\"kw1\", \"kw2\"], [\"kw3\", \"kw4\"]],\n"
        "      \"neg_bm25_gates\": [[\"kw1\", \"kw2\"], [\"kw3\", \"kw4\"]],\n"
        "      \"pos_emb_gates\": [\"sentence 1\", \"sentence 2\"],\n"
        "      \"neg_emb_gates\": [\"sentence 1\", \"sentence 2\"]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "AGENDA (verbatim):\n"
        f"{agenda_text}\n\n"
        "PRIMITIVES:\n"
        f"{primitives_text}\n\n"
        "TERM SCHEMA:\n"
        f"{_format_term_defs(term_defs)}\n"
    )


def build_gate_discover_prompt(
    primitive: Dict[str, Any],
    positive_snippets: List[str],
    negative_snippets: List[str],
) -> str:
    """Build GateDiscover prompt to propose new gates from verified snippets."""
    return (
        "You are discovering NEW cheap gates for a primitive clause.\n\n"
        "Given the primitive definition and verified review snippets, propose\n"
        "new positive and negative gates for BM25 and embeddings.\n\n"
        "Rules:\n"
        "- Output JSON only.\n"
        "- Provide MULTIPLE gate instances per primitive for BOTH BM25 and embeddings.\n"
        "- Use review-language phrasing; avoid rubric/verdict phrases.\n"
        "- BM25 gates are keyword groups (AND-groups): all keywords in a group must appear in a review.\n"
        "  Prefer single discriminative keywords or tight 2-word phrases; avoid generic words.\n"
        "- Embedding gates are full natural-language prototype sentences.\n"
        "- Embedding prototypes must include topic-specific markers seen in the snippets (not just generic negativity).\n"
        "- Avoid duplicating the exact phrasing from the snippets; generalize.\n\n"
        "OUTPUT JSON SCHEMA:\n"
        "{\n"
        "  \"primitive_id\": \"...\",\n"
        "  \"pos_bm25_gates\": [[\"kw1\", \"kw2\"], [\"kw3\", \"kw4\"]],\n"
        "  \"neg_bm25_gates\": [[\"kw1\", \"kw2\"], [\"kw3\", \"kw4\"]],\n"
        "  \"pos_emb_gates\": [\"sentence 1\", \"sentence 2\"],\n"
        "  \"neg_emb_gates\": [\"sentence 1\", \"sentence 2\"]\n"
        "}\n\n"
        f"PRIMITIVE:\n{_format_primitives([primitive])}\n\n"
        f"VERIFIED POSITIVE SNIPPETS:\n{_format_snippets(positive_snippets)}\n\n"
        f"VERIFIED HARD NEGATIVE SNIPPETS:\n{_format_snippets(negative_snippets)}\n"
    )


def build_extract_evident_prompt(
    review_text: str,
    term_defs: Dict[str, Any],
    review_id: str,
) -> str:
    """Build LLM prompt to extract review-level evidents."""
    return (
        "You are extracting structured evidents from a single review.\n"
        "Return STRICT JSON ONLY. Do not include prose.\n\n"
        "OUTPUT JSON SCHEMA:\n"
        "{\n"
        "  \"review_id\": \"...\",\n"
        "  \"evidents\": [\n"
        "    {\n"
        "      \"event_id\": \"...\",\n"
        "      \"fields\": {\"<term_id>\": \"<value_id>\" OR [\"<value_id>\", ...]},\n"
        "      \"snippet\": \"exact quote from review supporting this event\",\n"
        "      \"span\": {\"start_char\": int, \"end_char\": int}\n"
        "    }\n"
        "  ],\n"
        "  \"notes\": \"optional short\"\n"
        "}\n\n"
        "Rules:\n"
        "- Only include fields from the term schema below.\n"
        "- Use value_ids exactly as given.\n"
        "- Snippet must be an exact substring from the review.\n"
        "- If no relevant evidence, set \"evidents\": [].\n"
        "- Do NOT infer verdicts or label ordering.\n\n"
        f"REVIEW_ID: {review_id}\n\n"
        f"TERM SCHEMA:\n{_format_term_defs(term_defs)}\n\n"
        f"REVIEW:\n{review_text}\n"
    )


def _format_primitives(primitives: List[Dict[str, Any]]) -> str:
    lines = []
    for p in primitives:
        lines.append(f"- primitive_id: {p.get('primitive_id')}")
        if p.get("label"):
            lines.append(f"  label: {p.get('label')}")
        lines.append(f"  conditions: {p.get('conditions')}")
    return "\n".join(lines)


def _format_term_defs(term_defs: Dict[str, Any]) -> str:
    lines = []
    for field_id, values in term_defs.items():
        lines.append(f"- {field_id}:")
        for v in values:
            value_id = v.get("value_id")
            description = v.get("description")
            lines.append(f"  - {value_id}: {description}")
    return "\n".join(lines)


def _format_snippets(snippets: List[str]) -> str:
    if not snippets:
        return "(none)"
    return "\n".join(f"- {s}" for s in snippets)
