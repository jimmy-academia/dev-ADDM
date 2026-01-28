"""Phase 2 ATKD prompts for gate init, gate discovery, and verifier."""

from __future__ import annotations

from typing import Any, Dict, List


def build_gate_init_prompt(primitives: List[Dict[str, Any]]) -> str:
    """Build GateInit prompt for initial cheap gates per primitive."""
    return (
        "You are initializing cheap gates for Active Test-time Knowledge Discovery.\n\n"
        "Given a list of primitives (each is a clause with term/value definitions),\n"
        "propose multiple BM25 keyword groups and embedding prototype sentences.\n"
        "You MUST provide both positive and negative gates.\n\n"
        "Rules:\n"
        "- Output JSON only.\n"
        "- Provide MULTIPLE gate instances per primitive for BOTH BM25 and embeddings.\n"
        "- Do NOT include verdict logic or label ordering; only focus on the clause meaning.\n"
        "- BM25 gates are keyword groups (lists of short keywords/phrases).\n"
        "- Embedding gates are full natural-language prototype sentences.\n\n"
        "OUTPUT JSON SCHEMA:\n"
        "{\n"
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
        "PRIMITIVES:\n"
        f"{_format_primitives(primitives)}\n"
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
        "- Do NOT include verdict logic or label ordering; only focus on the clause meaning.\n"
        "- BM25 gates are keyword groups (lists of short keywords/phrases).\n"
        "- Embedding gates are full natural-language prototype sentences.\n"
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


def build_verifier_prompt(
    review_text: str,
    primitive: Dict[str, Any],
    term_defs: Dict[str, Any],
) -> str:
    """Build LLM verifier prompt for a review and primitive clause."""
    return (
        "You are verifying whether a single review satisfies a primitive clause.\n\n"
        "Return JSON ONLY.\n\n"
        "OUTPUT JSON SCHEMA:\n"
        "{\n"
        "  \"is_match\": true | false,\n"
        "  \"evidence_snippets\": [\"verbatim snippet\"],\n"
        "  \"fields\": {\n"
        "    \"FIELD_ID\": [\"value_id\", \"value_id\"]\n"
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- Only mark is_match true if the review satisfies the clause.\n"
        "- evidence_snippets must be exact substrings from the review.\n"
        "- fields should include any extracted term values relevant to the clause.\n"
        "- If no evidence, return is_match=false with empty snippets/fields.\n"
        "- Do NOT infer verdicts or use any label ordering.\n\n"
        f"PRIMITIVE:\n{_format_primitives([primitive])}\n\n"
        f"TERM DEFINITIONS:\n{_format_term_defs(term_defs)}\n\n"
        f"REVIEW:\n{review_text}\n"
    )


def _format_primitives(primitives: List[Dict[str, Any]]) -> str:
    lines = []
    for p in primitives:
        lines.append(f"- primitive_id: {p.get('primitive_id')}")
        lines.append(f"  clause_quote: {p.get('clause_quote')}")
        lines.append(f"  min_count: {p.get('min_count')}")
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
