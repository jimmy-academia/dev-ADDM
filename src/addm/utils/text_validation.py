"""Multi-span snippet validation utilities.

Validates that LLM-generated quotes exist in source text, with support for
non-adjacent sentence combinations (multi-span matching).

This is needed because LLMs often combine non-adjacent sentences when quoting:
  LLM quote: "I was super sick since that dinner... I was sick all night"
  Actual:    "I was super sick since that dinner [OTHER TEXT] I was sick all night"

The multi-span approach validates these combined quotes by:
1. Splitting into sentence segments
2. Finding each segment's position in the source
3. Checking 80% of segments exist AND appear in monotonic order
"""

import re
from typing import Any, Dict, List, Optional, Tuple


def _normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, collapse whitespace)."""
    return " ".join(text.lower().split())


def _split_into_segments(quote: str, min_length: int = 10) -> List[str]:
    """Split quote into sentence segments for multi-span matching.

    Splits on sentence boundaries (., !, ?) and filters to segments
    of at least min_length characters.

    Args:
        quote: The quote text to split
        min_length: Minimum segment length in characters

    Returns:
        List of normalized segments
    """
    # Split on sentence-ending punctuation followed by space or end
    # Keeps the punctuation with the sentence
    parts = re.split(r'(?<=[.!?])\s+', quote)

    segments = []
    for part in parts:
        normalized = _normalize_text(part)
        # Filter out very short segments (likely fragments or punctuation)
        if len(normalized) >= min_length:
            segments.append(normalized)

    # If no valid segments from sentence splitting, try splitting on ellipsis
    if not segments and "..." in quote:
        parts = quote.split("...")
        for part in parts:
            normalized = _normalize_text(part)
            if len(normalized) >= min_length:
                segments.append(normalized)

    # If still no segments, return the whole quote as one segment if long enough
    if not segments:
        normalized = _normalize_text(quote)
        if len(normalized) >= min_length:
            segments = [normalized]

    return segments


def _find_segment_position(segment: str, source: str, start_from: int = 0) -> int:
    """Find segment position in source text.

    Args:
        segment: Normalized segment to find
        source: Normalized source text
        start_from: Minimum position to search from (for monotonicity)

    Returns:
        Position of segment start, or -1 if not found
    """
    # Try exact match first
    pos = source.find(segment, start_from)
    if pos >= 0:
        return pos

    # Try fuzzy match: allow minor differences (strip trailing punctuation)
    segment_stripped = segment.rstrip(".,!?;:")
    if segment_stripped != segment:
        pos = source.find(segment_stripped, start_from)
        if pos >= 0:
            return pos

    return -1


def validate_multi_span_snippet(
    quote: str,
    source_text: str,
    min_segment_length: int = 10,
    segment_threshold: float = 0.8,
) -> Dict[str, Any]:
    """Validate snippet using multi-span matching with 80% threshold.

    Allows quotes that combine non-adjacent sentences, as long as:
    1. >=80% of segments exist in the source review
    2. Found segments appear in monotonic order (preserve reading order)

    Args:
        quote: The quoted text to validate
        source_text: The original review text
        min_segment_length: Minimum segment length for splitting (default: 10)
        segment_threshold: Fraction of segments that must be found (default: 0.8)

    Returns:
        Dict with:
        - valid: bool - whether quote passes validation
        - match_type: "exact" | "multi_span" | "word_overlap" | "no_match"
        - segments_found: int - number of segments found in source
        - segments_total: int - total segments in quote
        - match_ratio: float 0.0-1.0 - fraction of segments found
        - positions: List[int] - positions where segments were found (-1 if not found)
    """
    result = {
        "valid": False,
        "match_type": "no_match",
        "segments_found": 0,
        "segments_total": 0,
        "match_ratio": 0.0,
        "positions": [],
    }

    if not quote or not quote.strip():
        return result

    if not source_text or not source_text.strip():
        return result

    # Normalize both texts
    quote_normalized = _normalize_text(quote)
    source_normalized = _normalize_text(source_text)

    # Try 1: Exact substring match (fastest path)
    if quote_normalized in source_normalized:
        return {
            "valid": True,
            "match_type": "exact",
            "segments_found": 1,
            "segments_total": 1,
            "match_ratio": 1.0,
            "positions": [source_normalized.find(quote_normalized)],
        }

    # Try 2: Multi-span matching
    segments = _split_into_segments(quote, min_segment_length)
    result["segments_total"] = len(segments)

    if segments:
        positions = []
        found_count = 0
        last_pos = -1
        in_order = True

        for segment in segments:
            pos = _find_segment_position(segment, source_normalized)
            positions.append(pos)

            if pos >= 0:
                found_count += 1
                # Check monotonicity: each found segment should come after previous
                if last_pos >= 0 and pos <= last_pos:
                    in_order = False
                if pos > last_pos:  # Only update if found (for monotonicity tracking)
                    last_pos = pos

        result["positions"] = positions
        result["segments_found"] = found_count
        result["match_ratio"] = found_count / len(segments) if segments else 0.0

        # Valid if threshold met AND segments are in order
        if found_count / len(segments) >= segment_threshold and in_order:
            result["valid"] = True
            result["match_type"] = "multi_span"
            return result

    # Try 3: Word overlap fallback (for short quotes or heavy paraphrasing)
    quote_words = set(quote_normalized.split())
    if len(quote_words) >= 3:
        source_words = set(source_normalized.split())
        overlap = len(quote_words & source_words)
        overlap_ratio = overlap / len(quote_words)

        if overlap_ratio >= 0.8:
            result["valid"] = True
            result["match_type"] = "word_overlap"
            result["match_ratio"] = overlap_ratio
            return result

    return result


def validate_snippet_simple(quote: str, source_text: str) -> bool:
    """Simple validation: returns True if quote is valid.

    Convenience wrapper for validate_multi_span_snippet.

    Args:
        quote: The quoted text to validate
        source_text: The original review text

    Returns:
        True if quote passes validation
    """
    result = validate_multi_span_snippet(quote, source_text)
    return result["valid"]
