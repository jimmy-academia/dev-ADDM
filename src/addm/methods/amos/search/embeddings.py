"""Hybrid Retriever for AMOS.

Provides embedding-based retrieval to supplement keyword filtering when
the search strategy indicates more evidence is needed.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI

from addm.methods.amos.search.executor import SafeExpressionExecutor

logger = logging.getLogger(__name__)

# Embedding pricing (text-embedding-3-large): $0.13 per 1M tokens
EMBEDDING_COST_PER_1M_TOKENS = 0.13


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 chars)."""
    return len(text) // 4


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm_product)


class HybridRetriever:
    """Embedding retrieval triggered by search strategy conditions.

    Supplements keyword-based filtering when the LLM-generated strategy
    indicates that keyword matching yielded insufficient results.

    The retriever reuses the RAG embedding cache infrastructure.

    Example:
        >>> retriever = HybridRetriever()
        >>> strategy = {"use_embeddings_when": "len(keyword_matched) < 5"}
        >>> reviews = await retriever.retrieve_if_needed(
        ...     strategy=strategy,
        ...     keyword_matched=keyword_matched_reviews,
        ...     all_reviews=all_reviews,
        ...     query="Find allergy incidents",
        ...     executor=executor,
        ... )
    """

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        embedding_model: str = "text-embedding-3-large",
    ):
        """Initialize hybrid retriever.

        Args:
            cache_path: Path to embedding cache file.
                        Default: results/cache/rag_embeddings.json
            embedding_model: OpenAI embedding model to use.
        """
        self.cache_path = cache_path or Path("results/cache/rag_embeddings.json")
        self.embedding_model = embedding_model
        self._embedding_client: Optional[AsyncOpenAI] = None
        self._cache: Dict[str, Any] = {}
        self._load_cache()

        # Track metrics
        self._embedding_tokens: int = 0
        self._embedding_cost: float = 0.0
        self._cache_hit: bool = False

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
        except OSError as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    def _get_cache_key(self, sample_id: str, num_reviews: int) -> str:
        """Generate cache key for embeddings."""
        return f"amos_embeddings_{sample_id}_K{num_reviews}"

    def _get_embedding_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client for embeddings."""
        if self._embedding_client is None:
            self._embedding_client = AsyncOpenAI()
        return self._embedding_client

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using OpenAI embeddings API.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        client = self._get_embedding_client()
        response = await client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def get_metrics(self) -> Dict[str, Any]:
        """Get embedding metrics from last retrieval.

        Returns:
            Dict with embedding_tokens, embedding_cost_usd, cache_hit
        """
        return {
            "embedding_tokens": self._embedding_tokens,
            "embedding_cost_usd": self._embedding_cost,
            "cache_hit": self._cache_hit,
        }

    def reset_metrics(self) -> None:
        """Reset metrics for new sample."""
        self._embedding_tokens = 0
        self._embedding_cost = 0.0
        self._cache_hit = False

    async def retrieve_if_needed(
        self,
        strategy: Dict[str, Any],
        keyword_matched: List[Dict[str, Any]],
        all_reviews: List[Dict[str, Any]],
        query: str,
        executor: SafeExpressionExecutor,
        sample_id: str = "",
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve additional reviews via embeddings if strategy says so.

        Checks the `use_embeddings_when` expression in the search strategy.
        If True, embeds unmatched reviews and retrieves top-k by similarity.

        Args:
            strategy: Search strategy dict from Formula Seed
            keyword_matched: Reviews already matched by keywords
            all_reviews: All reviews in the dataset
            query: The agenda/query text for similarity comparison
            executor: SafeExpressionExecutor for evaluating conditions
            sample_id: Sample ID for caching
            top_k: Number of additional reviews to retrieve

        Returns:
            Combined list: keyword_matched + embedding-retrieved reviews
        """
        self.reset_metrics()

        # Check if embeddings should be used
        use_embeddings_expr = strategy.get("use_embeddings_when", "False")
        context = {
            "keyword_matched": keyword_matched,
            "total_reviews": len(all_reviews),
        }
        should_embed = executor.execute_bool(use_embeddings_expr, context)

        if not should_embed:
            logger.debug("Embedding retrieval not needed per strategy")
            return keyword_matched

        # Get unmatched reviews
        matched_ids = {r.get("review_id") for r in keyword_matched}
        unmatched = [r for r in all_reviews if r.get("review_id") not in matched_ids]

        if not unmatched:
            logger.debug("No unmatched reviews to embed")
            return keyword_matched

        # Try cache first
        num_reviews = len(all_reviews)
        cache_key = self._get_cache_key(sample_id, num_reviews)
        cached = self._cache.get(cache_key)

        if cached:
            self._cache_hit = True
            query_embedding = cached["query_embedding"]
            review_embeddings_map = cached.get("review_embeddings_map", {})
        else:
            self._cache_hit = False

            # Embed query + unmatched reviews
            texts_to_embed = [query] + [r.get("text", "") for r in unmatched]
            embeddings = await self._embed_batch(texts_to_embed)

            query_embedding = embeddings[0]
            review_embeddings_map = {
                unmatched[i].get("review_id"): embeddings[i + 1]
                for i in range(len(unmatched))
            }

            # Track cost
            self._embedding_tokens = sum(estimate_tokens(t) for t in texts_to_embed)
            self._embedding_cost = (
                self._embedding_tokens / 1_000_000
            ) * EMBEDDING_COST_PER_1M_TOKENS

            # Save to cache
            self._cache[cache_key] = {
                "query_embedding": query_embedding,
                "review_embeddings_map": review_embeddings_map,
                "num_reviews": num_reviews,
            }
            self._save_cache()

        # Compute similarities for unmatched reviews
        similarities = []
        for review in unmatched:
            review_id = review.get("review_id")
            review_embedding = review_embeddings_map.get(review_id)
            if review_embedding:
                sim = cosine_similarity(query_embedding, review_embedding)
                similarities.append((sim, review))
            else:
                similarities.append((0.0, review))

        # Sort by similarity (descending) and take top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_retrieved = [review for _, review in similarities[:top_k]]

        # Add embedding similarity to reviews for priority scoring
        for i, (sim, review) in enumerate(similarities[:top_k]):
            review["_embedding_sim"] = sim

        logger.debug(
            f"Retrieved {len(top_retrieved)} reviews via embeddings "
            f"(cache_hit={self._cache_hit})"
        )

        return keyword_matched + top_retrieved
