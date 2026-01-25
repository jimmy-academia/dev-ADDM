"""RAG (Retrieval Augmented Generation) baseline method."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

import numpy as np
from openai import AsyncOpenAI

from addm.data.types import Sample
from addm.llm import LLMService
from addm.methods.base import Method
from addm.utils.debug_logger import get_debug_logger
from addm.utils.results_manager import get_results_manager


# Embedding pricing (text-embedding-3-large): $0.13 per 1M tokens
EMBEDDING_COST_PER_1M_TOKENS = 0.13


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 chars)."""
    return len(text) // 4


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


class RAGMethod(Method):
    """RAG baseline - retrieves top-k reviews via embeddings before LLM call."""

    name = "rag"

    def __init__(
        self,
        top_k: int = 20,
        embedding_model: str = "text-embedding-3-large",
        cache_path: Optional[Path] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize RAG method.

        Args:
            top_k: Number of reviews to retrieve (default: 20, ~10% of K=200)
            embedding_model: OpenAI embedding model (default: text-embedding-3-large)
            cache_path: Path to embedding cache file (default: results/shared/yelp_embeddings.json)
            system_prompt: System prompt for structured output (optional)
        """
        self.top_k = top_k
        self.embedding_model = embedding_model
        # Use shared cache path from results manager
        if cache_path is None:
            results_manager = get_results_manager()
            self.cache_path = results_manager.get_shared_embeddings_path()
        else:
            self.cache_path = cache_path
        self.system_prompt = system_prompt
        self._embedding_client: Optional[AsyncOpenAI] = None
        self._cache: Dict[str, Any] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                self._cache = json.load(f)

    def _save_cache(self) -> None:
        """Save cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def _get_embedding_cache_key(self, sample_id: str, num_reviews: int) -> str:
        """Generate cache key for embeddings.

        Args:
            sample_id: Sample identifier
            num_reviews: Number of reviews in the dataset (to prevent cross-K cache hits)
        """
        return f"rag_embeddings_{sample_id}_K{num_reviews}"

    def _get_retrieval_cache_key(self, sample_id: str, num_reviews: int, k: int) -> str:
        """Generate cache key for retrieval results.

        Args:
            sample_id: Sample identifier
            num_reviews: Number of reviews in the source dataset (K value)
            k: Number of reviews to retrieve (top_k)
        """
        return f"rag_retrieval_{sample_id}_K{num_reviews}_topk{k}"

    def _load_cached_embeddings(self, sample_id: str, num_reviews: int) -> Optional[Dict[str, Any]]:
        """Load cached embeddings for a sample."""
        key = self._get_embedding_cache_key(sample_id, num_reviews)
        return self._cache.get(key)

    def _save_embeddings_to_cache(self, sample_id: str, num_reviews: int, data: Dict[str, Any]) -> None:
        """Save embeddings to cache."""
        key = self._get_embedding_cache_key(sample_id, num_reviews)
        self._cache[key] = data
        self._save_cache()

    def _load_cached_retrieval(self, sample_id: str, num_reviews: int, k: int) -> Optional[Dict[str, Any]]:
        """Load cached retrieval results."""
        key = self._get_retrieval_cache_key(sample_id, num_reviews, k)
        return self._cache.get(key)

    def _save_retrieval_to_cache(self, sample_id: str, num_reviews: int, k: int, data: Dict[str, Any]) -> None:
        """Save retrieval results to cache."""
        key = self._get_retrieval_cache_key(sample_id, num_reviews, k)
        self._cache[key] = data
        self._save_cache()

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
            List of embedding vectors (each is a list of floats)
        """
        client = self._get_embedding_client()
        response = await client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        # Extract embeddings in order
        embeddings = [item.embedding for item in response.data]
        return embeddings

    def _parse_restaurant(self, context: str) -> Dict[str, Any]:
        """Parse restaurant data from context JSON string.

        Args:
            context: JSON string containing restaurant data

        Returns:
            Parsed restaurant dict

        Raises:
            ValueError: If context is invalid JSON
        """
        try:
            return json.loads(context)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid context JSON: {e}")

    def _build_reduced_context(
        self,
        restaurant: Dict[str, Any],
        top_reviews: List[Dict[str, Any]],
    ) -> str:
        """Build context with only top-k retrieved reviews.

        Args:
            restaurant: Full restaurant data dict
            top_reviews: List of top-k retrieved reviews

        Returns:
            JSON string with restaurant data and only top-k reviews
        """
        # Create new restaurant dict with only retrieved reviews
        reduced = {
            "business": restaurant.get("business", {}),
            "reviews": top_reviews,
        }
        return json.dumps(reduced)

    def _build_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        """Build messages for LLM call.

        Uses system_prompt if provided (for structured output), otherwise
        falls back to simple evaluator prompt.

        Args:
            query: The agenda/query text
            context: Restaurant context (JSON string)

        Returns:
            List of message dicts for LLM call
        """
        if self.system_prompt:
            # Use structured output format matching direct method
            user_content = f"Context:\n\n{context}\n\nAgenda:\n{query}"
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ]
        else:
            # Fallback to simple prompt
            return [
                {"role": "system", "content": "You are a precise evaluator. Answer strictly based on the provided context."},
                {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}\n\nReturn answer required by the Query."},
            ]

    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        """Run RAG evaluation on a sample.

        Args:
            sample: Input sample with query and context (JSON string)
            llm: LLM service

        Returns:
            Dict with sample_id, output, and usage metrics
        """
        # Set debug logger context - all LLM calls go to debug/{sample_id}.jsonl
        if debug_logger := get_debug_logger():
            debug_logger.set_context(sample.sample_id)

        start_time = time.time()

        # Parse restaurant data
        try:
            restaurant = self._parse_restaurant(sample.context or "")
        except ValueError as e:
            return self._make_result(
                sample.sample_id,
                "",
                {},
                llm_calls=0,
                error=str(e),
            )

        reviews = restaurant.get("reviews", [])
        num_reviews = len(reviews)

        # Edge case: If K ≤ top_k, use all reviews (no retrieval needed)
        if num_reviews <= self.top_k:
            context = sample.context or ""
            messages = self._build_messages(sample.query, context)

            # Call LLM
            response, usage = await llm.call_async_with_usage(
                messages,
                context={"sample_id": sample.sample_id, "method": self.name},
            )

            usage["latency_ms"] = (time.time() - start_time) * 1000

            return self._make_result(
                sample.sample_id,
                response,
                usage,
                embedding_tokens=0,
                embedding_cost_usd=0.0,
                reviews_retrieved=num_reviews,
                reviews_total=num_reviews,
                top_review_indices=list(range(num_reviews)),
                cache_hit_embeddings=False,
                cache_hit_retrieval=False,
            )

        # Try to load cached retrieval results first
        cached_retrieval = self._load_cached_retrieval(sample.sample_id, num_reviews, self.top_k)
        if cached_retrieval:
            # Cache hit - use cached retrieval results
            top_indices = cached_retrieval["top_indices"]
            top_reviews = [reviews[i] for i in top_indices]
            reduced_context = self._build_reduced_context(restaurant, top_reviews)

            messages = self._build_messages(sample.query, reduced_context)

            # Call LLM
            response, usage = await llm.call_async_with_usage(
                messages,
                context={"sample_id": sample.sample_id, "method": self.name},
            )

            usage["latency_ms"] = (time.time() - start_time) * 1000

            return self._make_result(
                sample.sample_id,
                response,
                usage,
                embedding_tokens=0,
                embedding_cost_usd=0.0,
                reviews_retrieved=len(top_indices),
                reviews_total=num_reviews,
                top_review_indices=top_indices,
                cache_hit_embeddings=True,
                cache_hit_retrieval=True,
            )

        # Try to load cached embeddings
        cached_embeddings = self._load_cached_embeddings(sample.sample_id, num_reviews)
        cache_hit_embeddings = cached_embeddings is not None

        if cached_embeddings:
            # Use cached embeddings
            query_embedding = cached_embeddings["query_embedding"]
            review_embeddings = cached_embeddings["review_embeddings"]
            embedding_tokens = 0
            embedding_cost = 0.0
        else:
            # Compute embeddings
            query_text = sample.query
            review_texts = [r.get("text", "") for r in reviews]

            # Embed query + all reviews in batch
            all_texts = [query_text] + review_texts
            embeddings = await self._embed_batch(all_texts)

            query_embedding = embeddings[0]
            review_embeddings = embeddings[1:]

            # Estimate embedding cost
            embedding_tokens = sum(estimate_tokens(t) for t in all_texts)
            embedding_cost = (embedding_tokens / 1_000_000) * EMBEDDING_COST_PER_1M_TOKENS

            # Save embeddings to cache
            self._save_embeddings_to_cache(
                sample.sample_id,
                num_reviews,
                {
                    "query_embedding": query_embedding,
                    "review_embeddings": review_embeddings,
                    "num_reviews": num_reviews,
                },
            )

        # Compute similarities and rank
        similarities = [
            cosine_similarity(query_embedding, review_emb)
            for review_emb in review_embeddings
        ]

        # Get top-k indices (sorted by similarity, descending)
        top_indices = np.argsort(similarities)[-self.top_k :].tolist()
        top_indices.reverse()  # Highest similarity first

        top_similarities = [similarities[i] for i in top_indices]

        # Save retrieval results to cache
        self._save_retrieval_to_cache(
            sample.sample_id,
            num_reviews,
            self.top_k,
            {
                "top_indices": top_indices,
                "similarities": top_similarities,
                "num_reviews": num_reviews,
            },
        )

        # Build reduced context with only top-k reviews
        top_reviews = [reviews[i] for i in top_indices]
        reduced_context = self._build_reduced_context(restaurant, top_reviews)

        # Build prompt (same format as DirectMethod)
        messages = self._build_messages(sample.query, reduced_context)

        # Call LLM with reduced context
        response, usage = await llm.call_async_with_usage(
            messages,
            context={"sample_id": sample.sample_id, "method": self.name},
        )

        usage["latency_ms"] = (time.time() - start_time) * 1000
        usage["cost_usd"] = usage.get("cost_usd", 0.0) + embedding_cost

        return self._make_result(
            sample.sample_id,
            response,
            usage,
            embedding_tokens=embedding_tokens,
            embedding_cost_usd=embedding_cost,
            reviews_retrieved=len(top_indices),
            reviews_total=num_reviews,
            top_review_indices=top_indices,
            cache_hit_embeddings=cache_hit_embeddings,
            cache_hit_retrieval=False,  # Just computed fresh retrieval
        )
