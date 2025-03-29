from venv import create
from typing import Optional, List, Dict, Any
import uuid
import time
import numpy as np
from redis import Redis
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError
from redis_om import get_redis_connection

from llm_cache.embedding_generator import SentenceTransformer
from llm_cache.cohere_rerank import CohereRerank

class LLMCache:
    """A cache system for LLM responses using Redis vector similarity search."""
    
    CACHE_INDEX_NAME = "llm_cache_idx"
    CACHE_PREFIX = "llm_cache:"
    
    def __init__(
        self,
        redis_conn: Optional[Redis] = None,
        embedding_dimension: int = 384,
        embedding_generator: Optional[Any] = None,
        ttl_seconds: int = 3600,
        eviction_policy: str = "allkeys-lru",
        max_memory_bytes: int = 1_000_000_000,
        max_memory_samples: int = 10,
        enable_rerank: bool = False
    ):
        """
        Initialize the LLM cache with Redis connection and configuration.
        
        Args:
            redis_conn: Redis connection object. If None, creates a local connection
            embedding_dimension: Dimension of the embedding vectors
            embedding_generator: Custom embedding generator. If None, uses SentenceTransformer
            ttl_seconds: Time-to-live for cache entries in seconds
            eviction_policy: Redis eviction policy
            max_memory_bytes: Maximum memory limit for Redis in bytes
            max_memory_samples: Number of samples for Redis LRU algorithm
            enable_rerank: Whether to use Cohere's reranking
        """
        self.redis_conn = redis_conn or get_redis_connection(
            host="localhost", port=6379, decode_responses=True
        )
        self.ttl_seconds = ttl_seconds
        self.embedding_dimension = embedding_dimension
        self.embedding_generator = embedding_generator or SentenceTransformer()
        self.enable_rerank = enable_rerank
        
        self._configure_redis(eviction_policy, max_memory_bytes, max_memory_samples)
        self._initialize_search_index()
        
        if self.enable_rerank:
            self._initialize_reranker()

    def _configure_redis(self, policy: str, max_memory: int, sample_size: int) -> None:
        """Configure Redis memory and eviction settings."""
        if max_memory:
            self.redis_conn.config_set("maxmemory", max_memory)
        if policy:
            self.redis_conn.config_set("maxmemory-policy", policy)
        if sample_size:
            self.redis_conn.config_set("maxmemory-samples", sample_size)

    def _initialize_search_index(self) -> None:
        """Initialize the Redis search index for vector similarity search."""
        try:
            self.redis_conn.ft(self.CACHE_INDEX_NAME).info()
        except ResponseError:
            schema = (
                VectorField("embedding", "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": self.embedding_dimension,
                    "DISTANCE_METRIC": "COSINE"
                })
            )
            self.redis_conn.ft(self.CACHE_INDEX_NAME).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.CACHE_PREFIX],
                    index_type=IndexType.HASH
                )
            )

    def _initialize_reranker(self) -> None:
        """Initialize the Cohere reranking system."""
        import os
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is required for reranking")
        self.reranker = CohereRerank(
            model="rerank-multilingual-v3.0",
            api_key=cohere_api_key
        )

    def store_query_response(self, question: str, response: str) -> str:
        """
        Store a question-response pair in the cache.
        
        Args:
            question: The original question
            response: The LLM's response
            
        Returns:
            str: The cache entry key
        """
        embedding = self.embedding_generator.generate_embedding(question)
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        
        cache_key = f"{self.CACHE_PREFIX}{uuid.uuid4().hex}"
        cache_entry = {
            "embedding": embedding_bytes,
            "question": question,
            "response": response,
            "timestamp": int(time.time())
        }
        
        self.redis_conn.hset(cache_key, mapping=cache_entry)
        return cache_key

    def search_cache(
        self,
        query_text: str,
        top_k: int = 3,
        similarity_threshold: float = 0.8
    ) -> Optional[Dict[str, Any]]:
        """
        Search the cache for similar questions and their responses.
        
        Args:
            query_text: The question to search for
            top_k: Number of similar entries to retrieve
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            Optional[Dict]: The best matching cache entry or None
        """
        query_vector = np.array(
            self.embedding_generator.generate_embedding(query_text),
            dtype=np.float32
        )
        
        search_results = self._vector_search(query_vector, top_k)
        if not search_results:
            return None
            
        return self.post_process_cache_matches(
            query_text,
            search_results,
            similarity_threshold
        )

    def _vector_search(self, query_vector: np.ndarray, top_k: int) -> List[Any]:
        """Perform vector similarity search in Redis."""
        query = (
            Query(f"*=>[KNN {top_k} @embedding $query_vec AS score]")
            .sort_by("score")
            .return_fields("question", "response", "timestamp", "score")
            .paging(0, top_k)
            .dialect(2)
        )
        
        results = self.redis_conn.ft(self.CACHE_INDEX_NAME).search(
            query,
            query_params={"query_vec": query_vector.tobytes()}
        )
        return results.docs

    def post_process_cache_matches(
        self,
        query_text: str,
        cache_matches: List[Any],
        similarity_threshold: float
    ) -> Optional[Dict[str, Any]]:
        """
        Post-process and filter cache matches based on similarity scores.
        
        Args:
            query_text: Original query text
            cache_matches: List of potential cache matches
            similarity_threshold: Minimum similarity score required
            
        Returns:
            Optional[Dict]: Best matching cache entry or None
        """
        # Convert similarity threshold to cosine distance threshold
        # (Redis uses cosine distance where lower scores mean higher similarity)
        cosine_distance_threshold = 1 - similarity_threshold
        
        valid_matches = [
            match for match in cache_matches 
            if float(match.score) < cosine_distance_threshold
        ]

        if not valid_matches:
            return None

        if self.enable_rerank:
            return self._rerank_matches(query_text, valid_matches)
        
        return self._convert_match_to_dict(valid_matches[0])

    def _rerank_matches(self, query_text: str, matches: List[Any]) -> Optional[Dict[str, Any]]:
        """Rerank matches using Cohere's reranking system."""
        rerank_input = {
            "question": query_text,
            "answers": [match.response for match in matches]
        }
        
        reranked_results = self.reranker.evaluation(
            rerank_input,
            {"answers": rerank_input["answers"]},
            top_n=1
        )
        
        if reranked_results:
            best_match_index = reranked_results[0].index
            return self._convert_match_to_dict(matches[best_match_index])
        
        return self._convert_match_to_dict(matches[0])

    @staticmethod
    def _convert_match_to_dict(match: Any) -> Dict[str, Any]:
        """Convert a Redis search result to a dictionary."""
        return {
            "question": match.question,
            "response": match.response,
            "timestamp": int(match.timestamp),
            "similarity_score": float(match.score)
        }