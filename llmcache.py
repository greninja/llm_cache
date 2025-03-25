from venv import create
from embedding_generator import SentenceTransformer
import uuid
import time
import numpy as np

from redis import Redis
from redis.commands.search.field import TagField, TextField, NumericField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from redis_om import get_redis_connection, HashModel, Field

from cohere_rerank import CohereRerank

# class LLMCacheEntry(HashModel):
#     question: str = Field(index=True)  # Original question
#     response: str  # Cached response
#     timestamp: int = Field(index=True)  # Timestamp for TTL policies

#     class Meta:
#         database = redis_conn
#         model_key = "llm_cache"

class LLMCache:
    def __init__(self,
                 redis_conn: Redis = None,
                 embedding_dimension: int = 384,
                 embedding_generator = None,
                 ttl: int = 3600,
                 policy: str = "allkeys-lru",
                 maxmemory: int = 1000000000,
                 maxmemory_samples: int = 10,
                 do_rerank: bool = False): 
        if redis_conn is None:
            self.redis_conn = get_redis_connection( 
                host="localhost", port=6379, decode_responses=True
            )
        else:
            self.redis_conn = redis_conn
        self.ttl = ttl # have to add eviction logic
        self.initialize_eviction_params(policy, maxmemory, maxmemory_samples)
        self.embedding_dimension = embedding_dimension
        self.embedding_generator = embedding_generator if embedding_generator is not None else SentenceTransformer()
        self.do_rerank = do_rerank
        self.create_index()
        
        if self.do_rerank:
            import os
            co_api_key = os.getenv("COHERE_API_KEY")
            self.rerank_evaluation = CohereRerank(model="rerank-multilingual-v3.0", api_key=co_api_key)
        
    def initialize_eviction_params(self, policy, maxmemory, maxmemory_samples):
        if maxmemory:
            self.redis_conn.config_set("maxmemory", maxmemory)
        if policy:
            self.redis_conn.config_set("maxmemory-policy", policy)
        if maxmemory_samples:
            self.redis_conn.config_set("maxmemory-samples", maxmemory_samples)

    def create_index(self):
        try:
            self.redis_conn.ft("llm_cache_idx").info()  # Check if index exists
            print("Index already exists!")  # TODO: this should go in logging ideally
        except:
            schema = (
                VectorField("embedding", "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": 384,  # Adjust to match your embedding model
                    "DISTANCE_METRIC": "COSINE"
                })
            )
            self.redis_conn.ft("llm_cache_idx").create_index(
                schema,
                definition=IndexDefinition(prefix=["llm_cache:"], index_type=IndexType.HASH)
            )

    def store_query_response(self, question, response):
        # Generate an embedding
        embedding = self.embedding_generator.generate_embedding(question)  # Generate vector embedding for question
        embedding = np.array(embedding, dtype=np.float32).tobytes()

        # Generate a unique primary key
        pk = f"llm_cache:{uuid.uuid4().hex}"

        # Store data in Redis as a Hash
        self.redis_conn.hset(pk, mapping={
            "embedding": embedding,
            "question": question,
            "response": response,
            "timestamp": int(time.time())
        })

        # print(f"Stored entry with key: {pk}") --> this should go in logging ideally
        return pk

    def search_cache(self, query_text: str, top_k: int = 3, similarity_threshold=0.8):

        # generate embedding for the query
        query_vector = np.array(self.embedding_generator.generate_embedding(query_text), dtype=np.float32)

        query = (
            Query("*=>[KNN {} @embedding $query_vec AS score]".format(top_k))
            .sort_by("score")
            .return_fields("question", "response", "timestamp", "score")
            .paging(0, top_k)
            .dialect(2)
        )

        # uses vector search to find top k matches based on embedding similarity
        cache_matches = self.redis_conn.ft("llm_cache_idx").search(
            query, query_params={"query_vec": query_vector.tobytes()}
        ).docs

        if not cache_matches:
            return None
        
        # further filtering/post processing
        # 1st filterting step: only consider matches with similarity score > threshold
        # 2nd filtering step: only consider matches with relevance score higher than a threshold
        # of matched results's answers with query text
        best_match = self.post_process_cache_matches(query_text, cache_matches, similarity_threshold)
            
        # # Return all matches with their scores for post-processing
        # cache_matches = []
        # for match in cache_matches:
        #     similarity_score = float(match.score)
        #     cache_matches.append({
        #         "question": match.question,
        #         "response": match.response,
        #         "timestamp": match.timestamp,
        #         "similarity_score": similarity_score
        #     })
        
        return best_match

    def post_process_cache_matches(self, query_text: str, cache_matches, similarity_threshold=0.8):
        """
        Post-process cache matches with additional similarity checks
        
        Args:
            query_text: Original query text
            cache_matches: List of potential cache matches
            similarity_threshold: Minimum similarity score required
            
        Returns:
            Best matching cache entry or None if no good matches found
        """
        # Additional similarity checks can be implemented here
        # For example, you could:
        # 1. Compare exact text matches
        # 2. Apply different similarity metrics
        # 3. Check for keyword overlap
        # 4. Apply business logic filters

        # For cosine distance in Redis, LOWER scores mean MORE similar
        # So we want scores LESS THAN (1 - similarity_threshold)
        cosine_distance_threshold = 1 - similarity_threshold
        
        valid_matches = [
            match for match in cache_matches 
            if float(match.score) < cosine_distance_threshold
        ]

        if valid_matches:
            if self.do_rerank:
                # rerank the valid matches
                src_dict = {"question": query_text}
                cache_dict = {"answers": [match.response for match in valid_matches]}
                reranked_matches = self.rerank_evaluation.evaluation(src_dict, cache_dict, top_n=1)
                if reranked_matches is not None:
                    index = reranked_matches[0]["index"]
                    return valid_matches[index]
                else:
                    return valid_matches[0]
            else:
                return valid_matches[0]
        return None