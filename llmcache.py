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
                 maxmemory: int = 1000000000, # figure out how to set this
                 maxmemory_samples: int = 10): # figure out how to set this
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
        self.create_index()

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
            print("Index already exists!")
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
        query_vector = np.array(self.embedding_generator.generate_embedding(query_text), dtype=np.float32).tobytes()

        query = (
            Query("*=>[KNN {} @embedding $query_vec AS score]".format(top_k))
            .sort_by("score")
            .return_fields("question", "response", "timestamp", "score")
            .paging(0, top_k)
            .dialect(2)
        )

        results = self.redis_conn.ft("llm_cache_idx").search(
            query, query_params={"query_vec": query_vector}
        ).docs

        if results:
            best_match = results[0]
            similarity_score = float(best_match.score)

            if similarity_score > similarity_threshold:
                return None
            else:
                return {
                    "question": best_match.question,
                    "response": best_match.response,
                    "timestamp": best_match.timestamp,
                    "similarity_score": similarity_score
                }

        return None  # No matches at all

#llm_cache = LLMCache()
#print(llm_cache.store_query_response("What is the capital of France?", "Paris"))
#print(llm_cache.search_cache("What is the capital of France?", 1))