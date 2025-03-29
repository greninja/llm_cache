from llm_cache.embedding_generator import SentenceTransformer
from llm_cache.llmcache import LLMCache
from llm_cache.hf import HuggingFaceChat
from llm_cache.core import Core

def main():
    """Example of using a custom embedding model with the cache."""
    
    # Initialize custom embedding generator
    embedder = SentenceTransformer(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Initialize cache with custom embedder
    cache = LLMCache(
        embedding_generator=embedder,
        embedding_dimension=768,  # MPNet's embedding dimension
        ttl_seconds=7200  # 2 hour cache lifetime
    )
    
    # Initialize other components
    llm = HuggingFaceChat(model_name="meta-llama/Llama-3.2-3B-Instruct")
    core = Core(llm, cache)
    
    # Start chat
    core.run()

if __name__ == "__main__":
    main() 