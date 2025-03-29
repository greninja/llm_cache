from llmcache import LLMCache

def demonstrate_cache_operations():
    """Example of direct cache operations."""
    
    # Initialize cache
    cache = LLMCache()
    
    # Store some example Q&A pairs
    questions = [
        "What is Python?",
        "How do I learn programming?",
        "What are good coding practices?"
    ]
    
    responses = [
        "Python is a high-level programming language...",
        "Start with basic concepts, practice regularly...",
        "Write clean, documented, and tested code..."
    ]
    
    # Store responses in cache
    cache_keys = []
    for q, r in zip(questions, responses):
        key = cache.store_query_response(q, r)
        cache_keys.append(key)
        print(f"Stored Q&A pair with key: {key}")
    
    # Search similar questions
    test_query = "Tell me about Python programming"
    results = cache.search_cache(
        test_query,
        top_k=2,
        similarity_threshold=0.7
    )
    
    if results:
        print("\nFound similar cached responses:")
        print(f"Query: {test_query}")
        print(f"Best match: {results['response']}")
        print(f"Similarity score: {results['similarity_score']}")

if __name__ == "__main__":
    demonstrate_cache_operations() 