import pytest
import numpy as np
from llm_cache.llmcache import LLMCache
from redis.exceptions import ConnectionError

@pytest.fixture
def cache():
    return LLMCache()

def test_store_and_retrieve(cache):
    """Test basic storage and retrieval functionality."""
    question = "What is Python?"
    response = "Python is a programming language."
    
    # Store the Q&A pair
    key = cache.store_query_response(question, response)
    assert key is not None
    
    # Retrieve using the same question
    result = cache.search_cache(question)
    assert result is not None
    assert result["response"] == response

def test_similarity_search(cache):
    """Test semantic similarity search."""
    # Store a response
    cache.store_query_response(
        "What is Python?",
        "Python is a programming language."
    )
    
    # Search with a similar question
    result = cache.search_cache(
        "Tell me about Python programming",
        similarity_threshold=0.7
    )
    
    assert result is not None
    assert "Python" in result["response"]

def test_invalid_input(cache):
    """Test handling of invalid inputs."""
    with pytest.raises(ValueError):
        cache.store_query_response("", "Empty question")
    
    with pytest.raises(ValueError):
        cache.store_query_response("Question", "")

def test_cache_miss(cache):
    """Test behavior when no similar cached response exists."""
    result = cache.search_cache(
        "Something completely unrelated",
        similarity_threshold=0.9
    )
    assert result is None 