import pytest
from unittest.mock import Mock, patch

from llm_cache.core import Core
from llm_cache.llmcache import LLMCache

@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.get_response.return_value = [{"generated_text": "Test response"}]
    return llm

@pytest.fixture
def mock_cache():
    cache = Mock(spec=LLMCache)
    cache.search_cache.return_value = None
    return cache

@pytest.fixture
def core(mock_llm, mock_cache):
    return Core(mock_llm, mock_cache)

def test_cache_hit(core, mock_cache):
    """Test behavior when response is found in cache."""
    mock_cache.search_cache.return_value = {
        "response": "Cached response",
        "similarity_score": 0.95
    }
    
    response = core.get_response("test query")
    assert response == "Cached response"
    assert not core.llm_model.get_response.called

def test_cache_miss(core, mock_cache, mock_llm):
    """Test behavior when response is not found in cache."""
    mock_cache.search_cache.return_value = None
    
    response = core.get_response("test query")
    assert response == "Test response"
    assert mock_llm.get_response.called

def test_llm_error_handling(core, mock_llm):
    """Test handling of LLM errors."""
    mock_llm.get_response.side_effect = Exception("LLM Error")
    
    response = core.get_response("test query")
    assert "Error" in response 