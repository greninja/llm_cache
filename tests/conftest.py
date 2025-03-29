import pytest
import os
import redis
from unittest.mock import Mock

# Add fixtures that can be shared across test files here
@pytest.fixture(autouse=True)
def mock_redis(monkeypatch):
    """Mock Redis for all tests."""
    mock_redis = Mock(spec=redis.Redis)
    monkeypatch.setattr('redis.Redis', lambda *args, **kwargs: mock_redis)
    return mock_redis

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv('COHERE_API_KEY', 'test_key')
    monkeypatch.setenv('HUGGINGFACE_API_KEY', 'test_key') 