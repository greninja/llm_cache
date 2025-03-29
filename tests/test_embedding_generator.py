import pytest
import numpy as np
from llm_cache.embedding_generator import SentenceTransformer

@pytest.fixture
def embedder():
    return SentenceTransformer()

def test_embedding_generation(embedder):
    """Test basic embedding generation."""
    text = "This is a test sentence."
    embedding = embedder.generate_embedding(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == embedder.get_embedding_dimension()

def test_embedding_consistency(embedder):
    """Test that same input produces consistent embeddings."""
    text = "This is a test sentence."
    
    embedding1 = embedder.generate_embedding(text)
    embedding2 = embedder.generate_embedding(text)
    
    np.testing.assert_array_almost_equal(embedding1, embedding2)

def test_different_inputs(embedder):
    """Test that different inputs produce different embeddings."""
    text1 = "This is the first sentence."
    text2 = "This is a completely different sentence."
    
    embedding1 = embedder.generate_embedding(text1)
    embedding2 = embedder.generate_embedding(text2)
    
    # Embeddings should be different
    assert not np.array_equal(embedding1, embedding2) 