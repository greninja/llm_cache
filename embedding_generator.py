from typing import List, Optional, Union
import numpy as np
import torch
import sentence_transformers
from transformers import AutoTokenizer, AutoModel

class SentenceTransformer:
    """
    A class to generate embeddings using either sentence-transformers or Hugging Face transformers.
    """
    
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_LENGTH = 512

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the model to use (default: all-MiniLM-L6-v2)
        """
        self.model_name = model_name
        self.model: Optional[Union[
            sentence_transformers.SentenceTransformer,
            AutoModel
        ]] = None
        self.tokenizer = None
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the appropriate model based on availability."""
        try:
            self.model = sentence_transformers.SentenceTransformer(self.model_name)
            self.use_sentence_transformer = True
            
        except Exception as e:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.use_sentence_transformer = False
            
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for input text.
        
        Args:
            text: Input text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
            
        if self.use_sentence_transformer:
            return self.model.encode(text)
        
        return self._generate_transformer_embedding(text)
    
    def _generate_transformer_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding using Hugging Face transformers.
        
        Args:
            text: Input text
            
        Returns:
            np.ndarray: Embedding vector
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.MAX_LENGTH
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return self._mean_pooling(
            outputs.last_hidden_state,
            inputs['attention_mask']
        ).numpy()[0]
    
    @staticmethod
    def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings.
        
        Args:
            token_embeddings: Token-level embeddings
            attention_mask: Attention mask for tokens
            
        Returns:
            torch.Tensor: Pooled embedding
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vector.
        
        Returns:
            int: Embedding dimension
        """
        sample_embedding = self.generate_embedding("sample text")
        return len(sample_embedding)
    