import numpy as np
from typing import List, Dict, Union
import torch
import sentence_transformers
from transformers import AutoTokenizer, AutoModel

class SentenceTransformer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Generate embedding using huggingface or sentence-transformers
        
        Args:
            model_name: The name of the model to use for generating embeddings.
                        Default is a lightweight sentence-transformers model.
        """
        self.model_name = model_name
        try:
            # Try to use sentence-transformers for simplicity
            self.model = sentence_transformers.SentenceTransformer(model_name)
            self.use_sentence_transformer = True
        except:
            # Fall back to regular transformers
            # uncomment transformers part later
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.use_sentence_transformer = False
            
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate a vector embedding for the input text.
        
        Args:
            text: The input text to generate an embedding for
            
        Returns:
            A numpy array containing the embedding vector
        """
        if self.use_sentence_transformer:
            # use sentence-transformers
            return self.model.encode(text)
        # else:
        #     # Manual approach with transformers
        #     inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        #     with torch.no_grad():
        #         outputs = self.model(**inputs)
            
        #     # Use mean pooling of the last hidden state
        #     token_embeddings = outputs.last_hidden_state
        #     attention_mask = inputs['attention_mask']
            
        #     # Mask the padded tokens
        #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        #     sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        #     sum_mask = input_mask_expanded.sum(1)
        #     sum_mask = torch.clamp(sum_mask, min=1e-9)
            
        #     # Calculate `mean`
        #     embeddings = sum_embeddings / sum_mask
        #     return embeddings[0].numpy()
    
    # def batch_generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
    #     """
    #     Generate embeddings for a batch of texts.
        
    #     Args:
    #         texts: List of input texts
            
    #     Returns:
    #         List of numpy arrays containing the embedding vectors
    #     """
    #     if self.use_sentence_transformer:
    #         return self.model.encode(texts)
    #     # else:
    #     #     embeddings = []
    #     #     for text in texts:
    #     #         embeddings.append(self.generate_embedding(text))
    #     #     return embeddings

    def get_dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        random_embed = self.to_embeddings("random")
        return len(random_embed)
    