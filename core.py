from typing import Optional, Any, Dict
import logging
from hf import HuggingFaceChat
from llmcache import LLMCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Core:
    """
    Core class that handles the interaction between LLM and cache system.
    Manages response generation and caching logic.
    """
    
    def __init__(
        self,
        llm_model: Any,
        cache: LLMCache,
        top_k: int = 3,
        similarity_threshold: float = 0.8
    ):
        """
        Initialize the Core system.
        
        Args:
            llm_model: Language model instance
            cache: Cache system instance
            top_k: Number of top similar responses to consider
            similarity_threshold: Minimum similarity score for cache hits
        """
        self.llm_model = llm_model
        self.cache = cache
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def get_response(self, query: str) -> str:
        """
        Get response for a query, either from cache or LLM.
        
        Args:
            query: User input query
            
        Returns:
            str: Response text
        """
        cache_response = self._check_cache(query)
        if cache_response:
            logger.info("Cache hit, returning cached response")
            return cache_response["response"]
        
        logger.info("Cache miss, getting response from LLM")
        return self._get_and_cache_llm_response(query)
            
    def _check_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check if a valid response exists in cache.
        
        Args:
            query: User input query
            
        Returns:
            Optional[Dict]: Cache response if found, None otherwise
        """
        return self.cache.search_cache(
            query,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold
        )
    
    def _get_and_cache_llm_response(self, query: str) -> str:
        """
        Get response from LLM and cache it.
        
        Args:
            query: User input query
            
        Returns:
            str: Generated response or error message
        """
        try:
            response = self._get_response_from_llm(query)
            if self._is_valid_response(response):
                generated_text = response[0]['generated_text']
                self.cache.store_query_response(query, generated_text)
                return generated_text
            return "Error: Invalid response format from LLM"
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}", exc_info=True)
            return f"Error: Failed to get response from LLM - {str(e)}"
    
    def _get_response_from_llm(self, query: str) -> Any:
        """
        Get raw response from LLM.
        
        Args:
            query: User input query
            
        Returns:
            Any: Raw LLM response
        """
        return self.llm_model.get_response(query)
    
    @staticmethod
    def _is_valid_response(response: Any) -> bool:
        """
        Validate LLM response format.
        
        Args:
            response: Raw LLM response
            
        Returns:
            bool: True if response is valid
        """
        return (
            response 
            and isinstance(response, list) 
            and response[0].get('generated_text') is not None
        )
    
    def run(self) -> None:
        """Start interactive chat session."""
        logger.info("Chatbot initialized. Type 'exit' to end the chat.")
        
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'exit':
                logger.info("Ending chat session")
                break
            
            response = self.get_response(user_input)
            print("Bot:", response)
                
if __name__ == "__main__":
    llm_model = HuggingFaceChat(model_name="meta-llama/Llama-3.2-3B-Instruct")
    cache = LLMCache(enable_rerank=True)
    core = Core(llm_model, cache)
    core.run()