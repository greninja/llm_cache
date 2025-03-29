from llm_cache.core import Core
from llm_cache.hf import HuggingFaceChat
from llm_cache.llmcache import LLMCache

def main():
    """Simple example of using the LLM cache system with a chat interface."""
    
    # Initialize the components
    llm = HuggingFaceChat(model_name="meta-llama/Llama-3.2-3B-Instruct")
    cache = LLMCache(enable_rerank=True)
    core = Core(llm, cache)
    
    # Start the chat session
    core.run()

if __name__ == "__main__":
    main() 