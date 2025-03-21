# core.py 

from hf import HuggingFaceChat
from llmcache import LLMCache

class Core:
    def __init__(self, llm_obj, cache, similarity_threshold=0.8):
        self.llm_obj = llm_obj
        self.cache = cache
        self.similarity_threshold = similarity_threshold
    
    def get_response(self, query):
        
        # will do similarity matching here with llm cache
        cache_response = self.cache.search_cache(query, similarity_threshold=self.similarity_threshold)
        if cache_response:
            # TODO: post process the cache response --> its returning a dictionary with the response
            # print("Cache hit, returning cached response") # this should go in logging ideally
            # check if the response is valid
            return cache_response
        else:
            # print("Cache miss, getting response from LLM") # this should go in logging ideally
            try:
                response = self.get_response_from_llm(query)
            except (IndexError, KeyError, TypeError) as e:
                print("Bot: An error occurred while processing the response from LLM")
                print(f"Error details: {e}")                
            # store the response in the cache
            # Check if the response is valid
            valid_response = (
                response and isinstance(response, list) and 'generated_text' in response[0]
            )

            if valid_response:
                generated_text = response[0]['generated_text']
                self.cache.store_query_response(query, generated_text)
                return generated_text
            else:
                return "Error: Invalid response from LLM"
            
    def get_response_from_llm(self, query):
        response = self.llm_obj.get_response(query)
        # TODO: post process the response
        return response
    
    def run(self):
        
        print("Chatbot initialized. Type 'exit' to end the chat.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Ending chat. Goodbye!")
                break
            
            # here all the magic happens (response comes either from cache or from LLM)
            response = self.get_response(user_input)
            print("Bot:", response)
                
if __name__ == "__main__":
    llm_obj = HuggingFaceChat(model_name="meta-llama/Llama-3.2-3B-Instruct")
    cache_obj = LLMCache()
    core_obj = Core(llm_obj, cache_obj)
    core_obj.run()