# LLMCache

A simple LLM cache design intergrated with a command line LLM chatbot, that uses vector similarity search to efficiently store and retrieve responses, reducing API calls and improving response times.

## Overview

![LLM Cache System Architecture](./assets/sketch.png)

### System Components:

1. **Core**: Orchestrates the interaction between the user and Cache/LLM
2. **LLM Cache**: Object to store, search and retrieve cached responses
3. **Post Processing**: Reranking, filtering and formatting of responses
4. **LLM**: Handles communication with language models (currently supports only Hugging Face models)

### Other features/highlights:

- **Vector-based search**: Uses embeddings to store and find semantically similar queries
- **Redis Integration**: Leverages Redis for fast vector similarity search
- **Semantic Reranking**: Optional Cohere-based reranking for better match quality between query and cached responses

## Installation

Install dependencies

```bash
pip install -r requirements.txt
```

for conda:

```bash
conda env create -f environment.yml
```

## Quickstart

Make sure redis server is running (more details in the [Redis docs](https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/))

```python
from core import Core
from hf import HuggingFaceChat
from llmcache import LLMCache


# Initialize components
llm_model = HuggingFaceChat(model_name="meta-llama/Llama-3.2-3B-Instruct")
cache = LLMCache(enable_rerank=True)
core = Core(llm_model, cache)

# Start the chat
core.chat()
```

## Configuration

### LLM Cache Settings

```python
cache = LLMCache(
redis_conn=None, # Optional Redis connection
embedding_dimension=384, # Embedding vector size
ttl_seconds=3600, # Cache entry lifetime
eviction_policy="allkeys-lru", # Redis eviction policy
max_memory_bytes=1_000_000_000, # Redis memory limit
enable_rerank=True # Enable Cohere reranking
)
```

### Core Settings

```python
core = Core(
    llm_model=llm_model,
    cache=cache,
    top_k=3,  # Number of similar cache entries to consider
    similarity_threshold=0.8  # Minimum similarity score for cache hits
)
```

### LLM Interface Settings

```python
llm_model = HuggingFaceChat(
    model_name="meta-llama/Llama-3.2-3B-Instruct"
)
```

### Store a response

```python
cache_key = cache.store_query_response(question="What is Python?", response="Python is...")
```

### Search cache

```python
results = cache.search_cache(
query_text="Tell me about Python",
top_k=3,
similarity_threshold=0.8
)
```

## Clone the repository

```bash 
git clone https://github.com/yourusername/llm_cache.git
cd llm_cache
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License © 2025 Shadab Shaikh.

## Acknowledgments

Much of my implementation was inspired by Zilliz's [GPTCache](https://github.com/zilliztech/GPTCache). My motivation for building LLMCache stemmed from a deep curiosity to explore and understand the various system components that power fast LLM inference, with the goal of meaningfully reducing latency and costs.
