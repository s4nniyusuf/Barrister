# Barrister

Barrister is a citation-aware retrieval-augmented generation (RAG) system for Nigerian legal information, built on a curated corpus of Nigerian legal texts.

## Tech Stack

| Area | Technology | Role |
|---|---|---|
| Language | Python 3.12+ | Application and RAG implementation |
| Package management | uv | Dependency and environment management |
| Frontend | Streamlit | Chat interface |
| Backend API | FastAPI, Pydantic, Uvicorn | HTTP endpoints, request validation, and ASGI serving |
| RAG orchestration | LangChain | Prompts, query transformation, retrievers, and answer pipeline |
| Keyword retrieval | BM25 (`rank-bm25`) | Finds passages with matching legal terms |
| Text preprocessing | spaCy (`en_core_web_sm`) | Tokenizes and lowercases text for BM25 |
| Embeddings | Sentence Transformers: `BAAI/bge-small-en-v1.5` | Encodes chunks and queries for semantic search |
| Vector store | Chroma | Persists embeddings, chunk text, and metadata |
| Language model | OpenRouter, configured with NVIDIA Nemotron 3 Ultra | Query rewriting, query expansion, and answer generation |
| Chunking | LangChain Text Splitters, `tiktoken` | Creates overlapping, token-sized legal text chunks |


## The Problem

Nigerian legal information is spread across lengthy documents, making it difficult for people to find the specific provisions relevant to their questions. Understanding legal language takes additional effort.

Barrister addresses this by combining BM25 keyword search with semantic search to retrieve relevant legal passages and generate clear, context-aware answers with provision-level citations, allowing users to trace answers back to their sources.


## How it works

```
User submits a legal question
             ↓
BM25 keyword search and semantic search retrieve relevant passages
             ↓
Reciprocal Rank Fusion (RRF) combines and ranks the results
             ↓
Top-ranked legal passages are assembled into context
             ↓
The LLM generates an answer grounded in the retrieved context
             ↓
Answer is returned with provision-level citations to the legal texts
```

## System Architecture

Barrister consists of three layers: a **frontend** for chat, a **backend API** for request handling, and a **RAG pipeline** for retrieving legal passages and generating answers. The pipeline has two stages: offline document indexing and online retrieval and generation.

![Barrister system architecture](./assets/Barrister%20Hybrid%20Legal%20RAG%20Architecture.png)


### Frontend

The Streamlit interface provides a sleek, chat-style experience: it sends the user's question and recent conversation history to FastAPI's `/chat` endpoint and displays the returned answer.

### Backend

The FastAPI service validates incoming requests and passes the question and conversation history to a cached RAG pipeline. The pipeline is loaded once at startup because initializing the retrieval components and models is expensive; reusing it avoids repeating that work for every request and helps keep subsequent responses faster. FastAPI returns the generated answer as JSON.

### RAG Pipeline

#### Offline Indexing

The legal corpus is split into 1,000-token chunks with 200-token overlap, preserving source metadata. The chunks are saved to `data/chunks/chunks.jsonl`, embedded using `BAAI/bge-small-en-v1.5`, and stored with their text and metadata in a persistent Chroma database. The chunk file also supplies documents for BM25 retrieval.

#### Online Retrieval and Generation

The pipeline uses two parallel retrieval paths:

- **Keyword retrieval:** An LLM rewrites the question into keyword-focused alternatives. spaCy lowercases and tokenizes the queries, and BM25 retrieves relevant passages from the chunk documents.
- **Semantic retrieval:** An LLM expands the question into alternative phrasings. The embedding model converts the queries into vectors, and Chroma retrieves semantically similar passages.

Both paths also retrieve results for the original question. An equally weighted `EnsembleRetriever` combines the results using Reciprocal Rank Fusion (RRF), producing a unified ranking without a separate reranking model.

The pipeline selects the top 10 passages and formats their text and source metadata as context. The OpenRouter chat model receives this context alongside the original question and recent conversation history to generate an answer with provision-level citations. FastAPI returns the answer to Streamlit.

![Barrister system architecture](./assets/Barrister%20Hybrid%20Legal%20RAG%20Architecture.png)
