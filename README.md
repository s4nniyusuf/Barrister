# Barrister

Barrister is a citation-aware retrieval-augmented generation (RAG) system for Nigerian legal information. Its knowledge base includes the Constitution of Nigeria, the Criminal Code Act, the Lagos State Tenancy Law, and the Electoral Act. It combines BM25-based keyword search with semantic search to retrieve relevant legal passages, which are then used to generate clear, context-aware answers with provision-level citations, enabling users to trace responses back to the underlying legal texts.

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-interface-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Chroma](https://img.shields.io/badge/Chroma-vector%20store-5B4BDB)](https://www.trychroma.com/)
[![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-embeddings-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)



## RAG Pipeline

### Document Chunking

- Chunking is implemented independently in [`rag/src/chunk.py`](rag/src/chunk.py). It splits text into 1,000-token passages with 200-token overlap.
- Metadata is built from fields available in each corpus record. Null fields are omitted to avoid vector store metadata errors.

### Persistent Vector Store

- Chunks are embedded with `BAAI/bge-small-en-v1.5` and stored in Chroma in the project-root `vector_store/` directory. Persistence lets the application reuse the index across runs.

### Hybrid Retrieval

- **BM25 keyword search** finds passages with matching terms; **vector search** finds passages with similar meaning.
- Both retrievers use `MultiQueryRetriever` to search alternative versions of the question as well as the original, broadening retrieval coverage.

### BM25 Text Preprocessing

- `custom_preprocess` in [`rag/src/utils.py`](rag/src/utils.py) tokenizes and lowercases text to improve keyword matching while preserving stop words and avoiding stemming, helping retain important legal context and terminology.

### Query Transformation

- **Keyword-aware rewriting** creates two concise, term-focused queries for BM25.
- **Query expansion** creates three alternative phrasings for semantic search.

### Ensemble Retrieval

- An equally weighted `EnsembleRetriever` combines BM25 and vector search rankings using **Reciprocal Rank Fusion (RRF)**. RRF favors passages that rank well across both methods.

### Reranking

- No separate reranking model is used, reducing additional model calls and latency. RRF performs rank fusion and ordering; a dedicated reranker could be considered in future work.

### RAG Architecture
![Barrister architecture diagram](./assets/Barrister%20Architecture.png)
