# Barrister
A Retrieval Augmented Generation system for Nigerian Legal information

This study presents a novel Retrieval Augmented Generation (RAG), a text-based query system, for efficient access to Nigerian legal information. Utilizing the Nigerian Constitution and Criminal Code as its knowledge base, the system employs a pipeline involving semantic segmentation, Sentence Transformer embeddings, and vector database indexing for optimized information retrieval. User queries are refined by a Google Gemini large language model, trained as a Nigerian legal expert, to identify key terms and intent before searching the database for the top ten most relevant document chunks. These chunks, along with the refined query and keywords, are then fed back into Gemini to generate a detailed, referenced answer. The current implementation is evaluated using the precision. Recall, F1Score, perplexity and diversity metrics, and results fall within acceptable benchmarks of mean values (0.65, 0.73, 0.68, 14.42, 0.87) respectively, representing a significant advancement in making complex legal big data accessible.

LangChain and its integrations are used to build the RAG pipeline, connect embedding models, and integrate the vector database.


Nigerian legislation contains important information about people's rights, obligations and institutions, but finding the precise provision relevant to a question can be difficult. Barrister explores how retrieval-augmented generation can make selected Nigerian legal sources easier to search, understand and verify through provision-level citations.


It is a data resource, not a substitute for official legal sources or professional legal advice. Users should verify retrieved provisions against authoritative, current versions of the applicable legislation.


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