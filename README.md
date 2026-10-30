# Barrister

Barrister is a citation-aware retrieval-augmented generation (RAG) system for Nigerian legal information, built on a curated corpus of Nigerian legal texts.

## Dataset

I curated the Nigerian legal corpus used by this project. It currently includes the 1999 Constitution, the Electoral Act, the Criminal Code Act, and the Lagos Tenancy Law. The corpus is available in the [Nigerian Legal Corpus repository](https://github.com/s4nniyusuf/nigerian-legal-corpus), with original source files and processed JSONL records for retrieval and NLP.

The processed dataset is licensed under CC BY 4.0. See the corpus repository's README and `LICENSE.md` for details.


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

![Barrister system architecture](./assets/Barrister%20Architecture.png)


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


## Evaluation (llm-as-a-judge)
I generated 10 synthetic question-and-answer pairs from the legal corpus, each grounded in a specific legal provision, and split them into tuning and test sets. 

In the five-sample tuning evaluation, the LLM judge found that retrieved contexts generally contained the information needed to answer the questions, and that answers were faithful, relevant, correct, and sufficiently detailed. 

The judge model was GPT-4o.

The average scores across the five tuning samples were **4.8/5 for faithfulness**, **4.2/5 for precision**, **5.0/5 for recall**, **4.9/5 for answer relevance**, and **4.9/5 for answer correctness**.

The per-sample judge results are saved in [`evaluation/data/evaluation_results.json`](evaluation/data/evaluation_results.json).

## Tech Stack

| Area | Technology | Role |
|---|---|---|
| Language | Python 3.12+ | Application and RAG implementation |
| Package management | uv | Dependency and environment management |
| Frontend | Streamlit, Requests, markdown-it-py | Chat interface, API requests, and Markdown rendering |
| Backend API | FastAPI, Pydantic, Uvicorn | HTTP endpoints, request validation, and ASGI serving |
| RAG orchestration | LangChain and LangChain Classic | Prompt templates, query transformation, retrieval, and answer generation |
| Keyword retrieval | BM25 (`rank-bm25`) | Finds passages with matching legal terms |
| Semantic retrieval | Hugging Face Sentence Transformers, `BAAI/bge-small-en-v1.5` | Embeds chunks and queries for vector search |
| Vector store | Chroma | Persists embeddings, chunk text, and metadata |
| Language model | GPT-4o via OpenRouter | Query rewriting, query expansion, answer generation, and evaluation |
| Chunking | LangChain Text Splitters, `tiktoken` | Splits source text into 1,000-token chunks with 200-token overlap |
| Text preprocessing | spaCy (`en_core_web_sm`) | Tokenizes and lowercases text for BM25 |
| Evaluation | GPT-4o LLM-as-a-judge | Scores answer faithfulness, precision, recall, relevance, and correctness |


## Project Structure

```text
Barrister/
├── assets/
│   └── Barrister Architecture.png  # System architecture diagram
├── backend/
│   └── app/
│       ├── main.py                 # FastAPI application and chat endpoint
│       └── config.py               # Backend settings
├── data/
│   ├── corpus/                     # Nigerian legal source documents
│   │   ├── criminal_code_act.jsonl
│   │   ├── electoral_act_2026.jsonl
│   │   ├── nigeria-constitution-1999-amended-2023.jsonl
│   │   └── tenancy_law_2011.jsonl
│   └── chunks/
│       └── chunks.jsonl            # Chunked text and source metadata
├── evaluation/
│   ├── data/
│   │   ├── tuning.jsonl            # Tuning questions and reference answers
│   │   ├── test.jsonl              # Held-out questions and reference answers
│   │   └── evaluation_results.json # Per-sample metrics and average scores
│   ├── prompts/
│   │   └── eval_prompts.py         # QA generation and judge prompts
│   └── src/
│       ├── generate_qa_dataset.py  # Generate and split synthetic QA samples
│       ├── evaluate.py             # Run and summarize tuning-set evaluation
│       └── llm_as_a_judge.py       # LLM-as-a-judge pipeline
├── frontend/
│   ├── .streamlit/config.toml      # Streamlit configuration
│   └── app.py                      # Streamlit chat interface
├── rag/
│   ├── prompts/
│   │   └── rag_prompts.py          # Query and legal-answer prompts
│   └── src/
│       ├── build_vector_store.py   # Build the Chroma vector index
│       ├── chunk.py                # Load and chunk legal documents
│       ├── config.py               # Data paths, chunking, and model settings
│       ├── hybrid_retriever.py     # BM25 and vector retrieval with RRF
│       ├── llm.py                  # OpenRouter chat model setup
│       ├── pipeline.py             # Retrieval-to-answer pipeline
│       └── utils.py                # Document loading and formatting
├── .env.example                    # Environment variable template
├── .gitignore
├── .python-version
├── LICENSE
├── pyproject.toml                  # Project metadata and dependencies
├── README.md
└── uv.lock                         # Locked dependency versions
```


## Setup and Running

### Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- An OpenRouter API key

From the project root, install the project dependencies and spaCy's English model:

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` to your OpenRouter API key.

### Offline indexing

Prepare the chunk file from the JSONL legal corpus, then build the persistent Chroma vector store:

```bash
uv run python -m rag.src.chunk
uv run python -m rag.src.build_vector_store
```

This creates or updates `data/chunks/chunks.jsonl` and creates the `vector_store/` directory. The first embedding run downloads the Hugging Face model and may take a while.

### Online application

Complete offline indexing first, then run the API and frontend in separate terminals from the project root.

**Terminal 1 — FastAPI backend:**

```bash
uv run fastapi run backend/app/main.py
```

**Terminal 2 — Streamlit frontend:**

```bash
uv run streamlit run frontend/app.py
```

Open the Streamlit URL shown in the terminal (normally `http://localhost:8501`). The API health endpoint is at `http://localhost:8000/health`, and its interactive documentation is at `http://localhost:8000/docs`.

## Feedback
Open an issue, submit a pull request, or reach out to me on X [s4nni_yusuf](https://x.com/s4nni_yusuf)

If you find Barrister useful, please give it a star, it helps more people find the project.

## License
MIT [See License](./LICENSE)
