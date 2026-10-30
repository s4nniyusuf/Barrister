"""Configuration for RAG data paths, chunking, and model settings."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"


# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# Embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


# LLM model name
# LLM_MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b:free"
# LLM_MODEL_NAME = "qwen/qwen3.8-27b"
LLM_MODEL_NAME = "openai/gpt-4o"  # For evaluation