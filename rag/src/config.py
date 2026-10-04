from pathlib import Path

# Root directories
RAG_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = RAG_ROOT.parent
DATA_DIR = RAG_ROOT / "data"

# Corpus directories
CORPUS_DIR = DATA_DIR / "corpus"
CHUNKS_DIR = DATA_DIR / "chunks"

# Vector database directory
VECTOR_STORE_DIR = PROJECT_ROOT / "vectorstore"

# Text splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding model parameters
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"