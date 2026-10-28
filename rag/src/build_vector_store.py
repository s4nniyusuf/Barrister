"""
This module
- builds and persists a vector store from the saved text chunks.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .config import EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR
from .utils import load_documents


def create_vector_store():
    """Create a vector store from the loaded chunks."""
    docs = load_documents()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR)
    )
    return vector_store


if __name__ == "__main__":
    create_vector_store()