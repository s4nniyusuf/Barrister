import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHUNKS_DIR, EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR


def load_chunks():
    """Load the chunks from the out put directory."""
    chunks_path = Path(CHUNKS_DIR) / "chunks.jsonl"

    chunks = []

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def convert_chunks_to_documents(chunks: list[dict]):
    """Convert the loaded chunks to LangChain documents."""
    docs = [
        Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        )
        for chunk in chunks
    ]
    return docs


def create_vector_store():
    """Create a vector store from the loaded chunks."""
    chunks = load_chunks()
    docs = convert_chunks_to_documents(chunks)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR)
    )
    return vector_store


if __name__ == "__main__":
    create_vector_store()