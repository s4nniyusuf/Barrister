import json
from pathlib import Path

from langchain_core.documents import Document

from .config import CHUNKS_DIR


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


def load_documents():
    """Load chunks and convert them to LangChain documents."""
    chunks = load_chunks()
    docs = convert_chunks_to_documents(chunks)
    return docs


def format_docs(docs):
    return "\n\n".join(
        [
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in docs
        ]
    )