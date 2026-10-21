"""
This module contains functions/helpers that 
- Load saved text chunks from a jsonl file, 
- converts them into LangChain documents 
- prepares them for retrieval.
- preprocess text in bm25 queries and documents
- formats documents as readable source-and-content text for llm inference
"""


import json
from pathlib import Path

import spacy
from langchain_core.documents import Document

from .config import CHUNKS_DIR


nlp = spacy.load("en_core_web_sm")


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


def custom_preprocess(text: str) ->list[str]:
    """Tokenize text with spaCy and return each token in lowercase form."""
    doc = nlp(text)
    return [token.lower_ for token in doc]


def select_top_docs(docs, limit: int = 5):
    """Select the top-ranked documents to send to the generator."""
    return docs[:limit]


def format_docs(docs):
    return "\n\n".join(
        [
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in docs
        ]
    )
