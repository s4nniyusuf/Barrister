"""
This module builds the RAG pipeline to
- retrieve relevant documents
- format the question, chat history, and context
- generate an answer with the language model
"""

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from .hybrid_retriever import retriever
from .utils import select_top_docs, format_docs
from .llm import llm
from ..prompts.rag_prompts import LEGAL_QA_PROMPT


def format_history(messages: list[dict], limit: int = 12) -> str:
    """Format recent chat turns for the answer prompt."""
    recent_messages = messages[-limit:]
    return "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in recent_messages
        if message.get("content")
    )


rag_pipeline = (
    {
        "question": RunnableLambda(lambda values: values["question"]),
        "history": RunnableLambda(lambda values: values.get("history", []))
        | RunnableLambda(format_history),
        "context": RunnableLambda(lambda values: values["question"])
        | retriever
        | RunnableLambda(select_top_docs)
        | RunnableLambda(format_docs)
    } 
    | LEGAL_QA_PROMPT 
    | llm
    | StrOutputParser()
)
