"""Build the RAG pipeline for retrieving context and generating answers."""

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from .hybrid_retriever import retriever
from .utils import select_top_docs, format_docs
from .llm import llm
from ..prompts.rag_prompts import LEGAL_QA_PROMPT, CONTEXTUALIZE_QUESTION_PROMPT


def format_history(messages: list[dict], limit: int = 12) -> str:
    """Format recent chat turns for the answer prompt."""
    recent_messages = messages[-limit:]
    return "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in recent_messages
        if message.get("content")
    )


contextualize_question = CONTEXTUALIZE_QUESTION_PROMPT | llm | StrOutputParser()


def retrieval_question(values: dict) -> str:
    """Resolve follow-up references for retrieval, skipping rewriting without history."""
    question = values["question"]
    history = format_history(values.get("history", []))
    if not history:
        return question
    standalone = contextualize_question.invoke({"question": question, "history": history})
    return standalone.strip() or question


# def print_retrieved_context(context: str) -> str:
#     """Print the context sent to the answer model, without changing it."""
#     print(f"\nRetrieved context:\n{context}\n")
#     return context


rag_pipeline = (
    {
        "question": RunnableLambda(lambda values: values["question"]),
        "history": RunnableLambda(lambda values: values.get("history", []))
        | RunnableLambda(format_history),
        "context": RunnableLambda(retrieval_question)
        | retriever
        | RunnableLambda(select_top_docs)
        | RunnableLambda(format_docs)
        # | RunnableLambda(print_retrieved_context)
    } 
    | LEGAL_QA_PROMPT 
    | llm
    | StrOutputParser()
)
