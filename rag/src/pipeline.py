"""Build the RAG pipeline for retrieving context and generating answers."""

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
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


# def print_retrieved_context(context: str) -> str:
#     """Print the context sent to the answer model, without changing it."""
#     print(f"\nRetrieved context:\n{context}\n")
#     return context


rag_pipeline_with_context = (
    RunnableParallel({
        "question": RunnableLambda(lambda values: values["question"]),
        "history": RunnableLambda(lambda values: values.get("history", []))
        | RunnableLambda(format_history),
        "context": RunnableLambda(lambda values: values["question"])
        | retriever
        | RunnableLambda(select_top_docs)
        | RunnableLambda(format_docs)
        # | RunnableLambda(print_retrieved_context)
    })
    | RunnablePassthrough.assign(answer=LEGAL_QA_PROMPT | llm | StrOutputParser())
)

# Keep the application's existing string response while exposing the exact
# generation context to evaluation without a second retrieval.
rag_pipeline = rag_pipeline_with_context | RunnableLambda(lambda result: result["answer"])
