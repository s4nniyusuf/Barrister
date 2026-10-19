from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .retriever import retriever
from .utils import format_docs
from .generator import llm
from ..prompts.rag_prompts import LEGAL_QA_PROMPT


def print_retrieved_docs(docs):
    """Print retrieved documents for inspection, then pass them through unchanged."""
    print("\nRetrieved documents:")
    for index, doc in enumerate(docs, start=1):
        print(f"\n--- Document {index} ---")
        print(f"Source: {doc.metadata}")
        print(doc.page_content)
        print()
    
    print("\n")
    return docs


rag_pipeline = (
    {
        "question": RunnablePassthrough(),
        "context": retriever | RunnableLambda(print_retrieved_docs) | RunnableLambda(format_docs)
    } 
    | LEGAL_QA_PROMPT 
    | llm
    | StrOutputParser()
)

print(rag_pipeline.invoke("What are the punishment attached to vandalism")) 