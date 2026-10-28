"""
This module provides a command-line tool to test how the language model
- rewrites questions for BM25 keyword search
- expands questions for vector search
"""

from langchain_core.output_parsers import StrOutputParser

from .llm import llm
from ..prompts.rag_prompts import QUERY_REWRITING_PROMPT, QUERY_EXPANSION_PROMPT


def test_query_prompts(question: str) -> None:
    """Run the first two prompts with the shared LLM and print their outputs."""
    parser = StrOutputParser()

    print("\n=== Query rewriting prompt (BM25) ===")
    rewrite_chain = QUERY_REWRITING_PROMPT | llm | parser
    print(rewrite_chain.invoke({"question": question}))

    print("\n=== Query expansion prompt (vector search) ===")
    expansion_chain = QUERY_EXPANSION_PROMPT | llm | parser
    print(expansion_chain.invoke({"question": question}))


if __name__ == "__main__":
    print("Enter a question to test the query prompts. Type 'quit' or 'exit' to stop.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"quit", "exit"}:
            print("Done.")
            break
        if not question:
            print("Please enter a question.")
            continue
        test_query_prompts(question)
