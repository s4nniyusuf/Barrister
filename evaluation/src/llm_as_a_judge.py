"""Standalone LangChain pipeline for judging a single RAG response."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from ..prompts.eval_prompts import LLM_JUDGE_PROMPT
from rag.src.llm import llm


llm_as_a_judge = (
    {
        "question": RunnableLambda(lambda values: values["question"]),
        "retrieved_context": RunnableLambda(
            lambda values: values["retrieved_context"]
        ),
        "rag_answer": RunnableLambda(lambda values: values["rag_answer"]),
        "reference_answer": RunnableLambda(
            lambda values: values["reference_answer"]
        ),
    }
    | LLM_JUDGE_PROMPT
    | llm
    | StrOutputParser()
)