from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.retriever import retriever
from src.utils import format_docs
from src.generator import llm
from prompts.rag_prompts import LEGAL_QA_PROMPT


rag_pipeline = (
    {
        "question": RunnablePassthrough(),
        "context": retriever | RunnableLambda(format_docs)
    } 
    | LEGAL_QA_PROMPT 
    | llm
    | StrOutputParser()
)

# print(rag_pipeline.invoke("What punishment applies to those who violate nigerian law"))