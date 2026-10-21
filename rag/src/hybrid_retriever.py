from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import MultiQueryRetriever, EnsembleRetriever

from .generator import llm
from .utils import load_documents, custom_preprocess
from .config import EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR
from ..prompts.rag_prompts import QUERY_EXPANSION_PROMPT, QUERY_REWRITING_PROMPT


def load_vector_store():
    """Load the vector store from the output directory."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma(
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=embeddings
    )
    return vector_store


def retriever(top_k: int = 5):
    """Retrieve documents from the vector store based on a query."""

    # Multi-query-retriever for Key-word search retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(
        documents=load_documents(),
        preprocess_func=custom_preprocess
    )
    bm25_retriever.k = top_k

    multi_query_retriever_bm25 = MultiQueryRetriever.from_llm(
        retriever=bm25_retriever,
        llm=llm,
        prompt=QUERY_REWRITING_PROMPT,
        include_original=True,
    )

    # Multi-query-retriever for Vector_store retriever (Semantic Search)
    vector_store = load_vector_store()
    vector_search_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    multi_query_retriever_vector_search = MultiQueryRetriever.from_llm(
        retriever=vector_search_retriever,
        llm=llm,
        prompt=QUERY_EXPANSION_PROMPT,
        include_original=True,
    )

    # Ensemble retriever to combine and rerank retriever results
    ensemble_retiever = EnsembleRetriever(
        retrievers=[multi_query_retriever_bm25, multi_query_retriever_vector_search],
        weights=[0.5, 0.5]
    )

    return ensemble_retiever


retriever = retriever()