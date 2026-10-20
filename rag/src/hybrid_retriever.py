from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from .utils import load_documents, custom_preprocess
from .config import EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR


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

    # Key-word search retriever (BM25 Retriever)
    bm25_retriever = BM25Retriever.from_documents(
        documents=load_documents(),
        preprocess_func=custom_preprocess
    )
    bm25_retriever.k = top_k

    # Vector_store retriever (Semantic Search)
    vector_store = load_vector_store()
    vector_search_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    # Ensemble retriever to combine and rerank retriever results
    ensemble_retiever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_search_retriever],
        weights=[0.5, 0.5]
    )

    return ensemble_retiever


retriever = retriever()