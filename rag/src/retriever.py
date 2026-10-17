from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    return retriever


retriever = retriever()