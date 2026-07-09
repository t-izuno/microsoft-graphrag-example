"""Call the vector_rag pipeline's query path."""

from vector_rag.answer import answer_query
from vector_rag.config import load_config as load_vector_rag_config


def vector_rag_search(query: str) -> str:
    """Answer a query using the vector_rag pipeline."""
    config = load_vector_rag_config()
    return answer_query(query, config)
