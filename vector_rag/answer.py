"""Retrieve relevant chunks and generate an answer for a query."""

from pathlib import Path

from vector_rag.completion import complete
from vector_rag.config import VectorRagConfig
from vector_rag.embedding import embed_texts
from vector_rag.store import get_table, search

_PROMPT_PATH = Path(__file__).parent / "prompts" / "query_system_prompt.txt"


def answer_query(query: str, config: VectorRagConfig) -> str:
    """Answer a query using top-k similar chunks as context."""
    query_vector = embed_texts([query], config.embedding_model)[0]
    table = get_table(config.vector_store, embedding_dim=len(query_vector))
    chunks = search(table, query_vector, top_k=config.retrieval.top_k)

    context = "\n\n".join(chunk["text"] for chunk in chunks)
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"コンテキスト:\n{context}\n\n質問: {query}"},
    ]
    return complete(messages, config.completion_model)
