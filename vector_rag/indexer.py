"""Read input documents, chunk, embed, and store them in LanceDB."""

from pathlib import Path

from vector_rag.chunking import split_text
from vector_rag.config import VectorRagConfig
from vector_rag.embedding import embed_texts
from vector_rag.store import add_chunks, get_table


def run_index(config: VectorRagConfig) -> int:
    """Index every .txt file under config.input.base_dir. Returns chunk count."""
    input_dir = Path(config.input.base_dir)

    chunks: list[dict] = []
    for file_path in sorted(input_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        for i, chunk_text in enumerate(split_text(text, config.chunking)):
            chunks.append({"id": f"{file_path.stem}-{i}", "text": chunk_text})

    if not chunks:
        return 0

    vectors = embed_texts([chunk["text"] for chunk in chunks], config.embedding_model)
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk["vector"] = vector

    table = get_table(config.vector_store, embedding_dim=len(vectors[0]))
    add_chunks(table, chunks)
    return len(chunks)
