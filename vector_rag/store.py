"""LanceDB-backed chunk storage and similarity search."""

import lancedb
import pyarrow as pa

from vector_rag.config import VectorStoreConfig


def get_table(config: VectorStoreConfig, embedding_dim: int):
    """Open the chunk table, creating it if it does not already exist."""
    db = lancedb.connect(config.db_uri)
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
        ]
    )
    return db.create_table(config.table_name, schema=schema, exist_ok=True)


def add_chunks(table, chunks: list[dict]) -> None:
    """Insert chunk records ({id, text, vector}) into the table."""
    table.add(chunks)


def search(table, query_vector: list[float], top_k: int) -> list[dict]:
    """Return the top_k chunks nearest to the query vector."""
    return table.search(query_vector).limit(top_k).to_list()
