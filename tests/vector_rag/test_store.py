from vector_rag.config import VectorStoreConfig
from vector_rag.store import add_chunks, get_table, search


def _config(tmp_path) -> VectorStoreConfig:
    return VectorStoreConfig(db_uri=str(tmp_path / "lancedb"), table_name="chunks")


def test_get_table_creates_table_when_missing(tmp_path):
    table = get_table(_config(tmp_path), embedding_dim=2)

    assert table.count_rows() == 0


def test_get_table_reopens_existing_table_without_error(tmp_path):
    config = _config(tmp_path)
    get_table(config, embedding_dim=2)

    table = get_table(config, embedding_dim=2)

    assert table.count_rows() == 0


def test_add_chunks_then_search_returns_nearest_neighbor_first(tmp_path):
    table = get_table(_config(tmp_path), embedding_dim=2)
    add_chunks(
        table,
        [
            {"id": "a", "text": "chunk a", "vector": [1.0, 0.0]},
            {"id": "b", "text": "chunk b", "vector": [0.0, 1.0]},
        ],
    )

    results = search(table, query_vector=[0.9, 0.1], top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "a"
    assert results[0]["text"] == "chunk a"
