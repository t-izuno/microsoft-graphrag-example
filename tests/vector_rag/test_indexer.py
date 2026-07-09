import vector_rag.indexer as indexer_module
from vector_rag.config import (
    ChunkingConfig,
    InputConfig,
    ModelConfig,
    RetrievalConfig,
    VectorRagConfig,
    VectorStoreConfig,
)
from vector_rag.indexer import run_index
from vector_rag.store import get_table


def _config(tmp_path) -> VectorRagConfig:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "doc.txt").write_text("abcdefghij", encoding="utf-8")

    return VectorRagConfig(
        input=InputConfig(base_dir=str(input_dir)),
        completion_model=ModelConfig(model_provider="openai", model="gpt-4.1"),
        embedding_model=ModelConfig(
            model_provider="openai", model="text-embedding-3-large"
        ),
        chunking=ChunkingConfig(size=4, overlap=1, encoding_model="o200k_base"),
        vector_store=VectorStoreConfig(
            db_uri=str(tmp_path / "lancedb"), table_name="chunks"
        ),
        retrieval=RetrievalConfig(top_k=10),
    )


def test_run_index_chunks_embeds_and_stores_each_document(monkeypatch, tmp_path):
    config = _config(tmp_path)

    def fake_split_text(text, chunking_config):
        return [text[i : i + 4] for i in range(0, len(text), 4)]

    def fake_embed_texts(texts, model_config):
        return [[float(len(t)), 0.0] for t in texts]

    monkeypatch.setattr(indexer_module, "split_text", fake_split_text)
    monkeypatch.setattr(indexer_module, "embed_texts", fake_embed_texts)

    indexed_count = run_index(config)

    assert indexed_count == 3

    table = get_table(config.vector_store, embedding_dim=2)
    rows = table.search([4.0, 0.0]).limit(10).to_list()
    texts = sorted(r["text"] for r in rows)
    assert texts == ["abcd", "efgh", "ij"]


def test_run_index_returns_zero_when_no_input_files(monkeypatch, tmp_path):
    config = _config(tmp_path)
    for f in (tmp_path / "input").glob("*.txt"):
        f.unlink()

    monkeypatch.setattr(indexer_module, "split_text", lambda text, cfg: [])
    monkeypatch.setattr(indexer_module, "embed_texts", lambda texts, cfg: [])

    assert run_index(config) == 0
