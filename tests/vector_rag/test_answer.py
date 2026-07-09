import vector_rag.answer as answer_module
from vector_rag.answer import answer_query
from vector_rag.config import (
    ChunkingConfig,
    InputConfig,
    ModelConfig,
    RetrievalConfig,
    VectorRagConfig,
    VectorStoreConfig,
)


def _config(tmp_path) -> VectorRagConfig:
    return VectorRagConfig(
        input=InputConfig(base_dir=str(tmp_path / "input")),
        completion_model=ModelConfig(model_provider="openai", model="gpt-4.1"),
        embedding_model=ModelConfig(
            model_provider="openai", model="text-embedding-3-large"
        ),
        chunking=ChunkingConfig(size=1200, overlap=100, encoding_model="o200k_base"),
        vector_store=VectorStoreConfig(
            db_uri=str(tmp_path / "lancedb"), table_name="chunks"
        ),
        retrieval=RetrievalConfig(top_k=2),
    )


def test_answer_query_builds_context_from_retrieved_chunks(monkeypatch, tmp_path):
    config = _config(tmp_path)

    monkeypatch.setattr(answer_module, "embed_texts", lambda texts, cfg: [[1.0, 0.0]])
    monkeypatch.setattr(
        answer_module, "get_table", lambda cfg, embedding_dim: "fake-table"
    )
    monkeypatch.setattr(
        answer_module,
        "search",
        lambda table, query_vector, top_k: [
            {"id": "a-0", "text": "chunk one"},
            {"id": "a-1", "text": "chunk two"},
        ],
    )

    captured = {}

    def fake_complete(messages, model_config):
        captured["messages"] = messages
        return "final answer"

    monkeypatch.setattr(answer_module, "complete", fake_complete)

    result = answer_query("what is x?", config)

    assert result == "final answer"
    assert captured["messages"][0]["role"] == "system"
    user_message = captured["messages"][1]["content"]
    assert "chunk one" in user_message
    assert "chunk two" in user_message
    assert "what is x?" in user_message
