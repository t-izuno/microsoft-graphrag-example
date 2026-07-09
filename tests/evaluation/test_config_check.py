from evaluation.config_check import check_provider_consistency
from vector_rag.config import (
    ChunkingConfig,
    InputConfig,
    ModelConfig,
    RetrievalConfig,
    VectorRagConfig,
    VectorStoreConfig,
)


def _vector_rag_config(
    completion_provider="openai",
    completion_model="gpt-4.1",
    embedding_provider="openai",
    embedding_model="text-embedding-3-large",
) -> VectorRagConfig:
    return VectorRagConfig(
        input=InputConfig(base_dir="/tmp/input"),
        completion_model=ModelConfig(
            model_provider=completion_provider, model=completion_model
        ),
        embedding_model=ModelConfig(
            model_provider=embedding_provider, model=embedding_model
        ),
        chunking=ChunkingConfig(size=1200, overlap=100, encoding_model="o200k_base"),
        vector_store=VectorStoreConfig(db_uri="/tmp/db", table_name="chunks"),
        retrieval=RetrievalConfig(top_k=10),
    )


def _write_graphrag_settings(
    tmp_path,
    completion_provider="openai",
    completion_model="gpt-4.1",
    embedding_provider="openai",
    embedding_model="text-embedding-3-large",
):
    path = tmp_path / "settings.yaml"
    path.write_text(
        f"""
completion_models:
  default_completion_model:
    model_provider: {completion_provider}
    model: {completion_model}

embedding_models:
  default_embedding_model:
    model_provider: {embedding_provider}
    model: {embedding_model}
""",
        encoding="utf-8",
    )
    return path


def test_check_provider_consistency_returns_empty_when_matching(tmp_path):
    graphrag_path = _write_graphrag_settings(tmp_path)
    vector_rag_config = _vector_rag_config()

    warnings = check_provider_consistency(
        vector_rag_config, graphrag_settings_path=graphrag_path
    )

    assert warnings == []


def test_check_provider_consistency_flags_mismatched_completion_model(tmp_path):
    graphrag_path = _write_graphrag_settings(tmp_path, completion_model="gpt-4o")
    vector_rag_config = _vector_rag_config(completion_model="gpt-4.1")

    warnings = check_provider_consistency(
        vector_rag_config, graphrag_settings_path=graphrag_path
    )

    assert len(warnings) == 1
    assert "completion_model" in warnings[0]


def test_check_provider_consistency_flags_mismatched_embedding_provider(tmp_path):
    graphrag_path = _write_graphrag_settings(tmp_path, embedding_provider="azure")
    vector_rag_config = _vector_rag_config(embedding_provider="openai")

    warnings = check_provider_consistency(
        vector_rag_config, graphrag_settings_path=graphrag_path
    )

    assert len(warnings) == 1
    assert "embedding_model" in warnings[0]
