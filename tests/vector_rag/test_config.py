from vector_rag.config import load_config


def test_load_config_resolves_paths_and_substitutes_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_VECTOR_RAG_API_KEY", "secret-value")
    settings_dir = tmp_path / "vector_rag"
    settings_dir.mkdir()
    settings_path = settings_dir / "settings.yaml"
    settings_path.write_text(
        """
input:
  base_dir: ../input

completion_model:
  model_provider: openai
  model: gpt-4.1
  auth_method: api_key
  api_key: ${TEST_VECTOR_RAG_API_KEY}

embedding_model:
  model_provider: openai
  model: text-embedding-3-large
  auth_method: api_key
  api_key: ${TEST_VECTOR_RAG_API_KEY}

chunking:
  size: 1200
  overlap: 100
  encoding_model: o200k_base

vector_store:
  db_uri: output/lancedb
  table_name: chunks

retrieval:
  top_k: 10
""",
        encoding="utf-8",
    )

    config = load_config(settings_path)

    assert config.input.base_dir == str((tmp_path / "input").resolve())
    assert config.completion_model.model_provider == "openai"
    assert config.completion_model.api_key == "secret-value"
    assert config.embedding_model.api_key == "secret-value"
    assert config.vector_store.db_uri == str((settings_dir / "output" / "lancedb").resolve())
    assert config.chunking.size == 1200
    assert config.chunking.overlap == 100
    assert config.retrieval.top_k == 10
