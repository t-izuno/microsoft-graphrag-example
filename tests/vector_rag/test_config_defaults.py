from pathlib import Path

from vector_rag.config import load_config


def test_load_config_default_path_resolves_against_repo_layout():
    config = load_config()

    repo_root = Path(__file__).resolve().parents[2]

    assert config.input.base_dir == str((repo_root / "input").resolve())
    assert config.vector_store.db_uri == str(
        (repo_root / "vector_rag" / "output" / "lancedb").resolve()
    )
    assert config.completion_model.model_provider == "openai"
    assert config.retrieval.top_k == 10
