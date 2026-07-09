import json

import yaml

import evaluation.generate_qa as generate_qa_module
from evaluation.generate_qa import generate_qa_dataset
from vector_rag.config import (
    ChunkingConfig,
    InputConfig,
    ModelConfig,
    RetrievalConfig,
    VectorRagConfig,
    VectorStoreConfig,
)


def _vector_rag_config(tmp_path) -> VectorRagConfig:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "doc.txt").write_text("dummy text", encoding="utf-8")
    return VectorRagConfig(
        input=InputConfig(base_dir=str(input_dir)),
        completion_model=ModelConfig(model_provider="openai", model="gpt-4.1"),
        embedding_model=ModelConfig(
            model_provider="openai", model="text-embedding-3-large"
        ),
        chunking=ChunkingConfig(size=1200, overlap=100, encoding_model="o200k_base"),
        vector_store=VectorStoreConfig(
            db_uri=str(tmp_path / "lancedb"), table_name="chunks"
        ),
        retrieval=RetrievalConfig(top_k=10),
    )


def test_generate_qa_dataset_samples_chunks_and_writes_yaml(monkeypatch, tmp_path):
    vector_rag_config = _vector_rag_config(tmp_path)
    completion_model = ModelConfig(model_provider="openai", model="gpt-4.1")

    fake_chunks = [f"chunk-{i}" for i in range(6)]
    monkeypatch.setattr(
        generate_qa_module, "split_text", lambda text, cfg: fake_chunks
    )
    monkeypatch.setattr(
        generate_qa_module,
        "complete",
        lambda messages, model_config: json.dumps(
            {"question": "Q?", "expected_answer": "A."}
        ),
    )

    output_path = tmp_path / "qa_dataset.yaml"

    entries = generate_qa_dataset(
        target_count=3,
        vector_rag_config=vector_rag_config,
        completion_model=completion_model,
        output_path=output_path,
    )

    assert len(entries) == 3
    for entry in entries:
        assert entry["question"] == "Q?"
        assert entry["expected_answer"] == "A."
        assert entry["reviewed"] is False
        assert entry["source_chunk_id"].startswith("doc-")

    written = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert written == entries


def test_load_graphrag_completion_model_reads_and_substitutes_env(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TEST_GENQA_KEY", "secret")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        """
completion_models:
  default_completion_model:
    model_provider: openai
    model: gpt-4.1
    auth_method: api_key
    api_key: ${TEST_GENQA_KEY}
""",
        encoding="utf-8",
    )

    model = generate_qa_module.load_graphrag_completion_model(settings_path)

    assert model.model_provider == "openai"
    assert model.model == "gpt-4.1"
    assert model.api_key == "secret"
