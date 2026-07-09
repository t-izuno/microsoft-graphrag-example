from types import SimpleNamespace

import vector_rag.embedding as embedding_module
from vector_rag.config import ModelConfig
from vector_rag.embedding import embed_texts


def test_embed_texts_calls_litellm_with_provider_prefixed_model(monkeypatch):
    captured_kwargs = {}

    def fake_embedding(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            data=[{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]
        )

    monkeypatch.setattr(embedding_module.litellm, "embedding", fake_embedding)

    config = ModelConfig(
        model_provider="openai",
        model="text-embedding-3-large",
        auth_method="api_key",
        api_key="test-key",
    )

    result = embed_texts(["hello", "world"], config)

    assert captured_kwargs["model"] == "openai/text-embedding-3-large"
    assert captured_kwargs["input"] == ["hello", "world"]
    assert captured_kwargs["api_key"] == "test-key"
    assert result == [[0.1, 0.2], [0.3, 0.4]]
