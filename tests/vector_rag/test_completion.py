from types import SimpleNamespace

import vector_rag.completion as completion_module
from vector_rag.completion import complete
from vector_rag.config import ModelConfig


def test_complete_calls_litellm_with_provider_prefixed_model(monkeypatch):
    captured_kwargs = {}

    def fake_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="the answer"))]
        )

    monkeypatch.setattr(completion_module.litellm, "completion", fake_completion)

    config = ModelConfig(
        model_provider="openai",
        model="gpt-4.1",
        auth_method="api_key",
        api_key="test-key",
    )
    messages = [{"role": "user", "content": "hi"}]

    result = complete(messages, config)

    assert captured_kwargs["model"] == "openai/gpt-4.1"
    assert captured_kwargs["messages"] == messages
    assert captured_kwargs["api_key"] == "test-key"
    assert result == "the answer"
