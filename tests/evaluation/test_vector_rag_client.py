import evaluation.vector_rag_client as client_module
from evaluation.vector_rag_client import vector_rag_search


def test_vector_rag_search_loads_config_and_calls_answer_query(monkeypatch):
    monkeypatch.setattr(client_module, "load_vector_rag_config", lambda: "fake-config")

    captured = {}

    def fake_answer_query(query, config):
        captured["query"] = query
        captured["config"] = config
        return "vector rag answer"

    monkeypatch.setattr(client_module, "answer_query", fake_answer_query)

    result = vector_rag_search("what is x?")

    assert result == "vector rag answer"
    assert captured["query"] == "what is x?"
    assert captured["config"] == "fake-config"
