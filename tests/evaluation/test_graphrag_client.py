import os
from pathlib import Path

import evaluation.graphrag_client as client_module
from evaluation.graphrag_client import global_search, local_search


class FakeConfig:
    output_storage = object()
    table_provider = object()


class FakeTableProvider:
    async def has(self, name):
        return False


class FakeDataReader:
    def __init__(self, table_provider):
        self.table_provider = table_provider

    async def communities(self):
        return "communities-df"

    async def community_reports(self):
        return "community_reports-df"

    async def text_units(self):
        return "text_units-df"

    async def relationships(self):
        return "relationships-df"

    async def entities(self):
        return "entities-df"


def _patch_common(monkeypatch):
    monkeypatch.setattr(client_module, "load_config", lambda root_dir: FakeConfig())
    monkeypatch.setattr(client_module, "create_storage", lambda cfg: "fake-storage")
    monkeypatch.setattr(
        client_module,
        "create_table_provider",
        lambda provider, storage: FakeTableProvider(),
    )
    monkeypatch.setattr(client_module, "DataReader", FakeDataReader)


def test_local_search_loads_tables_and_calls_api(monkeypatch):
    _patch_common(monkeypatch)

    captured = {}

    async def fake_local_search(**kwargs):
        captured.update(kwargs)
        return ("the answer", {"some": "context"})

    monkeypatch.setattr(client_module.api, "local_search", fake_local_search)

    result = local_search("what is x?")

    assert result == "the answer"
    assert captured["query"] == "what is x?"
    assert captured["entities"] == "entities-df"
    assert captured["communities"] == "communities-df"
    assert captured["community_reports"] == "community_reports-df"
    assert captured["text_units"] == "text_units-df"
    assert captured["relationships"] == "relationships-df"
    assert captured["covariates"] is None


def test_global_search_loads_tables_and_calls_api(monkeypatch):
    _patch_common(monkeypatch)

    captured = {}

    async def fake_global_search(**kwargs):
        captured.update(kwargs)
        return ("global answer", {"some": "context"})

    monkeypatch.setattr(client_module.api, "global_search", fake_global_search)

    result = global_search("what is x?")

    assert result == "global answer"
    assert captured["query"] == "what is x?"
    assert captured["entities"] == "entities-df"
    assert captured["communities"] == "communities-df"
    assert captured["community_reports"] == "community_reports-df"


def test_load_graphrag_config_restores_original_cwd(monkeypatch, tmp_path):
    original_cwd = Path.cwd()

    def fake_load_config(root_dir):
        os.chdir(root_dir)  # simulate GraphRAG's load_config(set_cwd=True)
        return FakeConfig()

    monkeypatch.setattr(client_module, "load_config", fake_load_config)
    monkeypatch.setattr(client_module, "_GRAPHRAG_ROOT", tmp_path)

    client_module._load_graphrag_config()

    assert Path.cwd() == original_cwd
