from typer.testing import CliRunner

import vector_rag.cli as cli_module
from vector_rag.cli import app

runner = CliRunner()


def test_index_command_reports_chunk_count(monkeypatch):
    monkeypatch.setattr(cli_module, "load_config", lambda: "fake-config")
    monkeypatch.setattr(cli_module, "run_index", lambda config: 5)

    result = runner.invoke(app, ["index"])

    assert result.exit_code == 0
    assert "5" in result.stdout


def test_query_command_prints_answer(monkeypatch):
    monkeypatch.setattr(cli_module, "load_config", lambda: "fake-config")
    captured = {}

    def fake_answer_query(question, config):
        captured["question"] = question
        return "the answer"

    monkeypatch.setattr(cli_module, "answer_query", fake_answer_query)

    result = runner.invoke(app, ["query", "what is x?"])

    assert result.exit_code == 0
    assert "the answer" in result.stdout
    assert captured["question"] == "what is x?"
