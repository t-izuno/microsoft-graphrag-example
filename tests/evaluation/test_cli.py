from typer.testing import CliRunner

import evaluation.cli as cli_module
from evaluation.cli import app
from evaluation.run import NoReviewedQAError

runner = CliRunner()


def test_generate_qa_command_reports_entry_count(monkeypatch):
    captured = {}

    def fake_generate_qa_dataset(target_count):
        captured["target_count"] = target_count
        return [{"id": "qa-001"}, {"id": "qa-002"}]

    monkeypatch.setattr(cli_module, "generate_qa_dataset", fake_generate_qa_dataset)

    result = runner.invoke(app, ["generate-qa", "--target-count", "10"])

    assert result.exit_code == 0
    assert captured["target_count"] == 10
    assert "2" in result.stdout


def test_run_command_reports_all_output_paths(monkeypatch):
    captured = {}

    def fake_run_evaluation(run_id, question_ids=None):
        captured["run_id"] = run_id
        captured["question_ids"] = question_ids
        return {
            "output_path": f"results/{run_id}.json",
            "scored_path": f"results/{run_id}-scored.json",
            "report_path": f"results/{run_id}-report.md",
            "results": [1, 2, 3],
            "scored": [1, 2, 3],
        }

    monkeypatch.setattr(cli_module, "run_evaluation", fake_run_evaluation)

    result = runner.invoke(app, ["run", "my-run"])

    assert result.exit_code == 0
    assert captured["run_id"] == "my-run"
    assert captured["question_ids"] is None
    assert "results/my-run.json" in result.stdout
    assert "results/my-run-scored.json" in result.stdout
    assert "results/my-run-report.md" in result.stdout


def test_run_command_generates_run_id_when_omitted(monkeypatch):
    monkeypatch.setattr(cli_module, "_generate_run_id", lambda: "auto-generated-id")
    captured = {}

    def fake_run_evaluation(run_id, question_ids=None):
        captured["run_id"] = run_id
        return {
            "output_path": "results/auto-generated-id.json",
            "scored_path": "results/auto-generated-id-scored.json",
            "report_path": "results/auto-generated-id-report.md",
            "results": [],
            "scored": [],
        }

    monkeypatch.setattr(cli_module, "run_evaluation", fake_run_evaluation)

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    assert captured["run_id"] == "auto-generated-id"
    assert "auto-generated-id" in result.stdout


def test_run_command_passes_question_id_options_through(monkeypatch):
    captured = {}

    def fake_run_evaluation(run_id, question_ids=None):
        captured["question_ids"] = question_ids
        return {
            "output_path": "p",
            "scored_path": "p",
            "report_path": "p",
            "results": [],
            "scored": [],
        }

    monkeypatch.setattr(cli_module, "run_evaluation", fake_run_evaluation)

    result = runner.invoke(
        app, ["run", "my-run", "--question-id", "qa-002", "-q", "qa-005"]
    )

    assert result.exit_code == 0
    assert captured["question_ids"] == ["qa-002", "qa-005"]


def test_run_command_exits_with_error_when_no_reviewed_entries(monkeypatch):
    def fake_run_evaluation(run_id, question_ids=None):
        raise NoReviewedQAError("reviewed: trueのQAエントリがありません。")

    monkeypatch.setattr(cli_module, "run_evaluation", fake_run_evaluation)

    result = runner.invoke(app, ["run", "my-run"])

    assert result.exit_code != 0
    assert "reviewed: true" in result.stderr
