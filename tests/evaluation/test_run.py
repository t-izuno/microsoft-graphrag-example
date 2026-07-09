import json

import pytest
import yaml

import evaluation.run as run_module
from evaluation.run import NoReviewedQAError, run_evaluation


def _write_qa_dataset(tmp_path, entries=None):
    path = tmp_path / "qa_dataset.yaml"
    entries = (
        entries
        if entries is not None
        else [
            {
                "id": "qa-001",
                "question": "question one",
                "expected_answer": "answer one",
                "source_chunk_id": "doc-0",
                "reviewed": True,
            },
            {
                "id": "qa-002",
                "question": "question two",
                "expected_answer": "answer two",
                "source_chunk_id": "doc-1",
                "reviewed": True,
            },
            {
                "id": "qa-003",
                "question": "unreviewed question",
                "expected_answer": "unreviewed answer",
                "source_chunk_id": "doc-2",
                "reviewed": False,
            },
        ]
    )
    path.write_text(yaml.safe_dump(entries, allow_unicode=True), encoding="utf-8")
    return path


def _patch_searches(monkeypatch):
    call_log = []

    def make(name):
        def fn(question):
            call_log.append((name, question))
            return f"{name}:{question}"

        return fn

    monkeypatch.setattr(run_module, "local_search", make("local"))
    monkeypatch.setattr(run_module, "global_search", make("global"))
    monkeypatch.setattr(run_module, "vector_rag_search", make("vector_rag"))
    return call_log


def _patch_judge_and_report(monkeypatch):
    def fake_judge_results(
        results, completion_model=None, fresh_ids=None, existing_scored_by_id=None
    ):
        existing_scored_by_id = existing_scored_by_id or {}
        scored = []
        for entry in results:
            if (
                fresh_ids is not None
                and entry["id"] not in fresh_ids
                and entry["id"] in existing_scored_by_id
            ):
                scored.append(existing_scored_by_id[entry["id"]])
                continue
            scored.append(
                {**entry, "scores": {"graphrag_local": {"score": 5, "rationale": "r"}}}
            )
        return scored

    monkeypatch.setattr(run_module, "judge_results", fake_judge_results)
    monkeypatch.setattr(
        run_module, "build_report_markdown", lambda scored: f"REPORT:{len(scored)}"
    )


def test_run_evaluation_processes_reviewed_entries_and_writes_results_scored_report(
    monkeypatch, tmp_path
):
    qa_dataset_path = _write_qa_dataset(tmp_path)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(run_module, "load_vector_rag_config", lambda: "fake-config")
    monkeypatch.setattr(run_module, "check_provider_consistency", lambda cfg: [])
    call_log = _patch_searches(monkeypatch)
    _patch_judge_and_report(monkeypatch)

    summary = run_evaluation(
        run_id="test-run", qa_dataset_path=qa_dataset_path, results_dir=results_dir
    )

    assert len(summary["results"]) == 2  # qa-003 is not reviewed, excluded
    assert {c[1] for c in call_log if c[0] == "local"} == {
        "question one",
        "question two",
    }

    results_path = results_dir / "test-run.json"
    scored_path = results_dir / "test-run-scored.json"
    report_path = results_dir / "test-run-report.md"
    assert summary["output_path"] == str(results_path)
    assert summary["scored_path"] == str(scored_path)
    assert summary["report_path"] == str(report_path)
    assert json.loads(results_path.read_text(encoding="utf-8")) == summary["results"]
    assert json.loads(scored_path.read_text(encoding="utf-8")) == summary["scored"]
    assert report_path.read_text(encoding="utf-8") == "REPORT:2"


def test_run_evaluation_prints_config_mismatch_warnings_but_continues(
    monkeypatch, tmp_path, capsys
):
    qa_dataset_path = _write_qa_dataset(tmp_path)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(run_module, "load_vector_rag_config", lambda: "fake-config")
    monkeypatch.setattr(
        run_module,
        "check_provider_consistency",
        lambda cfg: ["completion_model: mismatch"],
    )
    _patch_searches(monkeypatch)
    _patch_judge_and_report(monkeypatch)

    summary = run_evaluation(
        run_id="test-run-2", qa_dataset_path=qa_dataset_path, results_dir=results_dir
    )

    assert len(summary["results"]) == 2
    assert "mismatch" in capsys.readouterr().out


def test_run_evaluation_with_question_ids_only_recomputes_those(monkeypatch, tmp_path):
    qa_dataset_path = _write_qa_dataset(tmp_path)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(run_module, "load_vector_rag_config", lambda: "fake-config")
    monkeypatch.setattr(run_module, "check_provider_consistency", lambda cfg: [])
    call_log = _patch_searches(monkeypatch)
    _patch_judge_and_report(monkeypatch)

    run_evaluation(
        run_id="run-1", qa_dataset_path=qa_dataset_path, results_dir=results_dir
    )
    call_log.clear()

    summary = run_evaluation(
        run_id="run-1",
        question_ids=["qa-002"],
        qa_dataset_path=qa_dataset_path,
        results_dir=results_dir,
    )

    assert {c[1] for c in call_log} == {"question two"}
    result_by_id = {r["id"]: r for r in summary["results"]}
    assert result_by_id["qa-001"]["answers"]["graphrag_local"] == "local:question one"
    assert result_by_id["qa-002"]["answers"]["graphrag_local"] == "local:question two"


def test_run_evaluation_raises_when_no_reviewed_entries(monkeypatch, tmp_path):
    qa_dataset_path = _write_qa_dataset(
        tmp_path,
        entries=[
            {
                "id": "qa-001",
                "question": "q",
                "expected_answer": "a",
                "source_chunk_id": "doc-0",
                "reviewed": False,
            }
        ],
    )
    results_dir = tmp_path / "results"
    monkeypatch.setattr(run_module, "load_vector_rag_config", lambda: "fake-config")

    with pytest.raises(NoReviewedQAError):
        run_evaluation(
            run_id="test-run",
            qa_dataset_path=qa_dataset_path,
            results_dir=results_dir,
        )
