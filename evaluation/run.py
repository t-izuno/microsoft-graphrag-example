"""Run GraphRAG/Vector RAG, judge the answers, and write the report — in one call."""

import json
from pathlib import Path

import yaml

from evaluation.config_check import check_provider_consistency
from evaluation.graphrag_client import global_search, local_search
from evaluation.judge import judge_results
from evaluation.report import build_report_markdown
from evaluation.vector_rag_client import vector_rag_search
from vector_rag.config import load_config as load_vector_rag_config

_QA_DATASET_PATH = Path(__file__).parent / "qa_dataset.yaml"
_RESULTS_DIR = Path(__file__).parent / "results"


class NoReviewedQAError(Exception):
    """Raised when qa_dataset.yaml has no reviewed: true entries to process."""


def _load_reviewed_qa(qa_dataset_path: Path) -> list[dict]:
    entries = yaml.safe_load(qa_dataset_path.read_text(encoding="utf-8")) or []
    return [entry for entry in entries if entry.get("reviewed")]


def _load_by_id(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in entries}


def run_evaluation(
    run_id: str,
    question_ids: list[str] | None = None,
    qa_dataset_path: Path | None = None,
    results_dir: Path | None = None,
    vector_rag_config=None,
    completion_model=None,
) -> dict:
    """Collect answers, judge them, and write the comparison report for one run.

    By default every reviewed QA entry is (re)collected and (re)judged. Pass
    question_ids to redo only those entries; answers/scores for the rest are
    reused from the existing run_id's results/scored files, if present, so a
    single failed or edited question doesn't require re-running the whole
    (expensive, LLM-calling) batch.
    """
    vector_rag_config = vector_rag_config or load_vector_rag_config()

    qa_entries = _load_reviewed_qa(qa_dataset_path or _QA_DATASET_PATH)
    if not qa_entries:
        msg = (
            "reviewed: trueのQAエントリがありません。"
            "evaluation/qa_dataset.yamlのレビューを完了してください。"
        )
        raise NoReviewedQAError(msg)

    for warning in check_provider_consistency(vector_rag_config):
        print(f"警告: {warning}")

    results_dir = results_dir or _RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{run_id}.json"
    scored_path = results_dir / f"{run_id}-scored.json"
    report_path = results_dir / f"{run_id}-report.md"

    existing_results_by_id = _load_by_id(results_path)
    question_id_set = set(question_ids) if question_ids is not None else None

    results = []
    fresh_ids: set[str] = set()
    for entry in qa_entries:
        qa_id = entry["id"]
        reuse = (
            question_id_set is not None
            and qa_id not in question_id_set
            and qa_id in existing_results_by_id
        )
        if reuse:
            results.append(existing_results_by_id[qa_id])
            continue

        fresh_ids.add(qa_id)
        question = entry["question"]
        results.append(
            {
                "id": qa_id,
                "question": question,
                "expected_answer": entry["expected_answer"],
                "answers": {
                    "graphrag_local": local_search(question),
                    "graphrag_global": global_search(question),
                    "vector_rag": vector_rag_search(question),
                },
            }
        )

    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    existing_scored_by_id = _load_by_id(scored_path)
    scored = judge_results(
        results,
        completion_model=completion_model,
        fresh_ids=fresh_ids,
        existing_scored_by_id=existing_scored_by_id,
    )
    scored_path.write_text(
        json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_path.write_text(build_report_markdown(scored), encoding="utf-8")

    return {
        "output_path": str(results_path),
        "scored_path": str(scored_path),
        "report_path": str(report_path),
        "results": results,
        "scored": scored,
    }
