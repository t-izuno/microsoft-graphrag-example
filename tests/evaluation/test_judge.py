import json

import evaluation.judge as judge_module
from evaluation.judge import judge_results
from vector_rag.config import ModelConfig


def test_judge_results_scores_each_method_answer(monkeypatch):
    completion_model = ModelConfig(model_provider="openai", model="gpt-4.1")

    monkeypatch.setattr(
        judge_module,
        "complete",
        lambda messages, model_config: json.dumps(
            {"score": 4, "rationale": "おおむね正しい"}
        ),
    )

    results = [
        {
            "id": "qa-001",
            "question": "Q",
            "expected_answer": "E",
            "answers": {
                "graphrag_local": "A1",
                "graphrag_global": "A2",
                "vector_rag": "A3",
            },
        }
    ]

    scored = judge_results(results, completion_model=completion_model)

    assert len(scored) == 1
    assert scored[0]["id"] == "qa-001"
    assert scored[0]["scores"]["graphrag_local"] == {
        "score": 4,
        "rationale": "おおむね正しい",
    }
    assert scored[0]["scores"]["graphrag_global"]["score"] == 4
    assert scored[0]["scores"]["vector_rag"]["score"] == 4


def test_judge_results_reuses_existing_score_for_non_fresh_ids(monkeypatch):
    completion_model = ModelConfig(model_provider="openai", model="gpt-4.1")

    call_count = {"n": 0}

    def fake_complete(messages, model_config):
        call_count["n"] += 1
        return json.dumps({"score": 4, "rationale": "新規採点"})

    monkeypatch.setattr(judge_module, "complete", fake_complete)

    results = [
        {
            "id": "qa-001",
            "question": "Q1",
            "expected_answer": "E1",
            "answers": {"graphrag_local": "A1"},
        },
        {
            "id": "qa-002",
            "question": "Q2",
            "expected_answer": "E2",
            "answers": {"graphrag_local": "A2"},
        },
    ]
    existing_scored_by_id = {
        "qa-001": {
            "id": "qa-001",
            "question": "Q1",
            "expected_answer": "E1",
            "answers": {"graphrag_local": "A1"},
            "scores": {"graphrag_local": {"score": 5, "rationale": "既存の採点"}},
        }
    }

    scored = judge_results(
        results,
        completion_model=completion_model,
        fresh_ids={"qa-002"},
        existing_scored_by_id=existing_scored_by_id,
    )

    assert call_count["n"] == 1  # only qa-002 was actually re-judged
    scored_by_id = {entry["id"]: entry for entry in scored}
    assert scored_by_id["qa-001"]["scores"]["graphrag_local"]["rationale"] == "既存の採点"
    assert scored_by_id["qa-002"]["scores"]["graphrag_local"]["rationale"] == "新規採点"
