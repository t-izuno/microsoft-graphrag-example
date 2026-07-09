"""LLM-as-judge scoring of GraphRAG/Vector RAG answers against expected answers."""

import json

from evaluation.generate_qa import load_graphrag_completion_model
from vector_rag.completion import complete
from vector_rag.config import ModelConfig

_JUDGE_PROMPT = (
    "あなたは回答の正確さを採点する審査員です。"
    "期待される模範解答と、実際の回答を比較し、"
    "1(全く不正解)から5(模範解答と同等に正確)の"
    "整数スコアと、その理由を日本語で1〜2文で示してください。"
    '出力は{"score": <1から5の整数>, "rationale": "..."}という'
    "JSON形式のみとし、他の文章を含めないでください。"
)


def _judge_answer(
    expected_answer: str, actual_answer: str, completion_model: ModelConfig
) -> dict:
    messages = [
        {"role": "system", "content": _JUDGE_PROMPT},
        {
            "role": "user",
            "content": f"模範解答:\n{expected_answer}\n\n実際の回答:\n{actual_answer}",
        },
    ]
    raw_response = complete(messages, completion_model)
    return json.loads(raw_response)


def judge_results(
    results: list[dict],
    completion_model: ModelConfig | None = None,
    fresh_ids: set[str] | None = None,
    existing_scored_by_id: dict[str, dict] | None = None,
) -> list[dict]:
    """Score every method's answer in each result entry.

    If fresh_ids is given, entries whose id is not in fresh_ids and already
    present in existing_scored_by_id are reused as-is instead of re-judged
    (avoids re-spending LLM calls on answers that were not recomputed).
    """
    completion_model = completion_model or load_graphrag_completion_model()
    existing_scored_by_id = existing_scored_by_id or {}

    scored = []
    for entry in results:
        entry_id = entry["id"]
        if (
            fresh_ids is not None
            and entry_id not in fresh_ids
            and entry_id in existing_scored_by_id
        ):
            scored.append(existing_scored_by_id[entry_id])
            continue

        scores = {
            method: _judge_answer(entry["expected_answer"], answer, completion_model)
            for method, answer in entry["answers"].items()
        }
        scored.append({**entry, "scores": scores})
    return scored
