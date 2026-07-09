"""Aggregate judged scores into a Markdown comparison report."""

_METHODS = ["graphrag_local", "graphrag_global", "vector_rag"]
_METHOD_LABELS = {
    "graphrag_local": "GraphRAG (local)",
    "graphrag_global": "GraphRAG (global)",
    "vector_rag": "Vector RAG",
}


def _average_scores(scored_results: list[dict]) -> dict[str, float]:
    averages = {}
    for method in _METHODS:
        scores = [
            entry["scores"][method]["score"]
            for entry in scored_results
            if method in entry["scores"]
        ]
        averages[method] = sum(scores) / len(scores) if scores else 0.0
    return averages


def build_report_markdown(scored_results: list[dict]) -> str:
    """Build a Markdown report comparing methods by average and per-question score."""
    averages = _average_scores(scored_results)

    lines = [
        "# Vector RAG比較評価レポート",
        "",
        "## 手法別 平均スコア",
        "",
        "| 手法 | 平均スコア |",
        "| --- | --- |",
    ]
    for method in _METHODS:
        lines.append(f"| {_METHOD_LABELS[method]} | {averages[method]:.2f} |")

    lines.extend(
        [
            "",
            "## 質問別スコア",
            "",
            "| ID | 質問 | GraphRAG (local) | GraphRAG (global) | Vector RAG |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for entry in scored_results:
        scores = entry["scores"]
        row = [
            entry["id"],
            entry["question"],
            *(str(scores.get(method, {}).get("score", "-")) for method in _METHODS),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    return "\n".join(lines)
