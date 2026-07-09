from evaluation.report import build_report_markdown

_SCORED_RESULTS = [
    {
        "id": "qa-001",
        "question": "Q1",
        "scores": {
            "graphrag_local": {"score": 4, "rationale": "r"},
            "graphrag_global": {"score": 2, "rationale": "r"},
            "vector_rag": {"score": 3, "rationale": "r"},
        },
    },
    {
        "id": "qa-002",
        "question": "Q2",
        "scores": {
            "graphrag_local": {"score": 2, "rationale": "r"},
            "graphrag_global": {"score": 4, "rationale": "r"},
            "vector_rag": {"score": 5, "rationale": "r"},
        },
    },
]


def test_build_report_markdown_includes_averages_and_per_question_rows():
    markdown = build_report_markdown(_SCORED_RESULTS)

    assert "GraphRAG (local)" in markdown
    assert "GraphRAG (global)" in markdown
    assert "Vector RAG" in markdown
    assert "3.00" in markdown  # average of graphrag_local scores (4, 2)
    assert "4.00" in markdown  # average of vector_rag scores (3, 5)
    assert "qa-001" in markdown
    assert "Q2" in markdown
