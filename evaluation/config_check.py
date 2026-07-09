"""Warn when GraphRAG and Vector RAG use different LLM/embedding models."""

from pathlib import Path

import yaml

from vector_rag.config import ModelConfig, VectorRagConfig

_DEFAULT_GRAPHRAG_SETTINGS_PATH = (
    Path(__file__).parent.parent / "graphrag" / "settings.yaml"
)


def check_provider_consistency(
    vector_rag_config: VectorRagConfig,
    graphrag_settings_path: Path | None = None,
) -> list[str]:
    """Compare model_provider/model between graphrag and vector_rag configs.

    Returns human-readable warning messages; an empty list means consistent.
    """
    settings_path = graphrag_settings_path or _DEFAULT_GRAPHRAG_SETTINGS_PATH
    graphrag_data = yaml.safe_load(settings_path.read_text(encoding="utf-8"))

    warnings: list[str] = []
    warnings.extend(
        _compare(
            "completion_model",
            graphrag_data["completion_models"]["default_completion_model"],
            vector_rag_config.completion_model,
        )
    )
    warnings.extend(
        _compare(
            "embedding_model",
            graphrag_data["embedding_models"]["default_embedding_model"],
            vector_rag_config.embedding_model,
        )
    )
    return warnings


def _compare(
    label: str, graphrag_model: dict, vector_rag_model: ModelConfig
) -> list[str]:
    if (
        graphrag_model["model_provider"] == vector_rag_model.model_provider
        and graphrag_model["model"] == vector_rag_model.model
    ):
        return []
    return [
        f"{label}: graphragとvector_ragでmodel_provider/modelが一致しません "
        f"(graphrag={graphrag_model['model_provider']}/{graphrag_model['model']}, "
        f"vector_rag={vector_rag_model.model_provider}/{vector_rag_model.model})"
    ]
