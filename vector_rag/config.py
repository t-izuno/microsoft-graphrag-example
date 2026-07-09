"""Configuration loading for the vector_rag pipeline."""

import os
from pathlib import Path
from string import Template

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

_DEFAULT_SETTINGS_PATH = Path(__file__).parent / "settings.yaml"
_GRAPHRAG_ENV_PATH = Path(__file__).parent.parent / "graphrag" / ".env"


class InputConfig(BaseModel):
    """Location of the input documents shared with GraphRAG."""

    base_dir: str


class ModelConfig(BaseModel):
    """LLM or embedding model connection settings."""

    model_provider: str
    model: str
    auth_method: str | None = None
    api_key: str | None = None
    api_base: str | None = None


class ChunkingConfig(BaseModel):
    """Token-based chunking parameters."""

    size: int
    overlap: int
    encoding_model: str


class VectorStoreConfig(BaseModel):
    """LanceDB connection settings."""

    db_uri: str
    table_name: str


class RetrievalConfig(BaseModel):
    """Similarity search parameters."""

    top_k: int


class VectorRagConfig(BaseModel):
    """Fully resolved vector_rag configuration."""

    input: InputConfig
    completion_model: ModelConfig
    embedding_model: ModelConfig
    chunking: ChunkingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig


def load_config(settings_path: Path | None = None) -> VectorRagConfig:
    """Load and resolve vector_rag configuration from a settings YAML file."""
    settings_path = Path(settings_path or _DEFAULT_SETTINGS_PATH)
    settings_dir = settings_path.parent

    if _GRAPHRAG_ENV_PATH.is_file():
        load_dotenv(_GRAPHRAG_ENV_PATH)

    raw_text = settings_path.read_text(encoding="utf-8")
    substituted_text = Template(raw_text).substitute(os.environ)
    data = yaml.safe_load(substituted_text)

    data["input"]["base_dir"] = str(
        (settings_dir / data["input"]["base_dir"]).resolve()
    )
    data["vector_store"]["db_uri"] = str(
        (settings_dir / data["vector_store"]["db_uri"]).resolve()
    )

    return VectorRagConfig(**data)
