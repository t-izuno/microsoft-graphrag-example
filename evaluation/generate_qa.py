"""Generate a candidate evaluation QA dataset from the shared input documents."""

import json
import os
from pathlib import Path
from string import Template

import yaml
from dotenv import load_dotenv

from vector_rag.chunking import split_text
from vector_rag.completion import complete
from vector_rag.config import ModelConfig, VectorRagConfig
from vector_rag.config import load_config as load_vector_rag_config

_GRAPHRAG_SETTINGS_PATH = Path(__file__).parent.parent / "graphrag" / "settings.yaml"
_GRAPHRAG_ENV_PATH = Path(__file__).parent.parent / "graphrag" / ".env"
_QA_DATASET_PATH = Path(__file__).parent / "qa_dataset.yaml"

_TARGET_QA_COUNT = 40

_QA_GENERATION_PROMPT = (
    "次の文章の内容から、質問と模範解答のペアを1つ作成してください。"
    "文章の内容だけから答えられる具体的な質問にしてください。"
    '出力は{"question": "...", "expected_answer": "..."}という'
    "JSON形式のみとし、他の文章を含めないでください。"
)


def load_graphrag_completion_model(settings_path: Path | None = None) -> ModelConfig:
    settings_path = settings_path or _GRAPHRAG_SETTINGS_PATH
    if _GRAPHRAG_ENV_PATH.is_file():
        load_dotenv(_GRAPHRAG_ENV_PATH)
    raw_text = settings_path.read_text(encoding="utf-8")
    substituted_text = Template(raw_text).substitute(os.environ)
    data = yaml.safe_load(substituted_text)
    return ModelConfig(**data["completion_models"]["default_completion_model"])


def _generate_qa_pair(chunk_text: str, completion_model: ModelConfig) -> dict:
    messages = [
        {"role": "system", "content": _QA_GENERATION_PROMPT},
        {"role": "user", "content": chunk_text},
    ]
    raw_response = complete(messages, completion_model)
    return json.loads(raw_response)


def generate_qa_dataset(
    target_count: int = _TARGET_QA_COUNT,
    vector_rag_config: VectorRagConfig | None = None,
    completion_model: ModelConfig | None = None,
    output_path: Path | None = None,
) -> list[dict]:
    """Sample chunks across all input documents and generate a QA pair each."""
    vector_rag_config = vector_rag_config or load_vector_rag_config()
    completion_model = completion_model or load_graphrag_completion_model()
    output_path = output_path or _QA_DATASET_PATH

    input_dir = Path(vector_rag_config.input.base_dir)

    entries: list[dict] = []
    for file_path in sorted(input_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        chunks = split_text(text, vector_rag_config.chunking)
        if not chunks:
            continue

        stride = max(1, len(chunks) // target_count)
        sampled = list(enumerate(chunks))[::stride][:target_count]

        for chunk_index, chunk_text in sampled:
            qa_pair = _generate_qa_pair(chunk_text, completion_model)
            entries.append(
                {
                    "id": f"qa-{len(entries) + 1:03d}",
                    "question": qa_pair["question"],
                    "expected_answer": qa_pair["expected_answer"],
                    "source_chunk_id": f"{file_path.stem}-{chunk_index}",
                    "reviewed": False,
                }
            )

    output_path.write_text(
        yaml.safe_dump(entries, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return entries
