"""litellm embedding wrapper."""

import litellm

from vector_rag.config import ModelConfig


def embed_texts(texts: list[str], config: ModelConfig) -> list[list[float]]:
    """Embed a batch of texts using the configured embedding model."""
    response = litellm.embedding(
        model=f"{config.model_provider}/{config.model}",
        input=texts,
        api_key=config.api_key,
        api_base=config.api_base,
    )
    return [item["embedding"] for item in response.data]
