"""litellm completion wrapper."""

import litellm

from vector_rag.config import ModelConfig


def complete(messages: list[dict], config: ModelConfig) -> str:
    """Generate a completion for the given chat messages."""
    response = litellm.completion(
        model=f"{config.model_provider}/{config.model}",
        messages=messages,
        api_key=config.api_key,
        api_base=config.api_base,
    )
    return response.choices[0].message.content
