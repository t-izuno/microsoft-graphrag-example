"""Token-based chunking, mirroring GraphRAG's TokenTextSplitter algorithm.

See graphrag/index/text_splitting/text_splitting.py
(split_single_text_on_tokens) in the installed graphrag package for the
reference implementation this intentionally matches.
"""

from collections.abc import Callable

import tiktoken

from vector_rag.config import ChunkingConfig

EncodeFn = Callable[[str], list[int]]
DecodeFn = Callable[[list[int]], str]


def split_text_on_tokens(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    encode: EncodeFn,
    decode: DecodeFn,
) -> list[str]:
    """Split text into overlapping windows of chunk_size tokens each."""
    result = []
    input_ids = encode(text)

    start_idx = 0
    cur_idx = min(start_idx + chunk_size, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]

    while start_idx < len(input_ids):
        result.append(decode(list(chunk_ids)))
        if cur_idx == len(input_ids):
            break
        start_idx += chunk_size - chunk_overlap
        cur_idx = min(start_idx + chunk_size, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]

    return result


def split_text(text: str, config: ChunkingConfig) -> list[str]:
    """Split text using the encoding named in the chunking config."""
    encoding = tiktoken.get_encoding(config.encoding_model)
    return split_text_on_tokens(
        text,
        chunk_size=config.size,
        chunk_overlap=config.overlap,
        encode=encoding.encode,
        decode=encoding.decode,
    )
