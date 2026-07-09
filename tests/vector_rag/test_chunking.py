from vector_rag.chunking import split_text_on_tokens


def _encode(text: str) -> list[int]:
    return [ord(c) for c in text]


def _decode(token_ids: list[int]) -> str:
    return "".join(chr(t) for t in token_ids)


def test_split_text_on_tokens_single_chunk_when_shorter_than_chunk_size():
    chunks = split_text_on_tokens(
        "hello",
        chunk_size=10,
        chunk_overlap=2,
        encode=_encode,
        decode=_decode,
    )

    assert chunks == ["hello"]


def test_split_text_on_tokens_slides_window_with_overlap():
    # 10 characters, chunk_size=4, overlap=1 => stride=3
    text = "abcdefghij"

    chunks = split_text_on_tokens(
        text,
        chunk_size=4,
        chunk_overlap=1,
        encode=_encode,
        decode=_decode,
    )

    assert chunks == ["abcd", "defg", "ghij"]
