from src.chunking import chunk_text

def test_chunk_text_returns_content():
    chunks = chunk_text("This is a sentence. " * 100, chunk_size=100, overlap=10)
    assert chunks
    assert all(isinstance(c, str) and c for c in chunks)

def test_invalid_overlap():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, overlap=10)
