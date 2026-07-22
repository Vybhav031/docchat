import pytest

from app.chunking import chunk_text, split_paragraphs


def test_paragraph_splitting_drops_blanks():
    text = "one\n\n\n\ntwo\n\nthree  "
    assert split_paragraphs(text) == ["one", "two", "three"]


def test_chunks_respect_size_limit():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(20))
    for chunk in chunk_text(text, chunk_size=400, overlap=50):
        assert 0 < len(chunk) <= 400


def test_long_paragraph_is_hard_split():
    text = "x" * 2500
    chunks = chunk_text(text, chunk_size=900, overlap=100)
    assert len(chunks) >= 3
    assert all(len(c) <= 900 for c in chunks)


def test_overlap_carries_text_between_chunks():
    text = "\n\n".join("sentence %d %s" % (i, "filler " * 40) for i in range(6))
    chunks = chunk_text(text, chunk_size=500, overlap=120)
    assert len(chunks) >= 2
    # The tail of chunk N should appear at the head of chunk N+1.
    assert chunks[0][-40:] in chunks[1]


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, overlap=100)
