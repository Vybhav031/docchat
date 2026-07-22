"""Split raw document text into retrieval-sized chunks.

Strategy: split on paragraph boundaries first (chunks that respect
meaning retrieve better than fixed-size slices), then pack paragraphs
into chunks of roughly `chunk_size` characters with `overlap`
characters carried between consecutive chunks so that answers spanning
a boundary are still retrievable.
"""


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in paragraphs if p]


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = split_paragraphs(text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # A single paragraph larger than chunk_size gets hard-split.
        while len(para) > chunk_size:
            head, para = para[:chunk_size], para[chunk_size - overlap:]
            chunks.append((current + "\n\n" + head).strip() if current else head)
            current = ""
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            # Carry a tail of the previous chunk forward as overlap.
            tail = current[-overlap:] if overlap else ""
            current = (tail + "\n\n" + para).strip() if tail else para

    if current:
        chunks.append(current)
    return chunks
