
def chunk_text(
    text,
    chunk_size=300,
    overlap=50
):
    """
    Split text into overlapping word-based chunks.

    chunk_size:
        Number of words in each chunk.

    overlap:
        Number of words repeated between consecutive chunks.
    """

    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk_size."
        )

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

        if start + chunk_size >= len(words):
            break

    return chunks