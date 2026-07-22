"""Rank stored chunks against a question.

Default scorer is TF-IDF + cosine similarity: fast, dependency-light,
fully offline, and easy to reason about. The Retriever rebuilds its
index whenever the corpus changes (fine at this scale; a vector
database becomes worthwhile when chunk counts reach the hundreds of
thousands).
"""
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Hit:
    chunk_id: int
    document_name: str
    position: int
    text: str
    score: float


class Retriever:
    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._meta: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        """chunks: [{"chunk_id", "document_name", "position", "text"}, ...]"""
        self._meta = chunks
        if not chunks:
            self._vectorizer, self._matrix = None, None
            return
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(c["text"] for c in chunks)

    @property
    def size(self) -> int:
        return len(self._meta)

    def search(self, question: str, top_k: int = 4) -> list[Hit]:
        if not self._meta or self._vectorizer is None:
            return []
        q_vec = self._vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self._matrix)[0]
        order = scores.argsort()[::-1][:top_k]
        return [
            Hit(
                chunk_id=self._meta[i]["chunk_id"],
                document_name=self._meta[i]["document_name"],
                position=self._meta[i]["position"],
                text=self._meta[i]["text"],
                score=round(float(scores[i]), 4),
            )
            for i in order
            if scores[i] > 0
        ]


retriever = Retriever()
