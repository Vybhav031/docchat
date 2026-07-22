from app.retrieval import Retriever

CORPUS = [
    {"chunk_id": 1, "document_name": "beaches.md", "position": 0,
     "text": "Rip currents are fast channels of water flowing away from shore at beaches."},
    {"chunk_id": 2, "document_name": "cooking.md", "position": 0,
     "text": "Simmer the tomato sauce gently and season the pasta water with salt."},
    {"chunk_id": 3, "document_name": "ml.md", "position": 0,
     "text": "Object detection models such as YOLO locate people in images in real time."},
]


def make_retriever():
    r = Retriever()
    r.build(CORPUS)
    return r


def test_relevant_chunk_ranks_first():
    r = make_retriever()
    hits = r.search("what is a rip current at the beach", top_k=3)
    assert hits and hits[0].document_name == "beaches.md"


def test_scores_are_descending_and_positive():
    r = make_retriever()
    hits = r.search("object detection with YOLO", top_k=3)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_unrelated_query_returns_no_hits():
    r = make_retriever()
    assert r.search("quantum entanglement thermodynamics", top_k=3) == []


def test_empty_corpus_is_safe():
    r = Retriever()
    r.build([])
    assert r.search("anything", top_k=3) == []
    assert r.size == 0
