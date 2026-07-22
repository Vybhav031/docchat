"""Retrieval evaluation: hit@k and MRR over eval/questions.json.

Runs entirely offline (no API key needed) because it measures the
retrieval stage, which bounds everything downstream. Usage:

    python -m eval.run_eval
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import chunk_text          # noqa: E402
from app.config import settings              # noqa: E402
from app.retrieval import Retriever          # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"
QUESTIONS = Path(__file__).resolve().parent / "questions.json"
K = settings.top_k


def build_corpus() -> Retriever:
    chunks, cid = [], 0
    for path in sorted(SAMPLE_DIR.glob("*.md")):
        for i, piece in enumerate(chunk_text(path.read_text(encoding="utf-8"),
                                             settings.chunk_size, settings.chunk_overlap)):
            chunks.append({"chunk_id": cid, "document_name": path.name,
                           "position": i, "text": piece})
            cid += 1
    r = Retriever()
    r.build(chunks)
    return r


def main() -> int:
    retriever = build_corpus()
    cases = json.loads(QUESTIONS.read_text(encoding="utf-8"))

    hits, rr_sum, kw_hits = 0, 0.0, 0
    rows = []
    for case in cases:
        results = retriever.search(case["question"], top_k=K)
        rank = next((i + 1 for i, h in enumerate(results)
                     if h.document_name == case["expected_doc"]), None)
        keyword_found = any(case["expected_keyword"].lower() in h.text.lower()
                            for h in results)
        if rank:
            hits += 1
            rr_sum += 1.0 / rank
        if keyword_found:
            kw_hits += 1
        rows.append((case["question"][:52], case["expected_doc"],
                     rank or "-", "yes" if keyword_found else "NO"))

    n = len(cases)
    print(f"\nDocChat retrieval evaluation  ({n} questions, k={K})")
    print("-" * 96)
    print(f"{'question':<54}{'expected doc':<28}{'rank':<6}{'keyword'}")
    print("-" * 96)
    for q, doc, rank, kw in rows:
        print(f"{q:<54}{doc:<28}{str(rank):<6}{kw}")
    print("-" * 96)
    print(f"hit@{K}:            {hits}/{n}  ({hits / n:.0%})")
    print(f"MRR:              {rr_sum / n:.3f}")
    print(f"keyword recall:   {kw_hits}/{n}  ({kw_hits / n:.0%})\n")

    # Regression guard: fail the run if retrieval quality collapses.
    return 0 if hits / n >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
