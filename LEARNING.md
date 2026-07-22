# Understanding this codebase

Read this before any interview where DocChat comes up. It walks the code in
the order a question travels through it, and ends with the questions an
interviewer is most likely to ask. **Do not skip the exercises** — being
able to modify this code matters more than being able to describe it.

## The life of a question

1. **Browser** (`static/index.html`) — `ask()` POSTs
   `{"question": "..."}` to `/api/ask`, then renders the answer bubble and
   the Evidence cards. Plain fetch, no framework.

2. **API layer** (`app/main.py`, `ask()` route) — Pydantic validates the
   body (`AskRequest`: 3–1000 chars → automatic `422` otherwise), the route
   checks the corpus isn't empty (`400`), calls the retriever, calls the
   LLM layer, saves the exchange, returns `AskResponse`.

3. **Retrieval** (`app/retrieval.py`) — the question becomes a TF-IDF
   vector using the same vectorizer fitted on the corpus; cosine similarity
   against every chunk vector; top-k indices with positive scores come back
   as `Hit` objects. The index is rebuilt from the database whenever a
   document is added or deleted (`rebuild_index` in `main.py`).

4. **Generation** (`app/llm.py`) — with an API key: the hits are numbered
   `[1]..[k]`, packed into a prompt whose system message forbids using
   anything outside them, and Claude answers with inline citations. Without
   a key: the top passages are returned verbatim (`retrieval-only` mode) —
   which is also what CI tests against, so tests are deterministic and free.

5. **Persistence** (`app/db.py`) — three tables: `documents`, `chunks`
   (many-to-one, cascade delete), `exchanges` (question, answer, sources as
   JSON, mode). SQLAlchemy 2.0 typed models; `DATABASE_URL` picks SQLite or
   MySQL.

## The pieces most worth understanding deeply

**Chunking (`app/chunking.py`).** Split on blank lines into paragraphs;
pack paragraphs into ≤ `chunk_size` chars; when a chunk closes, carry its
last `overlap` chars into the next one. Oversized single paragraphs get
hard-split. Why overlap? An answer sitting across a chunk boundary would
otherwise be retrievable by neither chunk. Why paragraphs? They usually
carry one idea, so relevance scores stay clean.

**TF-IDF (`app/retrieval.py`).** Term frequency × inverse document
frequency: a word scores high in a chunk if it's frequent there but rare
across the corpus. `ngram_range=(1,2)` also indexes two-word phrases;
`stop_words="english"` drops "the"/"of"/etc. Cosine similarity compares
direction, not length, so long and short chunks compete fairly.

**The grounding prompt (`app/llm.py`, `SYSTEM_PROMPT`).** Three rules:
only the provided passages; cite `[n]`; say when the context lacks the
answer. This is the anti-hallucination mechanism — remove it and the model
happily answers from training data.

**The eval (`eval/run_eval.py`).** 14 questions with known source docs.
hit@k = did the right doc appear in the top k. MRR = 1/rank averaged, so
rank 1 beats rank 4. It runs in CI and fails the build below 80% hit@k —
retrieval regressions become build failures instead of vibes.

## Known limitation — and why it's a feature of the story

The eval's one failing question ("Why does chunk size matter?") fails
because the corpus says *chunking*, never "chunk size" — TF-IDF has no idea
those are related. This is **vocabulary mismatch**, the canonical weakness
of lexical retrieval, caught by our own eval. The fix would be embeddings
(dense vectors where synonyms are neighbours) or a hybrid retriever. In an
interview, walking through this failure → measurement → planned fix is a
stronger story than claiming 100%.

## Exercises (do at least the first two before interviewing)

1. Change `TOP_K` to 2 and re-run `python -m eval.run_eval`. Explain the
   score change.
2. Add a question of your own to `eval/questions.json` that you predict
   will fail, and confirm it. Then reword it to pass.
3. Add an endpoint `GET /api/documents/{id}/chunks` returning a document's
   chunks. Write a test for it. (~20 lines total — touches every layer.)
4. Swap the retriever: implement `EmbeddingRetriever` with
   `sentence-transformers`, keep the same `search()` signature, and compare
   eval scores against TF-IDF.

## Questions an interviewer will ask — with pointers, not scripts

- *Why TF-IDF and not embeddings?* → Design decisions section of README;
  baseline-first + the eval exists to justify the upgrade with numbers.
- *How do you prevent hallucination?* → grounding prompt + visible sources;
  also the honest answer: reduce, not prevent.
- *What happens when retrieval fails?* → positive-score filter can return
  zero hits → "no-context" mode tells the user instead of guessing.
- *Why cosine similarity?* → length-invariance; `02_retrieval_methods.md`
  in the sample corpus explains it (yes, the demo corpus doubles as your
  study notes).
- *How would this scale?* → rebuild-on-ingest and in-memory matrix are fine
  to ~10⁵ chunks; past that: persistent vector index / vector DB,
  incremental updates, retrieval service split out.
- *Why does CI run without an API key?* → retrieval-only mode makes tests
  deterministic and free; the LLM call is the one thing mocked away.
- *Idempotency / duplicates?* → uploads are rejected with `409` on name
  collision; deletes are idempotent in effect (second call → `404`).
- *SQL schema?* → be able to draw documents / chunks / exchanges from
  memory, including the FK and cascade.

One more, non-technical, and you should decide your answer now: *"Did you
build this yourself?"* The honest answer is that it was built with AI
assistance and you studied and extended every part — which, in 2026, is a
normal and even attractive workflow **if** the exercises above are actually
done. The unforgivable version is claiming solo authorship and then being
unable to explain the chunker.
