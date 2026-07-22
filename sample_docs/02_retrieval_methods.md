# How retrieval works

The retrieval stage ranks every chunk in the corpus by how well it matches
the question. Two families of methods dominate: lexical and semantic.

Lexical retrieval matches words. TF-IDF (term frequency–inverse document
frequency) scores a chunk highly when it contains the question's words,
weighted so that rare, distinctive words count more than common ones. The
question and every chunk become sparse vectors of word weights, and cosine
similarity between vectors produces the ranking. TF-IDF is fast, needs no
machine learning model, runs entirely offline, and its scores are easy to
interpret and debug. Its weakness is vocabulary mismatch: a question about
"cars" will not match a chunk that only says "automobiles."

Semantic retrieval matches meaning. An embedding model converts text into
dense vectors positioned so that similar meanings land close together;
"cars" and "automobiles" end up as neighbours even though they share no
letters. Retrieval becomes a nearest-neighbour search in that vector space.
Embeddings handle paraphrase and synonyms far better than lexical methods,
at the cost of a model dependency, slower indexing, and scores that are
harder to interpret.

Production systems often combine both: a hybrid retriever runs lexical and
semantic search together and merges the rankings, which is more robust than
either alone. A sensible build order for a new system is to start with
TF-IDF as a baseline, measure retrieval quality with an evaluation set, and
introduce embeddings only if the measurements show vocabulary mismatch is
actually costing accuracy.

Cosine similarity, used by both families, measures the angle between two
vectors rather than their length. Two texts pointing the same direction in
vector space score near 1.0 regardless of how long they are, which is why
it is the standard choice for comparing documents of different sizes.
