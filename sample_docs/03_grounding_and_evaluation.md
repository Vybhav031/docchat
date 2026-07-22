# Grounding, hallucination, and evaluation

A language model hallucinates when it states something false with
confidence. In a RAG system the main defence is the grounding instruction:
the model is told to answer only from the retrieved passages, to cite the
passage numbers supporting each claim, and to say plainly when the passages
do not contain the answer. Showing the retrieved passages to the user
alongside the answer adds a second defence, because claims become checkable
against their sources.

Retrieval quality bounds answer quality. If the right passage is not
retrieved, no prompt can produce a correct grounded answer. This is why
RAG systems are evaluated at the retrieval stage first, before judging the
generated text.

The standard retrieval metric is hit rate at k, written hit@k: over a set
of test questions with known correct source documents, the fraction of
questions where a correct document appears in the top k retrieved chunks.
A related metric, mean reciprocal rank (MRR), also rewards the correct
document appearing earlier in the ranking. An evaluation set of even ten to
twenty questions catches most regressions: it turns "the retrieval feels
worse after this change" into a number that can be compared before and
after.

Answer-stage evaluation is harder because judging free text is subjective.
Common approaches include checking that generated answers contain expected
key phrases, and using a second model as a grader. For a small system, the
practical combination is automated hit@k for retrieval plus a short manual
review of generated answers on the same test questions.
