# Retrieval-Augmented Generation (RAG)

Retrieval-augmented generation, usually shortened to RAG, is a pattern for
answering questions with a large language model while keeping the answers
grounded in a specific set of documents. Instead of asking the model to
answer from its training data alone, a RAG system first searches a document
collection for passages relevant to the question, then hands those passages
to the model as context and instructs it to answer using only that context.

The pattern has three stages. Ingestion: documents are split into chunks and
indexed so they can be searched. Retrieval: when a question arrives, the
system ranks all chunks by relevance and selects the top few. Generation:
the language model composes an answer from the retrieved chunks, citing
which passages support each claim.

RAG systems hallucinate less than bare language models because the
generation step is constrained. The prompt tells the model to use only the
provided passages and to say when the passages do not contain the answer.
A bare model asked about something outside its knowledge will often produce
a fluent but invented response; a grounded model has both the relevant text
in front of it and explicit permission to decline.

Chunking matters more than it first appears. Chunks that are too small lose
the surrounding context an answer needs; chunks that are too large dilute
the relevance signal and waste the model's context window. Splitting on
paragraph boundaries with a modest overlap between consecutive chunks is a
strong default, because paragraphs tend to carry one idea each.
