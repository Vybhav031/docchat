from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Source(BaseModel):
    chunk_id: int
    document_name: str
    position: int
    text: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: str
    sources: list[Source]


class DocumentOut(BaseModel):
    id: int
    name: str
    chunk_count: int


class StatsOut(BaseModel):
    documents: int
    chunks: int
    llm_enabled: bool
    llm_provider: str
