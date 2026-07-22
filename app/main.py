"""DocChat — ask questions about your documents, with the evidence shown.

Endpoints:
    GET  /                     the web interface
    GET  /api/stats            corpus size + whether generation is enabled
    GET  /api/documents        list documents
    POST /api/documents        upload a .txt, .md, or .pdf file
    DELETE /api/documents/{id} remove a document (re-indexes)
    POST /api/ask              ask a question -> answer + cited sources
    GET  /api/history          past question/answer exchanges
"""
import io
import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import llm
from .chunking import chunk_text
from .config import settings
from .db import Chunk, Document, Exchange, get_session, init_db, SessionLocal
from .retrieval import retriever
from .schemas import AskRequest, AskResponse, DocumentOut, Source, StatsOut

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    session = SessionLocal()
    try:
        # Seed the sample corpus on first run so the demo works immediately.
        if session.scalar(select(func.count()).select_from(Document)) == 0 and SAMPLE_DIR.exists():
            for path in sorted(SAMPLE_DIR.glob("*.md")):
                ingest_document(session, path.name, path.read_text(encoding="utf-8"))
        rebuild_index(session)
    finally:
        session.close()
    yield


app = FastAPI(title="DocChat", version="1.0.0", lifespan=lifespan)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"
ALLOWED_SUFFIXES = {".txt", ".md", ".pdf"}


# ---------------------------------------------------------------- indexing

def rebuild_index(session: Session) -> None:
    rows = session.execute(
        select(Chunk.id, Document.name, Chunk.position, Chunk.text).join(Document)
    ).all()
    retriever.build(
        [
            {"chunk_id": r[0], "document_name": r[1], "position": r[2], "text": r[3]}
            for r in rows
        ]
    )


def extract_pdf_text(raw: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError:
        raise HTTPException(status_code=400, detail="Could not read the PDF file.")
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise HTTPException(status_code=400, detail="The PDF contains no extractable text.")
    return text


def ingest_document(session: Session, name: str, text: str) -> Document:
    if not text.strip():
        raise HTTPException(status_code=400, detail="The file is empty.")
    existing = session.scalar(select(Document).where(Document.name == name))
    if existing:
        raise HTTPException(status_code=409, detail=f"A document named '{name}' already exists.")

    doc = Document(name=name)
    session.add(doc)
    session.flush()
    for i, piece in enumerate(chunk_text(text, settings.chunk_size, settings.chunk_overlap)):
        session.add(Chunk(document_id=doc.id, position=i, text=piece))
    session.commit()
    rebuild_index(session)
    return doc


# ---------------------------------------------------------------- rate limiting

RATE_WINDOW_SECONDS = 3600
_ask_times: dict[str, deque[float]] = defaultdict(deque)  # ip -> request timestamps


def enforce_ask_rate_limit(request: Request) -> None:
    """Per-IP hourly cap on questions, in memory (single worker — see README).
    Disabled unless ASK_RATE_LIMIT is set."""
    limit = settings.ask_rate_limit
    if not limit:
        return
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    times = _ask_times[ip]
    while times and now - times[0] >= RATE_WINDOW_SECONDS:
        times.popleft()
    if len(times) >= limit:
        raise HTTPException(
            status_code=429,
            detail=(f"You've reached the limit of {limit} questions per hour. "
                    "Please try again a little later."),
        )
    times.append(now)


# ---------------------------------------------------------------- routes

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/stats", response_model=StatsOut)
def stats(session: Session = Depends(get_session)) -> StatsOut:
    return StatsOut(
        documents=session.scalar(select(func.count()).select_from(Document)) or 0,
        chunks=session.scalar(select(func.count()).select_from(Chunk)) or 0,
        llm_enabled=settings.llm_enabled,
        llm_provider=settings.llm_provider,
    )


@app.get("/api/documents", response_model=list[DocumentOut])
def list_documents(session: Session = Depends(get_session)) -> list[DocumentOut]:
    docs = session.scalars(select(Document).order_by(Document.name)).all()
    return [DocumentOut(id=d.id, name=d.name, chunk_count=len(d.chunks)) for d in docs]


@app.post("/api/documents", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile, session: Session = Depends(get_session)
) -> DocumentOut:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only .txt, .md, and .pdf files are supported.")
    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=400, detail="File is larger than 2 MB.")
    if suffix == ".pdf":
        text = extract_pdf_text(raw)
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be UTF-8 text.")

    doc = ingest_document(session, file.filename, text)
    return DocumentOut(id=doc.id, name=doc.name, chunk_count=len(doc.chunks))


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: int, session: Session = Depends(get_session)) -> None:
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    session.delete(doc)
    session.commit()
    rebuild_index(session)


@app.post("/api/ask", response_model=AskResponse, dependencies=[Depends(enforce_ask_rate_limit)])
def ask(payload: AskRequest, session: Session = Depends(get_session)) -> AskResponse:
    if retriever.size == 0:
        raise HTTPException(status_code=400, detail="Upload a document before asking questions.")

    hits = retriever.search(payload.question, top_k=settings.top_k)
    answer_text, mode = llm.answer(payload.question, hits)
    sources = [
        Source(
            chunk_id=h.chunk_id,
            document_name=h.document_name,
            position=h.position,
            text=h.text,
            score=h.score,
        )
        for h in hits
    ]

    session.add(
        Exchange(
            question=payload.question,
            answer=answer_text,
            sources_json=json.dumps([s.model_dump() for s in sources]),
            mode=mode,
        )
    )
    session.commit()
    return AskResponse(question=payload.question, answer=answer_text, mode=mode, sources=sources)


@app.get("/api/history", response_model=list[AskResponse])
def history(limit: int = 20, session: Session = Depends(get_session)) -> list[AskResponse]:
    rows = session.scalars(
        select(Exchange).order_by(Exchange.created_at.desc()).limit(min(limit, 100))
    ).all()
    return [
        AskResponse(
            question=r.question,
            answer=r.answer,
            mode=r.mode,
            sources=[Source(**s) for s in json.loads(r.sources_json)],
        )
        for r in rows
    ]
