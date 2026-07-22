"""End-to-end API tests. The app runs against a temporary SQLite file and
without an API key, so answers use retrieval-only mode — which is exactly
what we want in CI: deterministic and free."""
import io
import os

os.environ["DATABASE_URL"] = "sqlite:///./test_docchat.db"
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["LLM_PROVIDER"] = "anthropic"
os.environ["ASK_RATE_LIMIT"] = ""

import pytest
from fastapi.testclient import TestClient

from app.main import app


from app.db import Base, engine


@pytest.fixture()
def client():
    # Reset schema (not the file): dropping tables through the same engine
    # avoids stale pooled connections to a deleted SQLite file.
    Base.metadata.drop_all(engine)
    with TestClient(app) as c:   # context manager triggers lifespan (seeds samples)
        yield c
    engine.dispose()
    if os.path.exists("test_docchat.db"):
        os.remove("test_docchat.db")


def test_startup_seeds_sample_corpus(client):
    stats = client.get("/api/stats").json()
    assert stats["documents"] == 3
    assert stats["chunks"] > 0
    assert stats["llm_enabled"] is False
    assert stats["llm_provider"] == "anthropic"


def test_ask_returns_grounded_sources(client):
    res = client.post("/api/ask", json={"question": "How does TF-IDF ranking work?"})
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "retrieval-only"
    assert body["sources"], "expected at least one source"
    assert body["sources"][0]["document_name"] == "02_retrieval_methods.md"
    assert body["sources"][0]["score"] > 0


def test_ask_validates_input(client):
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/api/ask", json={}).status_code == 422


def test_upload_ask_delete_roundtrip(client):
    content = ("The zorblatt festival happens every June in Riverton.\n\n"
               "Tickets for the zorblatt festival cost twelve dollars.")
    res = client.post("/api/documents",
                      files={"file": ("festival.txt", io.BytesIO(content.encode()), "text/plain")})
    assert res.status_code == 201
    doc = res.json()
    assert doc["chunk_count"] >= 1

    body = client.post("/api/ask", json={"question": "when is the zorblatt festival"}).json()
    assert any(s["document_name"] == "festival.txt" for s in body["sources"])

    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 404


def test_upload_rejects_bad_files(client):
    r = client.post("/api/documents",
                    files={"file": ("evil.exe", io.BytesIO(b"binary"), "application/x-msdownload")})
    assert r.status_code == 400

    r = client.post("/api/documents",
                    files={"file": ("dup.txt", io.BytesIO(b"hello world"), "text/plain")})
    assert r.status_code == 201
    r = client.post("/api/documents",
                    files={"file": ("dup.txt", io.BytesIO(b"hello world"), "text/plain")})
    assert r.status_code == 409


def make_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF containing `text`, without extra dependencies.
    `text` must not contain parentheses or backslashes (PDF string delimiters)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF"
            % (len(objects) + 1, xref_pos))
    return bytes(out)


def test_upload_pdf_and_ask(client):
    pdf = make_pdf("The crimson lighthouse in Port Vela was built in 1911 by the Maritime Guild.")
    res = client.post("/api/documents",
                      files={"file": ("lighthouse.pdf", io.BytesIO(pdf), "application/pdf")})
    assert res.status_code == 201
    doc = res.json()
    assert doc["name"] == "lighthouse.pdf"
    assert doc["chunk_count"] >= 1

    body = client.post("/api/ask",
                       json={"question": "who built the crimson lighthouse"}).json()
    assert any(s["document_name"] == "lighthouse.pdf" for s in body["sources"])

    # A PDF that isn't really a PDF is rejected, not ingested as garbage.
    r = client.post("/api/documents",
                    files={"file": ("fake.pdf", io.BytesIO(b"not a pdf"), "application/pdf")})
    assert r.status_code == 400


def test_ask_rate_limit_enforced(client, monkeypatch):
    from app import main as app_main
    from app.config import settings

    monkeypatch.setattr(settings, "ask_rate_limit", 2)
    app_main._ask_times.clear()
    q = {"question": "How does TF-IDF ranking work?"}

    assert client.post("/api/ask", json=q).status_code == 200
    assert client.post("/api/ask", json=q).status_code == 200
    r = client.post("/api/ask", json=q)
    assert r.status_code == 429
    assert "2 questions per hour" in r.json()["detail"]
    app_main._ask_times.clear()


def test_ask_rate_limit_window_expires(client, monkeypatch):
    from app import main as app_main
    from app.config import settings

    monkeypatch.setattr(settings, "ask_rate_limit", 1)
    app_main._ask_times.clear()
    q = {"question": "How does TF-IDF ranking work?"}

    assert client.post("/api/ask", json=q).status_code == 200
    assert client.post("/api/ask", json=q).status_code == 429
    # Age the recorded request past the one-hour window: the IP may ask again.
    for times in app_main._ask_times.values():
        for i in range(len(times)):
            times[i] -= app_main.RATE_WINDOW_SECONDS + 1
    assert client.post("/api/ask", json=q).status_code == 200
    app_main._ask_times.clear()


def test_ask_rate_limit_disabled_by_default(client):
    q = {"question": "How does TF-IDF ranking work?"}
    for _ in range(5):
        assert client.post("/api/ask", json=q).status_code == 200


def test_history_records_exchanges(client):
    client.post("/api/ask", json={"question": "What is cosine similarity used for?"})
    history = client.get("/api/history").json()
    assert history and history[0]["question"] == "What is cosine similarity used for?"
