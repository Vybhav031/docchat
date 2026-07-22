"""Provider abstraction tests. Every network boundary is mocked, so these
run offline and free: the Anthropic SDK client is replaced with a fake,
and Ollama's HTTP call is intercepted at urllib level."""
import io
import json
from types import SimpleNamespace

import pytest

from app import llm
from app.config import settings
from app.retrieval import Hit

HITS = [
    Hit(chunk_id=1, document_name="guide.md", position=0,
        text="TF-IDF weighs rare terms more heavily than common ones.", score=0.62),
    Hit(chunk_id=2, document_name="guide.md", position=3,
        text="Cosine similarity compares the angle between two vectors.", score=0.41),
]


@pytest.fixture()
def provider(monkeypatch):
    """Set the selected provider for one test: provider('ollama')."""
    def set_provider(name, api_key=""):
        monkeypatch.setattr(settings, "llm_provider", name)
        monkeypatch.setattr(settings, "anthropic_api_key", api_key)
    return set_provider


# ---------------------------------------------------------------- selection

def test_no_hits_short_circuits(provider):
    provider("ollama")
    text, mode = llm.answer("anything", [])
    assert mode == "no-context"


def test_provider_none_uses_retrieval_only(provider):
    provider("none")
    text, mode = llm.answer("what is tf-idf", HITS)
    assert mode == "retrieval-only"
    assert "TF-IDF weighs rare terms" in text


def test_anthropic_without_key_uses_retrieval_only(provider):
    provider("anthropic", api_key="")
    text, mode = llm.answer("what is tf-idf", HITS)
    assert mode == "retrieval-only"


def test_unknown_provider_uses_retrieval_only(provider):
    provider("banana")
    text, mode = llm.answer("what is tf-idf", HITS)
    assert mode == "retrieval-only"


# ---------------------------------------------------------------- anthropic

def test_anthropic_generates_with_grounding_prompt(provider, monkeypatch):
    provider("anthropic", api_key="sk-test")
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="TF-IDF weighs rare terms [1].")]
            )

    class FakeClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)

    text, mode = llm.answer("what is tf-idf", HITS)
    assert mode == "generated"
    assert text == "TF-IDF weighs rare terms [1]."
    assert captured["system"] == llm.SYSTEM_PROMPT
    assert "[1] (from guide.md)" in captured["messages"][0]["content"]


def test_anthropic_api_error_degrades_gracefully(provider, monkeypatch):
    provider("anthropic", api_key="sk-test")
    import anthropic

    class FailingClient:
        def __init__(self, api_key):
            self.messages = SimpleNamespace(
                create=lambda **kw: (_ for _ in ()).throw(anthropic.AnthropicError("boom"))
            )

    monkeypatch.setattr(anthropic, "Anthropic", FailingClient)

    text, mode = llm.answer("what is tf-idf", HITS)
    assert mode == "retrieval-only"
    assert "could not be reached" in text
    assert "TF-IDF weighs rare terms" in text  # passages still shown


# ---------------------------------------------------------------- ollama

def _fake_urlopen(response_body, captured):
    def fake(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        class Resp:
            def __enter__(self):
                return io.BytesIO(json.dumps(response_body).encode("utf-8"))
            def __exit__(self, *args):
                return False
        return Resp()
    return fake


def test_ollama_generates_with_grounding_prompt(provider, monkeypatch):
    provider("ollama")
    monkeypatch.setattr(settings, "ollama_model", "llama3")
    captured = {}
    body = {"message": {"role": "assistant", "content": "Cosine compares angles [2]."}}
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(body, captured))

    text, mode = llm.answer("what does cosine similarity do", HITS)
    assert mode == "generated"
    assert text == "Cosine compares angles [2]."
    assert captured["url"] == llm.OLLAMA_URL
    assert captured["payload"]["model"] == "llama3"
    assert captured["payload"]["stream"] is False
    # The same grounding system prompt applies to every provider.
    assert captured["payload"]["messages"][0] == {"role": "system", "content": llm.SYSTEM_PROMPT}
    assert "[2] (from guide.md)" in captured["payload"]["messages"][1]["content"]


def test_ollama_unreachable_degrades_gracefully(provider, monkeypatch):
    provider("ollama")
    import urllib.error

    def refuse(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)

    text, mode = llm.answer("what is tf-idf", HITS)
    assert mode == "retrieval-only"
    assert "could not be reached" in text
    assert "TF-IDF weighs rare terms" in text


def test_ollama_malformed_response_degrades_gracefully(provider, monkeypatch):
    provider("ollama")
    monkeypatch.setattr("urllib.request.urlopen",
                        _fake_urlopen({"error": "model not found"}, {}))

    text, mode = llm.answer("what is tf-idf", HITS)
    assert mode == "retrieval-only"
    assert "TF-IDF weighs rare terms" in text


# ---------------------------------------------------------------- groq

def test_groq_without_key_uses_retrieval_only(provider, monkeypatch):
    provider("groq")
    monkeypatch.setattr(settings, "groq_api_key", "")
    text, mode = llm.answer("what is tf-idf", HITS)
    assert mode == "retrieval-only"


def test_groq_generates_with_grounding_prompt(provider, monkeypatch):
    provider("groq")
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    monkeypatch.setattr(settings, "groq_model", "llama-3.3-70b-versatile")
    captured = {}
    body = {"choices": [{"message": {"role": "assistant",
                                     "content": "TF-IDF weighs rare terms [1]."}}]}
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(body, captured))

    text, mode = llm.answer("what is tf-idf", HITS)
    assert mode == "generated"
    assert text == "TF-IDF weighs rare terms [1]."
    assert captured["url"] == llm.GROQ_URL
    assert captured["headers"]["Authorization"] == "Bearer gsk-test"
    assert captured["payload"]["model"] == "llama-3.3-70b-versatile"
    # The same grounding system prompt applies to every provider.
    assert captured["payload"]["messages"][0] == {"role": "system", "content": llm.SYSTEM_PROMPT}
    assert "[1] (from guide.md)" in captured["payload"]["messages"][1]["content"]


def test_groq_unreachable_degrades_gracefully(provider, monkeypatch):
    provider("groq")
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    import urllib.error

    def refuse(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)

    text, mode = llm.answer("what is tf-idf", HITS)
    assert mode == "retrieval-only"
    assert "could not be reached" in text
    assert "TF-IDF weighs rare terms" in text


def test_groq_malformed_response_degrades_gracefully(provider, monkeypatch):
    provider("groq")
    monkeypatch.setattr(settings, "groq_api_key", "gsk-test")
    monkeypatch.setattr("urllib.request.urlopen",
                        _fake_urlopen({"error": {"message": "invalid api key"}}, {}))

    text, mode = llm.answer("what is tf-idf", HITS)
    assert mode == "retrieval-only"
    assert "TF-IDF weighs rare terms" in text
