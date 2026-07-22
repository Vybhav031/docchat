"""Turn retrieved chunks into an answer.

Providers are pluggable behind LLMProvider (selected via LLM_PROVIDER):
- anthropic:      the Anthropic API composes a grounded answer.
- ollama:         a local model served by Ollama does the same, offline.
- groq:           Groq's OpenAI-compatible API, for fast hosted open models.
- none / neither configured: retrieval-only mode returns the strongest
                  passages verbatim so the app remains fully demoable.
Every provider gets the same grounding system prompt: the model may
only use the provided context, and must say so when the context does
not contain the answer. That single instruction is the main defence
against hallucinated answers in a RAG system. If the selected provider
fails at answer time, we degrade to retrieval-only instead of erroring.
"""
from .config import settings
from .retrieval import Hit

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite passages inline like [1] or [2] after the claims they support. "
    "If the context does not contain the answer, say exactly that and do not guess. "
    "Be concise: a short paragraph, or a few bullet points if the question asks for a list."
)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT_SECONDS = 120  # local models on modest hardware can be slow
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT_SECONDS = 60


class ProviderError(Exception):
    """The provider was selected and configured but could not produce an answer."""


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: int) -> dict:
    """POST JSON, return the decoded JSON response. Raises ProviderError on failure."""
    import json
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProviderError(f"Request to {url} failed: {exc}") from exc


def _build_user_message(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] (from {h.document_name})\n{h.text}" for i, h in enumerate(hits)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


class LLMProvider:
    """One answer backend. Subclasses set `name` and implement the two methods."""

    name: str

    def is_configured(self) -> bool:
        """Cheap static check (keys etc.) — no network calls."""
        raise NotImplementedError

    def generate(self, question: str, hits: list[Hit]) -> str:
        """Compose a grounded answer. Raises ProviderError on any failure."""
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def is_configured(self) -> bool:
        return bool(settings.anthropic_api_key)

    def generate(self, question: str, hits: list[Hit]) -> str:
        import anthropic  # imported lazily so other providers never need it installed

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        try:
            response = client.messages.create(
                model=settings.model,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _build_user_message(question, hits)}],
            )
        except anthropic.AnthropicError as exc:  # bad key, rate limit, network trouble
            raise ProviderError(str(exc)) from exc
        return "".join(block.text for block in response.content if block.type == "text").strip()


class OllamaProvider(LLMProvider):
    name = "ollama"

    def is_configured(self) -> bool:
        return True  # no key needed; reachability is only knowable at call time

    def generate(self, question: str, hits: list[Hit]) -> str:
        payload = {
            "model": settings.ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(question, hits)},
            ],
        }
        body = _post_json(OLLAMA_URL, payload, {}, OLLAMA_TIMEOUT_SECONDS)
        try:
            return str(body["message"]["content"]).strip()
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"Unexpected response from Ollama: {body!r}") from exc


class GroqProvider(LLMProvider):
    name = "groq"

    def is_configured(self) -> bool:
        return bool(settings.groq_api_key)

    def generate(self, question: str, hits: list[Hit]) -> str:
        payload = {
            "model": settings.groq_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(question, hits)},
            ],
        }
        headers = {"Authorization": f"Bearer {settings.groq_api_key}"}
        body = _post_json(GROQ_URL, payload, headers, GROQ_TIMEOUT_SECONDS)
        try:
            return str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected response from Groq: {body!r}") from exc


PROVIDERS: dict[str, LLMProvider] = {
    AnthropicProvider.name: AnthropicProvider(),
    OllamaProvider.name: OllamaProvider(),
    GroqProvider.name: GroqProvider(),
}


def get_provider() -> LLMProvider | None:
    """The selected, configured provider — or None for retrieval-only mode."""
    provider = PROVIDERS.get(settings.llm_provider)
    if provider is not None and provider.is_configured():
        return provider
    return None


def _retrieval_only_answer(hits: list[Hit]) -> str:
    lines = ["Top matching passages (retrieval-only mode — configure an LLM provider for a composed answer):", ""]
    for i, h in enumerate(hits, start=1):
        snippet = h.text if len(h.text) <= 400 else h.text[:400].rsplit(" ", 1)[0] + "…"
        lines.append(f"[{i}] {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def answer(question: str, hits: list[Hit]) -> tuple[str, str]:
    """Returns (answer_text, mode)."""
    if not hits:
        return (
            "None of the uploaded documents mention this. Try rephrasing, or upload a document that covers it.",
            "no-context",
        )
    provider = get_provider()
    if provider is not None:
        try:
            return provider.generate(question, hits), "generated"
        except ProviderError:
            # Degrade to the passages rather than surfacing a 500 to the user.
            note = (f"(The {provider.name} language model could not be reached, "
                    "so here are the strongest passages instead.)\n\n")
            return note + _retrieval_only_answer(hits), "retrieval-only"
    return _retrieval_only_answer(hits), "retrieval-only"
