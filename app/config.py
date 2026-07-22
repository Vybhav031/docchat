"""Central configuration, read once from environment variables."""
import os


class Settings:
    def __init__(self) -> None:
        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.database_url: str = os.getenv("DATABASE_URL", "sqlite:///./docchat.db")
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
        self.top_k: int = int(os.getenv("TOP_K", "4"))
        self.max_upload_bytes: int = 2 * 1024 * 1024  # 2 MB
        self.model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        # anthropic | ollama | groq | none — which backend composes answers.
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
        self.ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3").strip()
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        # Max POST /api/ask requests per IP per hour; unset/empty disables limiting.
        raw_limit = os.getenv("ASK_RATE_LIMIT", "").strip()
        self.ask_rate_limit: int | None = int(raw_limit) if raw_limit else None

    @property
    def llm_enabled(self) -> bool:
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        if self.llm_provider == "groq":
            return bool(self.groq_api_key)
        return self.llm_provider == "ollama"


settings = Settings()
