"""Application configuration loaded from environment variables / .env."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")  # does not override variables already set


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_env: str
    database_url: str
    enable_code_execution: bool
    llm_provider: str
    llm_model: str
    llm_api_key: str = field(repr=False)  # never printed
    questions_dir: Path = PROJECT_ROOT / "data" / "questions"
    ml_data_dir: Path = PROJECT_ROOT / "data" / "ml"
    models_dir: Path = PROJECT_ROOT / "models"

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.strip())


def get_settings() -> Settings:
    """Read settings from the environment at call time."""
    default_db = (PROJECT_ROOT / "data" / "skillbridge.db").as_posix()
    database_url = os.getenv("DATABASE_URL", "").strip() or f"sqlite:///{default_db}"
    return Settings(
        app_env=os.getenv("APP_ENV", "development").strip() or "development",
        database_url=database_url,
        enable_code_execution=_env_bool("ENABLE_CODE_EXECUTION", True),
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic").strip().lower(),
        llm_model=os.getenv("LLM_MODEL", "").strip(),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
    )