"""Typed application configuration loaded from the environment."""

from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "data" / "profile.json"


class Settings(BaseSettings):
    """Validate runtime limits and keep configuration outside application code."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    model_name: str = Field(default="claude-sonnet-4-6", min_length=1)
    model_timeout_seconds: int = Field(default=30, gt=0)
    max_input_chars: int = Field(default=12_000, gt=0)
    max_history_messages: int = Field(default=12, ge=0)
    rate_limit_per_minute: int = Field(default=30, gt=0)
    environment: str = "development"
    log_level: str = "INFO"
    profile_path: Path = DEFAULT_PROFILE_PATH

    @model_validator(mode="after")
    def require_production_api_key(self) -> "Settings":
        """Reject a production configuration without its required provider credential."""
        if self.environment.lower() == "production" and self.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is required when ENVIRONMENT=production")
        return self
