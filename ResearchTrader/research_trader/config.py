"""
Configuration settings for the ResearchTrader application
Uses Pydantic BaseSettings for environment variables handling
"""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API URLs
    ARXIV_API_URL: str = "https://export.arxiv.org/api/query"

    # API Keys
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(..., env="OPENAI_MODEL")

    # Caching settings
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 3600  # Cache time-to-live in seconds
    REDIS_URL: str | None = None

    # Rate limiting
    ARXIV_RATE_LIMIT: int = 60  # Requests per minute
    OPENAI_RATE_LIMIT: int = 100  # Tokens per minute (adjust based on your OpenAI plan)

    # Timeouts
    ARXIV_TIMEOUT: float = 10.0  # Seconds
    OPENAI_TIMEOUT: float = 60.0  # Seconds

    # Application settings
    MAX_PAPERS: int = Field(20, ge=1, le=50)
    DEFAULT_PAPERS: int = Field(10, ge=1, le=20)

    # Logging
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = True
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Export settings instance
settings = get_settings()
