"""Aplikasi configuration, read from .env / environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Face Attendance API"
    api_prefix: str = "/api"
    database_url: str = "postgresql+psycopg://face:face@localhost:5432/face_attendance"
    cors_origins: list[str] = ["http://localhost:3000"]
    face_match_threshold: float =0.35
    min_face_size: int =96
    blink_required: bool = True
    image_max_bytes: int =8 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
