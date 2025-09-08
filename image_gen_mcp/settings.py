from __future__ import annotations

from functools import lru_cache
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config: ClassVar[dict] = {"env_nested_delimiter": "__"}

    openai_api_key: str | None = Field(default=None, description="API key for OpenAI")

    azure_openai_key: str | None = Field(default=None, description="API key for Azure OpenAI")
    azure_openai_endpoint: str | None = Field(default=None, description="Endpoint for Azure OpenAI")
    azure_openai_api_version: str | None = Field(default="2024-02-15-preview", description="API version for Azure OpenAI")

    gemini_api_key: str | None = Field(default=None, description="API key for Google Gemini")

    use_vertex_ai: bool = Field(default=False, description="If true, prefer Google Vertex AI when available")
    vertex_project: str | None = Field(default=None, description="Project ID for Google Vertex AI")
    vertex_location: str | None = Field(default=None, description="Location for Google Vertex AI")
    vertex_credentials_path: str | None = Field(default=None, description="Path to Google Cloud credentials JSON file")

    openrouter_api_key: str | None = Field(default=None, description="API key for OpenRouter")

    @property
    def use_openai(self) -> bool:
        """Determine if OpenAI should be used based on available credentials."""
        return bool(self.openai_api_key)

    @property
    def use_azure_openai(self) -> bool:
        """Determine if Azure OpenAI should be used based on available credentials."""
        return bool(self.azure_openai_key and self.azure_openai_endpoint)

    @property
    def use_gemini(self) -> bool:
        """Determine if Gemini should be used based on available credentials."""
        return bool(self.gemini_api_key)

    @property
    def use_openrouter(self) -> bool:
        """Determine if OpenRouter should be used based on available credentials."""
        return bool(self.openrouter_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
