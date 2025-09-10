from __future__ import annotations

from enum import StrEnum
from typing import Self


class Provider(StrEnum):
    """Provider identifiers used for routing image requests.

    Values match the unified schema and env configuration keys. Keep names
    stable; member names may include legacy aliases, but their string values
    should be normalized to the public vocabulary.
    """

    OPENAI = "openai"
    OPENROUTER = "openrouter"
    # Normalized to 'azure' per schema; keep legacy alias name for internal use
    AZURE_OPENAI = "azure"
    GEMINI = "gemini"
    VERTEX = "vertex"

    @classmethod
    def from_str(cls, value: str | None) -> Self | None:
        if not value:
            return None
        v = value.strip().lower()
        try:
            return cls(v)  # type: ignore[arg-type]
        except Exception:
            return None

    def family(self) -> str | None:
        """Optional convenience: callers can set explicitly when known."""
        return None


class Family(StrEnum):
    """Engine family classifications used for routing and parameter validation.

    Values:
    - ``DIFFUSION``: diffusion-style image models (DALL·E, Imagen)
    - ``AR``: autoregressive / multimodal image models (gpt-image-1, Gemini)
    """

    DIFFUSION = "diffusion"
    AR = "ar"


class SizeCode(StrEnum):
    """Unified size classes (model-agnostic): S/M/L.

    Adapters convert these to native sizes or aspect + resolution per provider.
    """

    S = "S"
    M = "M"
    L = "L"


class Orientation(StrEnum):
    """Unified orientation preferences (model-agnostic)."""

    SQUARE = "square"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


class Quality(StrEnum):
    """Unified quality control (model-agnostic)."""

    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"


class Background(StrEnum):
    """Background alpha preference for AR engines that support it."""

    TRANSPARENT = "transparent"
    OPAQUE = "opaque"


class OutputFormat(StrEnum):
    """Supported output formats for image generation responses.

    This enum mirrors normalized constants used by adapters and request
    schemas (for example, 'url' and 'base64'). Using an enum improves
    type-safety across the codebase.
    """

    URL = "url"
    BASE64 = "base64"


class Model(StrEnum):
    """Curated model IDs allowed by this server.

    Values are provider-compatible strings. Keep the set intentionally small
    to simplify routing and validation.
    """

    # AR models
    GPT_IMAGE_1 = "gpt-image-1"  # OpenAI / Azure
    GEMINI_IMAGE_PREVIEW = "gemini-2.5-flash-image-preview"  # Gemini native; also permitted via Vertex engine
    OPENROUTER_GOOGLE_GEMINI_IMAGE = "google/gemini-2.5-flash-image-preview"  # OpenRouter

    # Diffusion models
    DALL_E_3 = "dall-e-3"  # OpenAI / Azure
    IMAGEN_4_STANDARD = "imagen-4.0-generate-001"  # Vertex
    IMAGEN_4_FAST = "imagen-4.0-fast-generate-001"  # Vertex
    IMAGEN_4_ULTRA = "imagen-4.0-ultra-generate-001"  # Vertex
    IMAGEN_3_GENERATE = "imagen-3.0-generate-002"  # Vertex
    IMAGEN_3_CAPABILITY = "imagen-3.0-capability-001"  # Vertex (supports editing)


# ---------------------- Provider-native helper enums ---------------------- #


class ImagenAspectRatio(StrEnum):
    """Native aspect ratio strings expected by Imagen (when supported)."""

    ONE_ONE = "1:1"
    THREE_FOUR = "3:4"
    FOUR_THREE = "4:3"


class ImagenResolutionHint(StrEnum):
    """Native resolution tier hints used by Imagen."""

    ONE_K = "1K"
    TWO_K = "2K"


__all__ = ["Provider", "Family", "OutputFormat", "Model", "SizeCode", "Orientation", "Quality", "Background", "ImagenAspectRatio", "ImagenResolutionHint"]
