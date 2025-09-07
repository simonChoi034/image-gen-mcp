from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from pydantic import BaseModel

from ..schema import (
    CapabilityReport,
    ImageEditRequest,
    ImageGenerateRequest,
    ImageResponse,
)
from ..shard.enums import Provider
from ..utils.image_utils import read_image_bytes_and_mime as _read_image_bytes_and_mime
from ..utils.image_utils import to_image_data_url as _to_image_data_url


class ImageEngine(ABC, BaseModel):
    """Abstract base for image engines."""

    name: str
    provider: Provider

    def __init__(self, provider: Provider, **data):
        """Initialize engine with provider."""
        super().__init__(provider=provider, **data)

    # ------------------------------------------------------------------
    # Shared image source utilities
    # ------------------------------------------------------------------
    def read_image_bytes_and_mime(self, source: str, *, validate: bool = True) -> tuple[bytes, str]:
        return _read_image_bytes_and_mime(source, validate=validate)

    def to_image_data_url(self, source: str, *, validate_remote: bool = False) -> str:
        return _to_image_data_url(source, validate_remote=validate_remote)

    @abstractmethod
    async def generate(self, req: ImageGenerateRequest) -> ImageResponse:
        raise NotImplementedError

    @abstractmethod
    async def edit(self, req: ImageEditRequest) -> ImageResponse:
        raise NotImplementedError

    @abstractmethod
    def get_capability_report(self) -> CapabilityReport:
        """Return the capability report for this engine.

        Each engine implementation should return its specific capabilities
        including supported models, output formats, sizes, and feature flags.
        """
        raise NotImplementedError


class EngineFactory(Protocol):
    def __call__(self) -> ImageEngine:  # pragma: no cover - interface
        ...
