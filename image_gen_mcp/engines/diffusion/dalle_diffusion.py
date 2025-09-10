from __future__ import annotations

from enum import StrEnum
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI

from ...schema import (
    CapabilityReport,
    EmbeddedResource,
    Error,
    ImageEditRequest,
    ImageGenerateRequest,
    ImageResponse,
    ModelCapability,
    ResourceContent,
)
from ...settings import get_settings
from ...shard import constants as C
from ...shard.enums import (
    Family,
    Model,
    Orientation,
    Provider,
)
from ...utils.error_helpers import augment_with_capability_tip
from ..base_engine import ImageEngine

settings = get_settings()
# DALL·E 3 enums/constants


class DalleSize(StrEnum):
    """DALL·E 3 supported size mappings."""

    SQUARE = "1024x1024"
    PORTRAIT = "1024x1792"
    LANDSCAPE = "1792x1024"

    @classmethod
    def from_orientation(cls, orientation: Orientation | None) -> str:
        """Map unified orientation to DALL·E 3 native size."""
        o = orientation or Orientation.SQUARE
        if o == Orientation.PORTRAIT:
            return cls.PORTRAIT.value
        if o == Orientation.LANDSCAPE:
            return cls.LANDSCAPE.value
        return cls.SQUARE.value


class DalleResponseFormat(StrEnum):
    """DALL·E 3 API response format options."""

    B64_JSON = "b64_json"
    URL = "url"


class DalleErrorType(StrEnum):
    """DALL·E specific error types."""

    NO_IMAGES = "no_images_generated"
    EDIT_NOT_SUPPORTED = "edit_not_supported"


# Default API version for Azure OpenAI
API_VERSION_DEFAULT = "2025-04-01-preview"


class DalleDiffusion(ImageEngine):
    """Unified DALL·E 3 adapter for OpenAI/Azure.

    Provides generation-only support and unified parameter handling.
    """

    def __init__(self, provider: Provider) -> None:
        super().__init__(provider=provider, name=f"diffusion:{provider.value}")

    # Capability discovery
    def get_capability_report(self) -> CapabilityReport:
        """Return capability report for this engine's provider."""
        # Single model; advertise shared parameters at report level.
        return CapabilityReport(
            provider=self.provider,
            family=Family.DIFFUSION,
            models=[
                ModelCapability(
                    model=Model.DALL_E_3,
                    supports_negative_prompt=False,
                    supports_background=False,
                    max_n=1,
                    supports_edit=False,
                    supports_mask=False,
                )
            ],
        )

    # Client management
    def _select_provider(self, req_provider: Provider | None) -> Provider:
        """Select provider - prefer request provider if compatible, otherwise use engine's provider."""
        if req_provider and req_provider == self.provider:
            return req_provider
        return self.provider

    def _openai_client(self) -> AsyncOpenAI:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set to use OpenAI provider")
        return AsyncOpenAI(api_key=settings.openai_api_key)

    def _azure_client(self) -> AsyncAzureOpenAI:
        if not settings.azure_openai_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable must be set to use Azure OpenAI")
        if not settings.azure_openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable must be set to use Azure OpenAI")
        return AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version or API_VERSION_DEFAULT,
        )

    def _get_client(self) -> AsyncAzureOpenAI | AsyncOpenAI:
        if self.provider == Provider.AZURE_OPENAI:
            return self._azure_client()
        return self._openai_client()

    # Parameter normalization
    def _normalize_count(self, req_n: int | None) -> tuple[int, dict[str, Any]]:
        """Normalize count parameter - DALL·E 3 only supports n=1."""
        normlog = {}
        n = 1  # DALL·E 3 limitation
        if req_n and req_n != 1:
            normlog["clamped_n"] = 1
            normlog["original_n"] = req_n
        else:
            normlog["n"] = 1
        return n, normlog

    def _normalize_size(self, size: str | None, orientation: Orientation | None) -> tuple[str, dict[str, Any]]:
        """Normalize size strictly from orientation (no raw overrides)."""
        normlog: dict[str, Any] = {}
        native_size = DalleSize.from_orientation(orientation)
        normlog["size"] = native_size
        return native_size, normlog

    def _normalize_output_format(self, provider: Provider) -> tuple[str, dict[str, Any]]:
        """Normalize output format preference based on provider.

        This server no longer exposes a user-selectable output_format; adapters
        always request base64 (`b64_json`) to enable embedding resources.
        """
        normlog: dict[str, Any] = {}
        # Prefer base64 for all providers to maintain embedded resource behavior.
        resp_format = DalleResponseFormat.B64_JSON.value
        if provider == Provider.AZURE_OPENAI:
            # Note for observability: Azure typically returns base64 only.
            normlog["forced_base64"] = "Azure OpenAI typically returns base64"
        normlog["response_format"] = resp_format
        return resp_format, normlog

    # Error handling
    def _create_provider_error(self, model: Model, provider: Provider, normlog: dict[str, Any], exception: Exception) -> ImageResponse:
        """Create error response for provider API failures."""

        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=C.ERROR_CODE_PROVIDER_ERROR, message=augment_with_capability_tip(str(exception))),
        )

    def _create_no_images_error(self, model: Model, provider: Provider, normlog: dict[str, Any]) -> ImageResponse:
        """Create error response when no images are generated."""
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=DalleErrorType.NO_IMAGES.value, message="No image content in response"),
        )

    def _create_edit_not_supported_error(self, model: Model, provider: Provider) -> ImageResponse:
        """Create error response for unsupported edit operations."""
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=DalleErrorType.EDIT_NOT_SUPPORTED.value, message="DALL·E 3 edits are not supported; use GPT-Image-1."),
        )

    # Response processing
    def _process_image_data(self, result: Any, want_b64: bool) -> list[ResourceContent]:
        """Process API response data into embedded ResourceContent objects."""
        images: list[ResourceContent] = []
        data = getattr(result, "data", None) or []

        for item in data:
            if want_b64:
                b64 = getattr(item, "b64_json", None)
                if b64:
                    resource_id = f"gen-{len(images) + 1}"
                    embedded_resource = EmbeddedResource(
                        uri=f"image://{resource_id}",
                        name=f"image-{resource_id}.png",
                        mimeType=C.DEFAULT_MIME,
                        blob=b64,
                        description="Generated image with embedded data",
                    )
                    images.append(ResourceContent(type="resource", resource=embedded_resource))
            else:
                url = getattr(item, "url", None)
                if url:
                    # Prefer to avoid embedding remote URLs directly; fetch would be required to embed.
                    # Since response was requested as URL, we cannot embed without fetching; so skip.
                    # The caller should request base64 to receive embedded resources.
                    pass

        return images

    # API operations
    async def generate(self, req: ImageGenerateRequest) -> ImageResponse:  # type: ignore[override]
        """Generate images using DALL·E 3 via OpenAI or Azure OpenAI."""
        provider = self._select_provider(getattr(req, "provider", None))
        model = req.model

        # Normalize parameters
        n, n_normlog = self._normalize_count(req.n)
        size, size_normlog = self._normalize_size(None, req.orientation)
        response_format, format_normlog = self._normalize_output_format(provider)

        # Combine normalization logs
        normlog = {**n_normlog, **size_normlog, **format_normlog}
        want_b64 = response_format == DalleResponseFormat.B64_JSON.value

        try:
            client = self._get_client()

            # Call async image generation on the selected client
            result = await client.images.generate(
                model=model.value,
                prompt=req.prompt,
                size=size,  # type: ignore[arg-type]
                n=n,
                response_format=response_format,  # type: ignore[arg-type]
            )

            images = self._process_image_data(result, want_b64)
            if not images:
                return self._create_no_images_error(model, provider, normlog)

            return ImageResponse(ok=True, content=images, model=model)

        except Exception as e:  # pragma: no cover - provider failure path
            return self._create_provider_error(model, provider, normlog, e)

    # -------------------------------- edit ------------------------------- #
    async def edit(self, req: ImageEditRequest) -> ImageResponse:  # type: ignore[override]
        """DALL·E 3 does not support edit operations."""
        provider = self._select_provider(getattr(req, "provider", None))
        model = req.model or Model.DALL_E_3
        return self._create_edit_not_supported_error(model, provider)
