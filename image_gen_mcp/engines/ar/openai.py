from __future__ import annotations

import base64
import contextlib
import urllib.request
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
    Background,
    Family,
    Model,
    Orientation,
    Provider,
    Quality,
)
from ...utils.error_helpers import augment_with_capability_tip
from ...utils.prompt import render_prompt_with_guidance
from ..base_engine import ImageEngine

settings = get_settings()
# Use grouped enums for related constant sets to improve discoverability

API_VERSION_DEFAULT = "2025-04-01-preview"


class GPTImage1Size(StrEnum):
    SQUARE = "1024x1024"
    PORTRAIT = "1024x1536"
    LANDSCAPE = "1536x1024"

    @classmethod
    def from_orientation(cls, orientation: Orientation | None) -> str:
        o = orientation or C.DEFAULT_ORIENTATION
        if o == Orientation.PORTRAIT:
            return cls.PORTRAIT.value
        if o == Orientation.LANDSCAPE:
            return cls.LANDSCAPE.value
        return cls.SQUARE.value


class GPTImage1Quality(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def from_quality(cls, quality: Quality | None) -> str | None:
        if quality is None:
            return None
        if quality == Quality.DRAFT:
            return cls.LOW.value
        if quality == Quality.HIGH:
            return cls.HIGH.value
        return cls.MEDIUM.value


class BackgroundValue(StrEnum):
    TRANSPARENT = "transparent"
    OPAQUE = "opaque"


def _is_url(value: str) -> bool:
    """Check if a string is an HTTP/HTTPS URL."""
    return value.startswith("http://") or value.startswith("https://")


def _read_image_bytes(source: str) -> bytes:
    """Read image bytes from a URL or base64 string.

    Accepts a data URL (data:image/png;base64,...) or raw base64.
    """
    if _is_url(source):
        with urllib.request.urlopen(source) as resp:  # nosec - trusted by caller
            return resp.read()
    # data URL
    if source.startswith("data:"):
        # format: data:<mime>;base64,<payload>
        try:
            b64 = source.split(",", 1)[1]
        except IndexError:
            raise ValueError("Invalid data URL for image")
        return base64.b64decode(b64)
    # assume raw base64
    return base64.b64decode(source)


class OpenAIAR(ImageEngine):
    """OpenAI/Azure AR adapter for GPT-Image-1.

    Provides unified parameter handling and client routing for OpenAI/Azure.
    Supports generation and editing with provider-specific routing and
    normalization of parameters such as size, quality, and background.
    """

    def __init__(self, provider: Provider) -> None:
        super().__init__(provider=provider, name=f"ar:{provider.value}")

    # Capability discovery
    def get_capability_report(self) -> CapabilityReport:
        """Return capability report for this engine's provider."""
        # Single model; parameters are shared, advertise them at report level.
        return CapabilityReport(
            provider=self.provider,
            family=Family.AR,
            models=[
                ModelCapability(
                    model=Model.GPT_IMAGE_1,
                    supports_negative_prompt=True,
                    supports_background=True,
                    max_n=4,
                    supports_edit=True,
                    supports_mask=True,
                    supports_multi_image_edit=True,
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
        """Normalize and clamp the count parameter."""
        normalization = {}
        n = int(req_n or C.DEFAULT_N)
        if n < 1:
            n = 1
        if n > C.MAX_N:
            n = C.MAX_N
            normalization["clamped_n"] = n
        return n, normalization

    def _normalize_size(self, orientation: Orientation | None) -> tuple[str, dict[str, Any]]:
        """Normalize size strictly from orientation (no raw overrides)."""
        normalization: dict[str, Any] = {}
        native_size = GPTImage1Size.from_orientation(orientation)
        normalization["size"] = native_size
        return native_size, normalization

    def _normalize_quality(self, quality: Quality | None) -> tuple[str | None, dict[str, Any]]:
        """Normalize quality parameter."""
        normalization = {}
        q_native = GPTImage1Quality.from_quality(quality)
        if q_native:
            normalization["quality"] = q_native
        return q_native, normalization

    def _normalize_background(self, background: Background | None) -> tuple[str | None, dict[str, Any], list[str]]:
        """Normalize background parameter and track dropped params."""
        normalization = {}
        dropped = []
        bg_native = None

        if background == Background.TRANSPARENT:
            bg_native = BackgroundValue.TRANSPARENT.value
            normalization["background"] = bg_native
        elif background is not None:
            # Opaque is default; record drop for observability
            dropped.append("background")

        return bg_native, normalization, dropped

    # Error handling

    def _create_provider_error(self, model: Model, provider: Provider, normlog: dict[str, Any], dropped: list[str], exception: Exception) -> ImageResponse:
        """Create error response for provider API failures."""
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=C.ERROR_CODE_PROVIDER_ERROR, message=augment_with_capability_tip(str(exception))),
        )

    def _create_provider_edit_error(self, model: Model, provider: Provider, normlog: dict[str, Any], dropped: list[str], exception: Exception) -> ImageResponse:
        """Create error response for provider API failures in edit operations."""

        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=C.ERROR_CODE_PROVIDER_ERROR, message=augment_with_capability_tip(str(exception))),
        )

    def _process_image_data(self, result: Any, response_format: str) -> list[ResourceContent]:
        """Process OpenAI image data and return ResourceContent objects."""
        images: list[ResourceContent] = []
        for idx, item in enumerate(result.data, start=1):
            if hasattr(item, "url") and item.url:
                # Convert URL to embedded resource if it's a data URL
                if item.url.startswith("data:"):
                    try:
                        _, b64_part = item.url.split(",", 1)
                        resource_id = f"gen-{idx}"
                        embedded_resource = EmbeddedResource(uri=f"image://{resource_id}", name=f"image-{resource_id}.png", mimeType=C.DEFAULT_MIME, blob=b64_part, description="Generated image with embedded data")
                        images.append(ResourceContent(type="resource", resource=embedded_resource))
                    except ValueError:
                        pass  # Skip malformed data URLs
            elif hasattr(item, "b64_json") and item.b64_json:
                resource_id = f"gen-{idx}"
                embedded_resource = EmbeddedResource(uri=f"image://{resource_id}", name=f"image-{resource_id}.png", mimeType=C.DEFAULT_MIME, blob=item.b64_json, description="Generated image with embedded data")
                images.append(ResourceContent(type="resource", resource=embedded_resource))
        return images

    def _normalize_common(
        self,
        *,
        req_n: int | None,
        orientation: Orientation | None,
        quality: Quality | None,
        background: Background | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        """Return (native_params, normalization_log, dropped_params).

        - Maps unified parameters to provider native params.
        - Clamps n to 1..4 for GPT-Image-1.
        """
        normalization: dict[str, Any] = {}
        dropped: list[str] = []

        # Process each parameter type
        n, n_norm = self._normalize_count(req_n)
        normalization.update(n_norm)

        native_size, size_norm = self._normalize_size(orientation)
        normalization.update(size_norm)

        q_native, quality_norm = self._normalize_quality(quality)
        normalization.update(quality_norm)

        bg_native, bg_norm, bg_dropped = self._normalize_background(background)
        normalization.update(bg_norm)
        dropped.extend(bg_dropped)

        # Always drop unsupported fields

        # Build native parameters
        native: dict[str, Any] = {"n": n, "size": native_size}
        if q_native:
            native["quality"] = q_native
        if bg_native:
            native["background"] = bg_native

        return native, normalization, dropped

    # ------------------------------- generate ----------------------------- #
    async def generate(self, req: ImageGenerateRequest) -> ImageResponse:  # type: ignore[override]
        provider = self._select_provider(getattr(req, "provider", None))

        # model passthrough; default to gpt-image-1 for AR OpenAI/Azure
        model = req.model

        native, normlog, dropped = self._normalize_common(
            req_n=req.n,
            orientation=req.orientation,
            quality=req.quality,
            background=req.background,
        )

        try:
            # Build prompt with XML guidance for unsupported knobs (negative_prompt)
            final_prompt, augment_log = render_prompt_with_guidance(
                prompt=req.prompt,
                model=model,
                negative_prompt=req.negative_prompt,
            )
            normlog.update(augment_log)

            client = self._get_client()

            # Azure GPT-Image-1 returns base64; response_format may be ignored.
            result = await client.images.generate(
                model=model.value,
                prompt=final_prompt,
                size=native["size"],
                n=native["n"],
                quality=native.get("quality"),
                background=native.get("background"),
            )
            images = self._process_image_data(result, "b64_json")  # Azure returns base64
            return ImageResponse(ok=True, content=images, model=model)

        except Exception as e:  # pragma: no cover - provider failure path
            return self._create_provider_error(model, provider, normlog, dropped, e)

    async def edit(self, req: ImageEditRequest) -> ImageResponse:  # type: ignore[override]
        provider = self._select_provider(getattr(req, "provider", None))
        model = req.model

        native, normlog, dropped = self._normalize_common(
            req_n=req.n,
            orientation=req.orientation,
            quality=req.quality,
            background=req.background,
        )

        try:
            if len(req.images) == 0:
                raise ValueError("At least one image must be provided for editing")

            # Use ExitStack to ensure all opened file handles are closed on exit
            with contextlib.ExitStack() as stack:
                # Open mask if provided
                mask_io = stack.enter_context(open(req.mask, "rb")) if req.mask else None

                # Open all image sources as file-like objects; the stack will close them
                image_files = [stack.enter_context(open(image_src, "rb")) for image_src in req.images]

                # Build prompt with XML guidance for unsupported knobs (negative_prompt)
                final_prompt, augment_log = render_prompt_with_guidance(
                    prompt=req.prompt,
                    model=model,
                    negative_prompt=req.negative_prompt,
                )
                normlog.update(augment_log)

                client = self._get_client()

                result = await client.images.edit(
                    model=model.value,
                    image=image_files,
                    prompt=final_prompt,
                    size=native["size"],
                    n=native["n"],
                    quality=native.get("quality"),
                    background=native.get("background"),
                    **{"mask": mask_io} if mask_io else {},  # type: ignore
                )
                images = self._process_image_data(result, "b64_json")  # Azure returns base64
                return ImageResponse(ok=True, content=images, model=model)

        except Exception as e:  # pragma: no cover - provider failure path
            return self._create_provider_edit_error(model, provider, normlog, dropped, e)
