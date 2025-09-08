from __future__ import annotations

import base64
import logging
from enum import StrEnum
from typing import Any

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
    SizeCode,
)
from ...utils.prompt import render_prompt_with_guidance
from ..base_engine import ImageEngine

logger = logging.getLogger(__name__)
settings = get_settings()


# ============================================================================
# GEMINI-SPECIFIC ENUMS AND CONSTANTS
# ============================================================================


class GeminiErrorType(StrEnum):
    """Gemini-specific error types for consistent error handling."""

    NO_IMAGES = "no_images_generated"
    SDK_MISSING = "sdk_missing"


class GeminiResponseFormat(StrEnum):
    """Response format constants for Gemini API responses."""

    BASE64 = "base64"


# Try to import Google Gen AI SDK. Defer hard failure to runtime usage.
try:  # pragma: no cover - import-time feature detection
    from google import genai as genai_sdk  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except Exception:  # pragma: no cover - SDK not installed
    genai_sdk = None  # type: ignore
    genai_types = None  # type: ignore


class GeminiAR(ImageEngine):
    """Gemini/Vertex AR adapter for 'gemini-2.5-flash-image-preview'.

    This adapter provides a clean, maintainable interface for Google's Gemini model
    supporting both Gemini Developer API and Vertex AI providers. Key features:

    - Uses official Google Gen AI SDK for both Gemini and Vertex AI endpoints
    - Handles unified schema but drops unsupported parameters (size/quality/etc)
    - Returns base64 images from response inline data parts
    - Supports semantic editing via prompt-based instructions
    - Provides comprehensive capability discovery for runtime introspection

    Architecture:
    - Modular normalization methods for parameter handling
    - Reusable error response creation with consistent formatting
    - Streamlined client operations with provider-specific routing
    - Drops unsupported parameters
    """

    def __init__(self, provider: Provider) -> None:
        super().__init__(provider=provider, name=f"ar:{provider.value}")

    # ========================================================================
    # CAPABILITY DISCOVERY
    # ========================================================================
    # --------------------------- capabilities ----------------------------- #
    def get_capability_report(self) -> CapabilityReport:
        """Return capability report for this engine's provider."""
        # Single model currently exposed for Gemini image preview

        # For AR Gemini image preview:
        # - negative prompt is not a native knob (we fold guidance), so advertise False
        # - background alpha not supported
        # - n effectively 1 for deterministic output
        # - edit supported (prompt + reference image), mask not supported
        return CapabilityReport(
            provider=self.provider,
            family=Family.AR,
            models=[
                ModelCapability(
                    model=Model.GEMINI_IMAGE_PREVIEW,
                    supports_negative_prompt=True,
                    supports_background=True,
                    max_n=1,
                    supports_edit=True,
                    supports_mask=False,
                )
            ],
        )

    # ========================================================================
    # CLIENT MANAGEMENT
    # ========================================================================

    # --------------------------- client helpers --------------------------- #
    def _select_provider(self, req_provider: Provider | None) -> Provider:
        """Select provider - prefer request provider if specified, otherwise check use_vertex_ai setting, otherwise use engine's provider."""
        if req_provider:
            return req_provider

        # If use_vertex_ai is True and Vertex credentials are available, prefer Vertex
        if settings.use_vertex_ai and settings.vertex_project and settings.vertex_location:
            return Provider.VERTEX

        return self.provider

    def _genai_client(self, provider: Provider):
        if genai_sdk is None:  # pragma: no cover
            raise RuntimeError("google-genai SDK is required but not installed")

        if provider == Provider.VERTEX:
            # Route via Vertex AI using project/location. Credentials via ADC or key file env.
            if not settings.vertex_project:
                raise ValueError("VERTEX_PROJECT environment variable must be set to use Vertex AI provider")
            if not settings.vertex_location:
                raise ValueError("VERTEX_LOCATION environment variable must be set to use Vertex AI provider")
            return genai_sdk.Client(
                vertexai=True,
                project=settings.vertex_project,
                location=settings.vertex_location,
            )

        # Gemini Developer API path — require explicit API key
        api_key = settings.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set to use Gemini provider")
        return genai_sdk.Client(api_key=api_key)

    # ========================================================================
    # PARAMETER NORMALIZATION
    # ========================================================================

    # --------------------------- normalization utils ---------------------- #
    def _normalize_count(self, req_n: int | None) -> tuple[int, dict[str, Any]]:
        """Normalize and clamp the count parameter for Gemini."""
        normlog = {}
        n = int(req_n or C.DEFAULT_N)
        if n < 1:
            n = 1
        if n > C.MAX_N:
            n = C.MAX_N
            normlog["clamped_n"] = n
        return n, normlog

    def _collect_dropped_params(self, size: SizeCode | None, orientation: Orientation | None, quality: Quality | None, background: Background | None, negative_prompt: str | None) -> list[str]:
        """Collect parameters that are not supported by Gemini API."""
        dropped = []
        for field_name, value in (
            ("size", size),
            ("orientation", orientation),
            ("quality", quality),
            ("background", background),
            ("negative_prompt", negative_prompt),
        ):
            if value is not None:
                dropped.append(field_name)
        return dropped

    def _normalize_common(
        self,
        req_n: int | None,
        size: SizeCode | None,
        orientation: Orientation | None,
        quality: Quality | None,
        background: Background | None,
        negative_prompt: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        """Return (native_params, normalization_log, dropped_params).

        Gemini image generation currently doesn't expose explicit size/quality/
        background/seed parameters via the SDK; these should be encoded in the
        prompt. Unsupported knobs are dropped.
        """
        # Normalize count parameter
        n, normlog = self._normalize_count(req_n)

        # Collect dropped parameters
        # We will handle negative_prompt by folding it into prompt guidance later,
        # so do not mark it as dropped here.
        dropped = self._collect_dropped_params(size, orientation, quality, None, None)

        # Unsupported fields are dropped silently

        # Build native parameters
        native: dict[str, Any] = {"n": n}
        normlog["n"] = n
        return native, normlog, dropped

    # --------------------------- error handling --------------------------- #

    # Unsupported-field error helpers removed; engine always drops unsupported fields.

    def _provider_error_generate(
        self,
        model: Model,
        provider: Provider,
        normlog: dict[str, Any],
        dropped: list[str],
        ex: Exception,
    ) -> ImageResponse:
        """Create error response for provider API failures."""
        from ...utils.error_helpers import augment_with_capability_tip

        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=C.ERROR_CODE_PROVIDER_ERROR, message=augment_with_capability_tip(str(ex))),
        )

    def _provider_error_edit(
        self,
        model: Model,
        provider: Provider,
        normlog: dict[str, Any],
        dropped: list[str],
        ex: Exception,
    ) -> ImageResponse:
        """Create error response for provider API failures in edit operations."""
        from ...utils.error_helpers import augment_with_capability_tip

        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=C.ERROR_CODE_PROVIDER_ERROR, message=augment_with_capability_tip(str(ex))),
        )

    def _create_no_images_error(self, model: Model, provider: Provider, normlog: dict[str, Any], dropped: list[str]) -> ImageResponse:
        """Create error response when no images are generated."""
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=GeminiErrorType.NO_IMAGES.value, message="No images were returned by the model. Try a different prompt or model parameters."),
        )

    def _create_no_images_edit_error(self, model: Model, provider: Provider, normlog: dict[str, Any], dropped: list[str]) -> ImageResponse:
        """Create error response when no images are generated in edit operations."""
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=GeminiErrorType.NO_IMAGES.value, message="No images were returned by the model. Try a different prompt or model parameters."),
        )

    # ========================================================================
    # RESPONSE PROCESSING
    # ========================================================================

    # --------------------------- response helpers -------------------------- #
    @staticmethod
    def _collect_images_from_response(resp: Any) -> list[ResourceContent]:
        images: list[ResourceContent] = []
        try:
            candidates = getattr(resp, "candidates", []) or []
            if not candidates:
                return images
            parts = getattr(candidates[0], "content", None)
            parts = getattr(parts, "parts", []) if parts is not None else []
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline is None:
                    continue
                data = getattr(inline, "data", None)  # bytes
                mime = getattr(inline, "mime_type", None) or C.DEFAULT_MIME
                if not data:
                    continue
                b64 = base64.b64encode(data).decode("utf-8")
                resource_id = f"gen-{len(images) + 1}"
                embedded_resource = EmbeddedResource(
                    uri=f"image://{resource_id}",
                    name=f"image-{resource_id}.png",
                    mimeType=mime,
                    blob=b64,
                    description="Generated image with embedded data",
                )
                images.append(ResourceContent(type="resource", resource=embedded_resource))
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Failed to parse Gemini response: %s", e)
        return images

    # ========================================================================
    # API OPERATIONS
    # ========================================================================

    # ------------------------------- generate ----------------------------- #
    async def generate(self, req: ImageGenerateRequest) -> ImageResponse:  # type: ignore[override]
        provider = self._select_provider(getattr(req, "provider", None))
        model = req.model or Model.GEMINI_IMAGE_PREVIEW

        native, normlog, dropped = self._normalize_common(
            req_n=req.n,
            size=req.size,
            orientation=req.orientation,
            quality=req.quality,
            background=req.background,
            negative_prompt=req.negative_prompt,
        )

        try:
            client = self._genai_client(provider)
            # Gemini supports interleaved responses; we request with only text prompt.
            # Build XML-tagged prompt using utility template
            prompt, augment_log = render_prompt_with_guidance(
                prompt=req.prompt,
                model=model,
                size=req.size,
                orientation=req.orientation,
                quality=req.quality,
                background=req.background,
                negative_prompt=req.negative_prompt,
            )
            normlog.update(augment_log)
            resp = await client.aio.models.generate_content(
                model=str(model),
                contents=[prompt],
            )

            images = self._collect_images_from_response(resp)
            if not images:
                return self._create_no_images_error(model, provider, normlog, dropped)

            return ImageResponse(ok=True, content=images, model=model)

        except Exception as e:  # pragma: no cover - provider failure
            return self._provider_error_generate(model, provider, normlog, dropped, e)

    # -------------------------------- edit ------------------------------- #
    async def edit(self, req: ImageEditRequest) -> ImageResponse:  # type: ignore[override]
        provider = self._select_provider(getattr(req, "provider", None))
        model = req.model or Model.GEMINI_IMAGE_PREVIEW

        native, normlog, dropped = self._normalize_common(
            req_n=req.n,
            size=req.size,
            orientation=req.orientation,
            quality=req.quality,
            background=req.background,
            negative_prompt=req.negative_prompt,
        )

        try:
            image_src: str | None = None
            if req.images and len(req.images) > 0:
                image_src = req.images[0]
            if not image_src:
                raise ValueError("images[0] is required for edit")

            image_bytes, mime = self.read_image_bytes_and_mime(image_src)

            if genai_types is None:  # pragma: no cover
                raise RuntimeError("google-genai SDK is required but not installed")

            # Construct inline image part explicitly to avoid requiring PIL.
            image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime)

            client = self._genai_client(provider)
            # Build XML-tagged prompt for edit as well
            prompt, augment_log = render_prompt_with_guidance(
                prompt=req.prompt,
                model=model,
                size=req.size,
                orientation=req.orientation,
                quality=req.quality,
                background=req.background,
                negative_prompt=req.negative_prompt,
            )
            normlog.update(augment_log)
            resp = await client.aio.models.generate_content(
                model=str(model),
                contents=[prompt, image_part],
            )

            images = self._collect_images_from_response(resp)
            if not images:
                return ImageResponse(
                    ok=False,
                    content=[],
                    model=model,
                    error=Error(code=C.ERROR_CODE_PROVIDER_ERROR, message="No image content in response"),
                )
            return ImageResponse(ok=True, content=images, model=model)

        except Exception as e:  # pragma: no cover - provider failure
            return self._provider_error_edit(model, provider, normlog, dropped, e)
