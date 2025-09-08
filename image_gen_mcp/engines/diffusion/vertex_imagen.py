from __future__ import annotations

import base64
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
    Family,
    ImagenAspectRatio,
    ImagenResolutionHint,
    Model,
    Orientation,
    Provider,
    Quality,
    SizeCode,
)
from ..base_engine import ImageEngine

settings = get_settings()


# ============================================================================
# IMAGEN SPECIFIC ENUMS AND CONSTANTS
# ============================================================================


class ImagenErrorType(StrEnum):
    """Imagen-specific error types."""

    NO_IMAGES = "no_images_generated"
    SDK_MISSING = "sdk_missing"
    CONFIG_MISSING = "config_missing"
    EDIT_NOT_SUPPORTED = "edit_not_supported"


class ImagenResponseFormat(StrEnum):
    """Imagen response format constants."""

    BASE64 = "base64"


# Try to import Google Gen AI SDK. Defer hard failure to runtime usage.
try:  # pragma: no cover - import-time feature detection
    from google import genai as genai_sdk  # type: ignore
except Exception:  # pragma: no cover - SDK not installed
    genai_sdk = None  # type: ignore


# ============================================================================
# VERTEX IMAGEN ENGINE CLASS
# ============================================================================


class VertexImagen(ImageEngine):
    """Vertex AI Imagen adapter using the Google Gen AI SDK.

    This adapter provides a clean interface for Google's Imagen models (3.0 and 4.0)
    via Vertex AI. Key features:

    - Supports Imagen 4 (imagen-4.0-generate-001) and Imagen 3 (imagen-3.0-generate-002)
    - Uses official Google Gen AI SDK for Vertex AI routing
    - Returns base64 images extracted from inline response parts
    - Handles parameter normalization and error conditions gracefully
    - No edit support (Imagen limitation)

    Architecture:
    - Modular error handling and response processing methods
    - Consistent normalization and metadata structure
    - Comprehensive capability discovery
    - Graceful SDK availability checking
    """

    def __init__(self, provider: Provider) -> None:
        super().__init__(provider=provider, name=f"diffusion:{provider.value}")

    # ========================================================================
    # CAPABILITY DISCOVERY
    # ========================================================================

    # --------------------------- capabilities ----------------------------- #
    def get_capability_report(self) -> CapabilityReport:
        """Return capability report for Vertex AI Imagen."""
        # Imagen 4 and Imagen 3 models are supported

        # Both Imagen models share parameters; advertise them at report level.
        return CapabilityReport(
            provider=self.provider,
            family=Family.DIFFUSION,
            models=[
                ModelCapability(
                    model=Model.IMAGEN_4_STANDARD,
                    supports_negative_prompt=True,
                    supports_background=False,
                    max_n=4,
                    supports_edit=False,
                    supports_mask=False,
                ),
                ModelCapability(
                    model=Model.IMAGEN_3_GENERATE,
                    supports_negative_prompt=True,
                    supports_background=False,
                    max_n=4,
                    supports_edit=False,
                    supports_mask=False,
                ),
            ],
        )

    # ========================================================================
    # CLIENT MANAGEMENT
    # ========================================================================

    # --------------------------- client helpers --------------------------- #
    def _client(self) -> Any:
        """Create Vertex AI client using Google Gen AI SDK."""
        if genai_sdk is None:  # pragma: no cover
            raise RuntimeError("google-genai SDK is required but not installed for Vertex Imagen")
        if not settings.vertex_project:
            raise ValueError("VERTEX_PROJECT environment variable must be set to use Vertex AI provider")
        if not settings.vertex_location:
            raise ValueError("VERTEX_LOCATION environment variable must be set to use Vertex AI provider")
        return genai_sdk.Client(vertexai=True, project=settings.vertex_project, location=settings.vertex_location)

    # ========================================================================
    # PARAMETER NORMALIZATION
    # ========================================================================

    # --------------------------- normalization utils ---------------------- #
    def _normalize_count(self, req_n: int | None) -> tuple[int, dict[str, Any]]:
        """Normalize and clamp the count parameter for Imagen."""
        normlog = {}
        n = int(req_n or C.DEFAULT_N)
        if n < 1:
            n = 1
        if n > C.MAX_N:
            n = C.MAX_N
            normlog["clamped_n"] = n
        normlog["n"] = n
        return n, normlog

    def _normalize_aspect_and_resolution(
        self,
        size: SizeCode | None,
        orientation: Orientation | None,
        quality: Quality | None,
    ) -> tuple[ImagenAspectRatio | None, ImagenResolutionHint | None, dict[str, Any]]:
        """Map unified enums to intended Imagen-native controls.

        Returns (aspect_ratio, resolution_hint, normalization_log).

        Note: Current SDK call below does not set native fields; we record
        normalization and fold guidance into the prompt for consistency.
        """
        normlog: dict[str, Any] = {}

        # Orientation → aspect ratio
        aspect_map: dict[Orientation, ImagenAspectRatio] = {
            Orientation.SQUARE: ImagenAspectRatio.ONE_ONE,
            Orientation.PORTRAIT: ImagenAspectRatio.THREE_FOUR,
            Orientation.LANDSCAPE: ImagenAspectRatio.FOUR_THREE,
        }
        aspect_ratio = aspect_map.get(orientation or Orientation.SQUARE)
        normlog["aspect_ratio"] = aspect_ratio.value if aspect_ratio else None

        # Size → resolution hint (clamp L to 2K)
        hint_map: dict[SizeCode, ImagenResolutionHint] = {
            SizeCode.S: ImagenResolutionHint.ONE_K,
            SizeCode.M: ImagenResolutionHint.TWO_K,
            SizeCode.L: ImagenResolutionHint.TWO_K,
        }
        resolution_hint = hint_map.get(size or SizeCode.M)
        normlog["resolution_hint"] = resolution_hint.value if resolution_hint else None

        # Quality is often ignored for Imagen public surfaces; record if present
        if quality is not None:
            normlog["quality"] = quality.value

        return aspect_ratio, resolution_hint, normlog

    # --------------------------- error handling --------------------------- #
    def _create_provider_error(self, model: Model, normlog: dict[str, Any], exception: Exception) -> ImageResponse:
        """Create error response for provider API failures."""
        from ...utils.error_helpers import augment_with_capability_tip

        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=C.ERROR_CODE_PROVIDER_ERROR, message=augment_with_capability_tip(str(exception))),
        )

    def _create_no_images_error(self, model: Model, normlog: dict[str, Any]) -> ImageResponse:
        """Create error response when no images are generated."""
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=ImagenErrorType.NO_IMAGES.value, message="No image content in response"),
        )

    def _create_edit_not_supported_error(self, model: Model) -> ImageResponse:
        """Create error response for unsupported edit operations."""
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=ImagenErrorType.EDIT_NOT_SUPPORTED.value, message="Imagen edits not implemented yet."),
        )

    # ========================================================================
    # RESPONSE PROCESSING
    # ========================================================================

    # --------------------------- response helpers -------------------------- #
    def _collect_images_from_response(self, resp: Any) -> list[ResourceContent]:
        """Extract embedded resources from Vertex AI response."""
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

                data = getattr(inline, "data", None)
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

        except Exception:
            # Return empty list; calling method will handle appropriately
            pass

        return images

    # ========================================================================
    # API OPERATIONS
    # ========================================================================

    # ------------------------------- generate ----------------------------- #
    async def generate(self, req: ImageGenerateRequest) -> ImageResponse:  # type: ignore[override]
        """Generate images using Imagen via Vertex AI."""
        model = req.model or Model.IMAGEN_4_STANDARD

        # Normalize parameters
        n, normlog = self._normalize_count(req.n)
        aspect_ratio, resolution_hint, shape_norm = self._normalize_aspect_and_resolution(
            size=getattr(req, "size", None),
            orientation=getattr(req, "orientation", None),
            quality=getattr(req, "quality", None),
        )
        normlog.update(shape_norm)

        try:
            client = self._client()
            # Fold guidance into prompt until native fields are exposed via SDK
            guidance_bits: list[str] = []
            if aspect_ratio:
                guidance_bits.append(f"aspect: {aspect_ratio.value}")
            if resolution_hint:
                guidance_bits.append(f"approx resolution: {resolution_hint.value}")
            if getattr(req, "quality", None) is not None:
                guidance_bits.append(f"quality: {getattr(req, 'quality').value}")
            if getattr(req, "negative_prompt", None):
                guidance_bits.append(f"avoid: {getattr(req, 'negative_prompt')}")

            prompt = req.prompt
            if guidance_bits:
                prompt = req.prompt + "\n\n[guidance: " + "; ".join(guidance_bits) + "]"

            resp = await client.aio.models.generate_content(model=str(model), contents=[prompt])

            images = self._collect_images_from_response(resp)
            if not images:
                return self._create_no_images_error(model, normlog)

            return ImageResponse(ok=True, content=images, model=model)

        except Exception as e:  # pragma: no cover - provider failure path
            return self._create_provider_error(model, normlog, e)

    # -------------------------------- edit ------------------------------- #
    async def edit(self, req: ImageEditRequest) -> ImageResponse:  # type: ignore[override]
        """Imagen does not support edit operations."""
        model = req.model or Model.IMAGEN_4_STANDARD
        return self._create_edit_not_supported_error(model)
