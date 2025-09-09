from __future__ import annotations

import base64
import io
from enum import StrEnum
from typing import Any

from google import genai
from google.genai import types
from google.oauth2 import service_account
from PIL import Image as PILImage

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
from ..base_engine import ImageEngine

# Removed unused tempfile/os imports after refactor


settings = get_settings()

SCOPES = [
    "https://www.googleapis.com/auth/generative-language",
    "https://www.googleapis.com/auth/cloud-platform",
]

# ============================================================================
# IMAGEN SPECIFIC ENUMS AND CONSTANTS
# ============================================================================


class ImagenErrorType(StrEnum):
    """Imagen-specific error types."""

    NO_IMAGES = "no_images_generated"
    INVALID_CONFIG = "invalid_config"
    CLIENT_ERROR = "client_error"


# ============================================================================
# VERTEX IMAGEN ENGINE CLASS
# ============================================================================


class VertexImagen(ImageEngine):
    """Vertex AI Imagen adapter using the Google Gen AI SDK.

    This adapter provides a clean interface for Google's Imagen models (3.0 and 4.0)
    via Vertex AI. Key features:

    - Supports Imagen 4 (imagen-4.0-generate-001) and Imagen 3 (imagen-3.0-generate-002)
    - Uses official Google Gen AI SDK for Vertex AI routing
    - Returns base64 images extracted from response.generated_images
    - Handles parameter normalization and error conditions gracefully
    - Supports both image generation and editing with reference images and masks

    Architecture:
    - Consistent normalization and metadata structure
    - Comprehensive capability discovery
    - Proper SDK client management
    """

    def __init__(self, provider: Provider) -> None:
        super().__init__(provider=provider, name=f"diffusion:{provider.value}")

    # ========================================================================
    # CAPABILITY DISCOVERY
    # ========================================================================

    def get_capability_report(self) -> CapabilityReport:
        """Return capability report for Vertex AI Imagen."""
        return CapabilityReport(
            provider=self.provider,
            family=Family.DIFFUSION,
            models=[
                ModelCapability(
                    model=Model.IMAGEN_4_STANDARD,
                    supports_negative_prompt=True,
                    supports_background=False,
                    max_n=4,
                    supports_edit=True,
                    supports_mask=True,
                ),
                ModelCapability(
                    model=Model.IMAGEN_3_GENERATE,
                    supports_negative_prompt=True,
                    supports_background=False,
                    max_n=4,
                    supports_edit=True,
                    supports_mask=True,
                ),
            ],
        )

    # ========================================================================
    # CLIENT MANAGEMENT
    # ========================================================================

    def _create_client(self) -> genai.Client:
        """Create Vertex AI client using Google Gen AI SDK."""
        if not settings.vertex_project:
            raise ValueError("VERTEX_PROJECT environment variable must be set to use Vertex AI provider")
        if not settings.vertex_location:
            raise ValueError("VERTEX_LOCATION environment variable must be set to use Vertex AI provider")
        if not settings.vertex_credentials_path:
            raise ValueError("VERTEX_CREDENTIALS_PATH environment variable must be set to use Vertex AI provider")

        credentials = service_account.Credentials.from_service_account_file(settings.vertex_credentials_path, scopes=SCOPES)
        return genai.Client(vertexai=True, project=settings.vertex_project, location=settings.vertex_location, credentials=credentials)

    # ========================================================================
    # PARAMETER NORMALIZATION
    # ========================================================================

    def _normalize_count(self, req_n: int | None) -> int:
        """Normalize and clamp the count parameter for Imagen."""
        n = int(req_n or C.DEFAULT_N)
        if n < 1:
            n = 1
        if n > C.MAX_N:
            n = C.MAX_N
        return n

    def _map_aspect_ratio(self, orientation: Orientation | None) -> str:
        """Map unified orientation to Imagen aspect ratio string."""
        if orientation == Orientation.PORTRAIT:
            return "3:4"
        elif orientation == Orientation.LANDSCAPE:
            return "4:3"
        else:
            return "1:1"  # Default to square

    # ========================================================================
    # ERROR HANDLING
    # ========================================================================

    def _create_provider_error(self, model: Model, exception: Exception) -> ImageResponse:
        """Create error response for provider API failures."""
        from ...utils.error_helpers import augment_with_capability_tip

        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=C.ERROR_CODE_PROVIDER_ERROR, message=augment_with_capability_tip(str(exception))),
        )

    def _create_no_images_error(self, model: Model) -> ImageResponse:
        """Create error response when no images are generated."""
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=ImagenErrorType.NO_IMAGES.value, message="No image content in response"),
        )

    def _create_invalid_config_error(self, model: Model, message: str) -> ImageResponse:
        """Create error response for invalid configuration."""
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=ImagenErrorType.INVALID_CONFIG.value, message=message),
        )

    # ========================================================================
    # RESPONSE PROCESSING
    # ========================================================================

    def _extract_images_from_response(self, response: Any) -> list[ResourceContent]:
        """Extract embedded resources from Vertex AI response."""
        images: list[ResourceContent] = []

        try:
            # Try the modern SDK response shape first: `generated_images` with
            # `image.image_bytes`.
            try:
                generated_images = response.generated_images  # may raise AttributeError
            except Exception:
                generated_images = None

            if generated_images:
                for i, generated_image in enumerate(generated_images):
                    try:
                        img_bytes = generated_image.image.image_bytes
                    except Exception:
                        continue

                    b64_data = base64.b64encode(img_bytes).decode("utf-8")
                    resource_id = f"gen-{i + 1}"

                    embedded_resource = EmbeddedResource(
                        uri=f"image://{resource_id}",
                        name=f"image-{resource_id}.png",
                        mimeType="image/png",
                        blob=b64_data,
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

    async def generate(self, req: ImageGenerateRequest) -> ImageResponse:
        """Generate images using Imagen via Vertex AI."""
        model = req.model or Model.IMAGEN_4_STANDARD

        try:
            client = self._create_client()

            # Normalize parameters
            n = self._normalize_count(req.n)
            aspect_ratio = self._map_aspect_ratio(req.orientation)

            # Build configuration for Imagen generation
            config = types.GenerateImagesConfig(
                number_of_images=n,
                aspect_ratio=aspect_ratio,
                include_rai_reason=True,
                output_mime_type="image/png",
            )

            # Add negative prompt if provided
            if req.negative_prompt:
                config.negative_prompt = req.negative_prompt

            # Call the API
            response = await client.aio.models.generate_images(
                model=str(model),
                prompt=req.prompt,
                config=config,
            )

            # Extract images from response
            images = self._extract_images_from_response(response)
            if not images:
                return self._create_no_images_error(model)

            return ImageResponse(ok=True, content=images, model=model)

        except ValueError as e:
            return self._create_invalid_config_error(model, str(e))
        except Exception as e:
            return self._create_provider_error(model, e)

    async def edit(self, req: ImageEditRequest) -> ImageResponse:
        """Edit images using Imagen via Vertex AI with reference images and optional masks."""
        model = req.model or Model.IMAGEN_4_STANDARD

        try:
            # Validate required input image
            if not req.images or len(req.images) == 0:
                return self._create_invalid_config_error(model, "images[0] is required for edit")

            image_src = req.images[0]
            if not image_src:
                return self._create_invalid_config_error(model, "images[0] is required for edit")

            client = self._create_client()

            # Normalize parameters
            n = self._normalize_count(req.n)
            aspect_ratio = self._map_aspect_ratio(req.orientation)

            # Read the base image (PNG/JPEG/etc) and construct PIL image; the SDK accepts PIL images.
            image_bytes, image_mime = self.read_image_bytes_and_mime(image_src)
            try:
                base_pil = PILImage.open(io.BytesIO(image_bytes))
            except Exception:  # pragma: no cover
                return self._create_invalid_config_error(model, "Failed to decode base image bytes for edit")

            reference_images: list[Any] = [
                types.RawReferenceImage(
                    reference_id=1,
                    reference_image=base_pil,  # type: ignore[arg-type] - PIL image accepted at runtime
                )
            ]

            # Optional user-supplied mask handling; treat provided mask as foreground region to modify.
            if req.mask:
                mask_bytes, mask_mime = self.read_image_bytes_and_mime(req.mask)
                try:
                    mask_pil = PILImage.open(io.BytesIO(mask_bytes))
                except Exception:  # pragma: no cover
                    return self._create_invalid_config_error(model, "Failed to decode mask image bytes for edit")

                mask_ref_image = types.MaskReferenceImage(
                    reference_id=2,
                    reference_image=mask_pil,  # type: ignore[arg-type]
                    config=types.MaskReferenceConfig(
                        mask_mode=types.MaskReferenceMode.MASK_MODE_FOREGROUND,
                        mask_dilation=0,
                    ),
                )
                reference_images.append(mask_ref_image)

            # Build edit config
            edit_config = types.EditImageConfig(
                edit_mode=types.EditMode.EDIT_MODE_INPAINT_INSERTION,
                number_of_images=n,
                aspect_ratio=aspect_ratio,
                include_rai_reason=True,
                output_mime_type="image/png",
            )

            # Add negative prompt if provided
            if req.negative_prompt:
                edit_config.negative_prompt = req.negative_prompt

            # Call edit_image API
            response = await client.aio.models.edit_image(
                model=str(model),
                prompt=req.prompt,
                reference_images=reference_images,
                config=edit_config,
            )

            # Extract images from response
            images = self._extract_images_from_response(response)
            if not images:
                return self._create_no_images_error(model)

            return ImageResponse(ok=True, content=images, model=model)

        except ValueError as e:
            return self._create_invalid_config_error(model, str(e))
        except Exception as e:
            return self._create_provider_error(model, e)
