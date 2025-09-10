from __future__ import annotations

import base64
from enum import StrEnum
from typing import Any

from google import genai
from google.genai import types
from google.oauth2 import service_account

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
from ..factory import ModelFactory

settings = get_settings()

SCOPES = [
    "https://www.googleapis.com/auth/generative-language",
    "https://www.googleapis.com/auth/cloud-platform",
]

# Imagen-specific enums/constants


class ImagenAspect(StrEnum):
    """Imagen aspect ratio mapping from unified `Orientation` enum."""

    SQUARE = "1:1"
    PORTRAIT = "9:16"
    LANDSCAPE = "16:9"

    @classmethod
    def from_orientation(cls, orientation: Orientation | None) -> str:
        if orientation == Orientation.PORTRAIT:
            return cls.PORTRAIT.value
        if orientation == Orientation.LANDSCAPE:
            return cls.LANDSCAPE.value
        return cls.SQUARE.value


class ImagenErrorType(StrEnum):
    """Imagen-specific error types."""

    NO_IMAGES = "no_images_generated"
    INVALID_CONFIG = "invalid_config"
    CLIENT_ERROR = "client_error"


# Vertex Imagen engine


class VertexImagen(ImageEngine):
    """Vertex Imagen adapter for Vertex AI."""

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
                # Generation-only Imagen models
                ModelCapability(
                    model=Model.IMAGEN_4_STANDARD,
                    supports_negative_prompt=True,
                    supports_background=False,
                    max_n=4,
                    supports_edit=False,
                    supports_mask=False,
                ),
                ModelCapability(
                    model=Model.IMAGEN_4_FAST,
                    supports_negative_prompt=True,
                    supports_background=False,
                    max_n=4,
                    supports_edit=False,
                    supports_mask=False,
                ),
                ModelCapability(
                    model=Model.IMAGEN_4_ULTRA,
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
                # Only Imagen model that supports editing
                ModelCapability(
                    model=Model.IMAGEN_3_CAPABILITY,
                    supports_negative_prompt=True,
                    supports_background=False,
                    max_n=4,
                    supports_edit=True,
                    supports_mask=True,
                ),
            ],
        )

    # Client management

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

    def _create_invalid_config_error(self, model: Model, message: str) -> ImageResponse:
        """Create error response for invalid configuration."""
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=ImagenErrorType.INVALID_CONFIG.value, message=message),
        )

    # Internal helpers for edit
    def _create_base_image(self, image_src: str) -> tuple[types.Image, bytes, str] | ImageResponse:
        """Read image bytes and return a `types.Image` plus the raw bytes and mime type.

        Returns ImageResponse on error to allow early return from caller.
        """
        try:
            image_bytes, image_mime = self.read_image_bytes_and_mime(image_src)
            base_image = types.Image(image_bytes=image_bytes, mime_type=image_mime)
            return base_image, image_bytes, image_mime
        except Exception:
            return self._create_invalid_config_error(Model.IMAGEN_3_CAPABILITY, "Failed to create Image object from base image for edit")

    def _create_mask_ref(self, mask_src: str | None, fallback_bytes: bytes, fallback_mime: str) -> types.MaskReferenceImage | None:
        """Create a MaskReferenceImage from provided mask_src or fallback bytes.

        Returns None if creation failed.
        """
        try:
            if mask_src:
                mask_bytes, mask_mime = self.read_image_bytes_and_mime(mask_src)
            else:
                mask_bytes, mask_mime = fallback_bytes, fallback_mime

            mask_image = types.Image(image_bytes=mask_bytes, mime_type=mask_mime)
            return types.MaskReferenceImage(
                reference_id=2,
                reference_image=mask_image,
                config=types.MaskReferenceConfig(
                    mask_mode=types.MaskReferenceMode.MASK_MODE_FOREGROUND,
                    mask_dilation=0,
                ),
            )
        except Exception:
            return None

    def _build_reference_images(self, image_src: str, mask_src: str | None) -> tuple[list[Any], ImageResponse | None]:
        """Build RawReferenceImage and optional MaskReferenceImage list.

        Returns a tuple of (reference_images, ImageResponse) where ImageResponse
        is non-None when an early error occurred (so caller can return it).
        """
        base_result = self._create_base_image(image_src)
        if isinstance(base_result, ImageResponse):
            return [], base_result
        base_image, image_bytes, image_mime = base_result

        refs: list[Any] = [types.RawReferenceImage(reference_id=1, reference_image=base_image)]
        mask_ref = self._create_mask_ref(mask_src, image_bytes, image_mime)
        if mask_ref:
            refs.append(mask_ref)
        return refs, None

    def _build_edit_config(self, n: int, aspect_ratio: str, negative_prompt: str | None) -> types.EditImageConfig:
        cfg = types.EditImageConfig(
            edit_mode=types.EditMode.EDIT_MODE_INPAINT_INSERTION,
            number_of_images=n,
            aspect_ratio=aspect_ratio,
            include_rai_reason=True,
            output_mime_type="image/png",
        )
        if negative_prompt:
            cfg.negative_prompt = negative_prompt
        return cfg

    # Parameter normalization

    def _normalize_count(self, req_n: int | None) -> int:
        """Normalize and clamp the count parameter for Imagen."""
        n = int(req_n or C.DEFAULT_N)
        if n < 1:
            n = 1
        if n > C.MAX_N:
            n = C.MAX_N
        return n

    def _map_aspect_ratio(self, orientation: Orientation | None) -> str:
        """Map unified orientation to Imagen aspect ratio string using `ImagenAspect`."""
        return ImagenAspect.from_orientation(orientation)

    # Error handling

    def _create_provider_error(self, model: Model, exception: Exception) -> ImageResponse:
        """Create error response for provider API failures."""
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

    # Response processing

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

    # API operations

    async def generate(self, req: ImageGenerateRequest) -> ImageResponse:
        """Generate images using Imagen via Vertex AI."""
        model = req.model

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
        model = req.model

        # Validation is performed after we determine whether a mask will be
        # supplied or a fallback mask is created (so models that require a
        # mask pass validation when we generate a fallback mask from the
        # original image).

        try:
            # Validate required input image
            if not req.images or len(req.images) == 0:
                return self._create_invalid_config_error(model, "images[0] is required for edit")

            image_src = req.images[0]
            if not image_src:
                return self._create_invalid_config_error(model, "images[0] is required for edit")

            # Early validation - check if model supports editing at all
            validation_error = ModelFactory.validate_edit_request(model, has_mask=False)
            if validation_error:
                return ImageResponse(ok=False, content=[], model=model, error=validation_error)

            client = self._create_client()

            # Normalize parameters
            n = self._normalize_count(req.n)
            aspect_ratio = self._map_aspect_ratio(req.orientation)

            # Build reference images (base + optional mask) - helper handles early errors
            reference_images, err = self._build_reference_images(image_src, req.mask)
            if err:
                return err

            # Final validation - check masking capability if mask is present
            if req.mask and not ModelFactory.model_supports_masking(model):
                validation_error = ModelFactory.validate_edit_request(model, has_mask=True)
                if validation_error:
                    return ImageResponse(ok=False, content=[], model=model, error=validation_error)

            # Build edit config and call API
            edit_config = self._build_edit_config(n, aspect_ratio, req.negative_prompt)
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
