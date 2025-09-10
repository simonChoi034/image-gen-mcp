from __future__ import annotations

import base64
from enum import StrEnum
from typing import Any

from google import genai
from google.genai import types
from google.oauth2 import service_account
from loguru import logger

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
from ...utils.error_helpers import augment_with_capability_tip
from ...utils.prompt import render_prompt_with_guidance
from ..base_engine import ImageEngine

settings = get_settings()

SCOPES = [
    "https://www.googleapis.com/auth/generative-language",
    "https://www.googleapis.com/auth/cloud-platform",
]


class GeminiErrorType(StrEnum):
    """Gemini-specific error types."""

    NO_IMAGES = "no_images_generated"
    SDK_MISSING = "sdk_missing"


class GeminiAR(ImageEngine):
    """Gemini/Vertex AR image adapter.

    Routes requests to Vertex AI or the Gemini Developer API. Unsupported
    knobs are folded into prompt guidance via `render_prompt_with_guidance`.
    """

    def __init__(self, provider: Provider) -> None:
        super().__init__(provider=provider, name=f"ar:{provider.value}")

    def get_capability_report(self) -> CapabilityReport:
        """Return capability report for this engine's provider."""
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
                    supports_multi_image_edit=True,
                )
            ],
        )

    def _select_provider(self, req_provider: Provider | None) -> Provider:
        if req_provider:
            return req_provider
        if settings.use_vertex:
            return Provider.VERTEX
        return self.provider

    def _genai_client(self, provider: Provider):
        if provider == Provider.VERTEX:
            if not settings.vertex_project:
                raise ValueError("VERTEX_PROJECT environment variable must be set to use Vertex AI provider")
            if not settings.vertex_location:
                raise ValueError("VERTEX_LOCATION environment variable must be set to use Vertex AI provider")
            if not settings.vertex_credentials_path:
                raise ValueError("VERTEX_CREDENTIALS_PATH environment variable must be set to use Vertex AI provider")

            credentials = service_account.Credentials.from_service_account_file(settings.vertex_credentials_path, scopes=SCOPES)
            return genai.Client(vertexai=True, project=settings.vertex_project, location=settings.vertex_location, credentials=credentials)

        api_key = settings.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set to use Gemini provider")
        return genai.Client(api_key=api_key)

    def _normalize_count(self, req_n: int | None) -> tuple[int, dict[str, Any]]:
        normlog: dict[str, Any] = {}
        n = int(req_n or C.DEFAULT_N)
        if n < 1:
            n = 1
        if n > C.MAX_N:
            n = C.MAX_N
            normlog["clamped_n"] = n
        return n, normlog

    def _collect_dropped_params(self, size: SizeCode | None, orientation: Orientation | None, quality: Quality | None, background: Background | None, negative_prompt: str | None) -> list[str]:
        dropped: list[str] = []
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
        n, normlog = self._normalize_count(req_n)
        dropped = self._collect_dropped_params(size, orientation, quality, None, None)
        native: dict[str, Any] = {"n": n}
        normlog["n"] = n
        return native, normlog, dropped

    def _provider_error_generate(self, model: Model, provider: Provider, normlog: dict[str, Any], dropped: list[str], ex: Exception) -> ImageResponse:
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=C.ERROR_CODE_PROVIDER_ERROR, message=augment_with_capability_tip(str(ex))),
        )

    def _provider_error_edit(self, model: Model, provider: Provider, normlog: dict[str, Any], dropped: list[str], ex: Exception) -> ImageResponse:
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=C.ERROR_CODE_PROVIDER_ERROR, message=augment_with_capability_tip(str(ex))),
        )

    def _create_no_images_error(self, model: Model, provider: Provider, normlog: dict[str, Any], dropped: list[str]) -> ImageResponse:
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=GeminiErrorType.NO_IMAGES.value, message="No images were returned by the model. Try a different prompt or model parameters."),
        )

    @staticmethod
    def _collect_images_from_response(resp: types.GenerateContentResponse) -> list[ResourceContent]:
        images: list[ResourceContent] = []
        try:
            candidates = resp.candidates
            if not candidates:
                return images
            parts = candidates[0].content.parts if candidates[0].content and candidates[0].content.parts else []
            for part in parts:
                inline = part.inline_data
                if inline is None:
                    continue
                data = inline.data  # bytes
                mime = inline.mime_type or C.DEFAULT_MIME
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

    async def generate(self, req: ImageGenerateRequest) -> ImageResponse:  # type: ignore[override]
        provider = self._select_provider(req.provider)
        model = req.model

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
            prompt, augment_log = render_prompt_with_guidance(
                prompt=req.prompt,
                model=model,
                orientation=req.orientation,
                quality=req.quality,
                background=req.background,
                negative_prompt=req.negative_prompt,
            )
            normlog.update(augment_log)

            generate_content_config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            )

            resp = await client.aio.models.generate_content(model=model.value, contents=[prompt], config=generate_content_config)  # type: ignore[arg-type]

            images = self._collect_images_from_response(resp)
            if not images:
                return self._create_no_images_error(model, provider, normlog, dropped)

            return ImageResponse(ok=True, content=images, model=model)

        except Exception as e:  # pragma: no cover - provider failure
            return self._provider_error_generate(model, provider, normlog, dropped, e)

    async def edit(self, req: ImageEditRequest) -> ImageResponse:  # type: ignore[override]
        provider = self._select_provider(req.provider)
        model = req.model

        native, normlog, dropped = self._normalize_common(
            req_n=req.n,
            size=req.size,
            orientation=req.orientation,
            quality=req.quality,
            background=req.background,
            negative_prompt=req.negative_prompt,
        )

        try:
            if len(req.images) == 0:
                raise ValueError("At least one image must be provided for editing")

            client = self._genai_client(provider)

            # Read and prepare all input images
            image_parts = []
            for image in req.images:
                image_bytes, mime = self.read_image_bytes_and_mime(image)
                image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
                image_parts.append(image_part)

            prompt, augment_log = render_prompt_with_guidance(
                prompt=req.prompt,
                model=model,
                orientation=req.orientation,
                quality=req.quality,
                background=req.background,
                negative_prompt=req.negative_prompt,
            )
            normlog.update(augment_log)

            generate_content_config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            )

            resp = await client.aio.models.generate_content(model=model.value, contents=[prompt, *image_parts], config=generate_content_config)

            images = self._collect_images_from_response(resp)
            if not images:
                return ImageResponse(ok=False, content=[], model=model, error=Error(code=C.ERROR_CODE_PROVIDER_ERROR, message="No image content in response"))
            return ImageResponse(ok=True, content=images, model=model)

        except Exception as e:  # pragma: no cover - provider failure
            return self._provider_error_edit(model, provider, normlog, dropped, e)
