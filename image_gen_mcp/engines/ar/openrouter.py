from __future__ import annotations

from enum import StrEnum
from typing import Any

from openai import AsyncOpenAI

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


# OpenRouter-specific enums/constants


class OpenRouterEndpoint(StrEnum):
    """OpenRouter API endpoints."""

    BASE = "https://openrouter.ai/api/v1"


class OpenRouterErrorType(StrEnum):
    """OpenRouter-specific error types for consistent error handling."""

    NO_IMAGES = "no_images_generated"
    API_KEY_MISSING = "api_key_missing"
    EDIT_NOT_SUPPORTED = "edit_not_supported"
    UNSUPPORTED_FIELDS = "unsupported_fields"


class OpenRouterResponseFormat(StrEnum):
    """OpenRouter response format constants."""

    BASE64 = "base64"


class OpenRouterImageKey(StrEnum):
    """Common keys used to extract images from OpenRouter responses."""

    B64_JSON = "b64_json"
    BASE64 = "base64"
    B64 = "b64"
    IMAGE = "image"
    URL = "url"


# OpenRouter AR engine


class OpenRouterAR(ImageEngine):
    """OpenRouter AR adapter for Gemini image preview.

    Wraps OpenRouter's OpenAI-compatible API surface and provides robust
    response parsing and error handling for image generation and editing.
    """

    def __init__(self, provider: Provider) -> None:
        super().__init__(provider=provider, name=f"ar:{provider.value}")

    # Capability discovery
    def get_capability_report(self) -> CapabilityReport:
        """Return capability report for OpenRouter provider."""
        # Single model; advertise shared parameters at report level.
        return CapabilityReport(
            provider=Provider.OPENROUTER,
            family=Family.AR,
            models=[
                ModelCapability(
                    model=Model.OPENROUTER_GOOGLE_GEMINI_IMAGE,
                    supports_negative_prompt=True,
                    supports_background=True,
                    max_n=1,
                    supports_edit=True,
                    supports_mask=False,
                    supports_multi_image_edit=True,
                )
            ],
        )

    # Parameter normalization
    def _normalize_count(self, req_n: int | None) -> tuple[int, dict[str, Any]]:
        """Normalize and clamp the count parameter for OpenRouter."""
        normlog: dict[str, Any] = {}
        n = int(req_n or C.DEFAULT_N)
        if n < 1:
            n = 1
        if n > C.MAX_N:
            n = C.MAX_N
            normlog["clamped_n"] = n
        normlog["n"] = n
        return n, normlog

    def _collect_dropped_params(self, size: SizeCode | None, orientation: Orientation | None, quality: Quality | None, background: Background | None, negative_prompt: str | None) -> list[str]:
        """Collect parameters that are not supported by OpenRouter API."""
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

    # Error handling

    def _create_provider_error(self, model: Model, normlog: dict[str, Any], dropped: list[str], exception: Exception) -> ImageResponse:
        """Create error response for provider API failures."""
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=C.ERROR_CODE_PROVIDER_ERROR, message=augment_with_capability_tip(str(exception))),
        )

    def _create_no_images_error(self, model: Model, normlog: dict[str, Any], dropped: list[str], raw_shape: list[str]) -> ImageResponse:
        """Create error response when no images are found in response."""
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=OpenRouterErrorType.NO_IMAGES.value, message="No images found in OpenRouter response"),
        )

    def _create_edit_not_supported_error(self, model: Model) -> ImageResponse:
        """Create error response for unsupported edit operations."""
        return ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code=OpenRouterErrorType.EDIT_NOT_SUPPORTED.value, message="Edit is not supported via OpenRouter for this model."),
        )

    # Unsupported-field helpers removed; engine always drops unsupported fields.

    # HTTP client operations
    def _client(self) -> AsyncOpenAI:
        """Return configured AsyncOpenAI client routed to OpenRouter base URL."""
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable must be set to use OpenRouter provider")
        return AsyncOpenAI(
            base_url=OpenRouterEndpoint.BASE.value,
            api_key=settings.openrouter_api_key,
        )

    def _build_payload(self, model: Model, prompt: str, n: int) -> dict[str, Any]:
        """Build Chat Completions payload for OpenRouter (OpenAI-compatible)."""
        return {
            "model": str(model),
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
        }

    def _build_messages_payload(self, model: Model, messages: list[dict[str, Any]], n: int) -> dict[str, Any]:
        """Build Chat Completions payload using explicit message parts (e.g., image_url)."""
        return {"model": str(model), "messages": messages, "n": n}

    def _as_image_url(self, source: str) -> str:
        """Return a value suitable for OpenAI-style image_url.url.
        Delegates to base engine utility to normalize to a data URL.
        """
        return self.to_image_data_url(source)

    # Response processing helpers
    def _extract_images_from_response(self, obj: Any) -> list[ResourceContent]:
        """Best-effort recursive extraction of images from OpenRouter responses."""
        results: list[ResourceContent] = []

        def add_b64(b64: str, mime: str | None = None) -> None:
            resource_id = f"gen-{len(results) + 1}"
            embedded_resource = EmbeddedResource(uri=f"image://{resource_id}", name=f"image-{resource_id}.png", mimeType=mime or C.DEFAULT_MIME, blob=b64, description="Generated image with embedded data")
            results.append(ResourceContent(type="resource", resource=embedded_resource))

        def add_url(url: str, mime: str | None = None) -> None:
            if url.startswith("data:"):
                try:
                    _, b64_part = url.split(",", 1)
                    add_b64(b64_part, mime)
                except ValueError:
                    pass  # Skip malformed data URLs

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                # Check common image keys using our enum
                if OpenRouterImageKey.B64_JSON.value in node and isinstance(node[OpenRouterImageKey.B64_JSON.value], str):
                    add_b64(node[OpenRouterImageKey.B64_JSON.value])
                if OpenRouterImageKey.BASE64.value in node and isinstance(node[OpenRouterImageKey.BASE64.value], str):
                    add_b64(node[OpenRouterImageKey.BASE64.value])
                if OpenRouterImageKey.B64.value in node and isinstance(node[OpenRouterImageKey.B64.value], str):
                    add_b64(node[OpenRouterImageKey.B64.value])
                if OpenRouterImageKey.IMAGE.value in node and isinstance(node[OpenRouterImageKey.IMAGE.value], str):
                    val = node[OpenRouterImageKey.IMAGE.value]
                    if val.startswith("data:image/") and "," in val:
                        try:
                            add_b64(val.split(",", 1)[1])
                        except Exception:
                            pass
                if OpenRouterImageKey.URL.value in node and isinstance(node[OpenRouterImageKey.URL.value], str):
                    add_url(node[OpenRouterImageKey.URL.value])

                # Recursively walk all values
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(obj)
        return results

    # API operations: generate / edit
    async def generate(self, req: ImageGenerateRequest) -> ImageResponse:  # type: ignore[override]
        """Generate images using OpenRouter API."""
        model = req.model

        # Normalize parameters
        n, normlog = self._normalize_count(req.n)
        # The helper now expects (size, orientation, quality, background,
        # negative_prompt, seed, output_format). Pass the unified request
        # fields directly.
        dropped = self._collect_dropped_params(
            req.size,
            req.orientation,
            req.quality,
            req.background,
            None,  # we'll fold negative_prompt into prompt guidance instead of dropping
        )

        try:
            # Use XML-tagged guidance template for AR folding
            prompt, augment_log = render_prompt_with_guidance(
                prompt=req.prompt,
                model=model,
                orientation=req.orientation,
                quality=req.quality,
                background=req.background,
                negative_prompt=req.negative_prompt,
            )
            normlog.update(augment_log)

            payload = self._build_payload(model, prompt, n)
            client = self._client()
            completion = await client.chat.completions.create(**payload)
            # Convert to pure dict to reuse robust extraction
            resp_json = completion.model_dump() if hasattr(completion, "model_dump") else getattr(completion, "to_dict", lambda: {})()
            images = self._extract_images_from_response(resp_json)

            if not images:
                raw_shape = list(resp_json.keys()) if isinstance(resp_json, dict) else []
                return self._create_no_images_error(model, normlog, dropped, raw_shape)

            return ImageResponse(ok=True, content=images, model=model)

        except Exception as e:  # pragma: no cover - network/runtime
            return self._create_provider_error(model, normlog, dropped, e)

    # -------------------------------- edit ------------------------------- #
    async def edit(self, req: ImageEditRequest) -> ImageResponse:  # type: ignore[override]
        """Edit an image via OpenRouter using OpenAI-compatible image input.

        Constructs chat-style messages where the user's content is an array of
        parts including a text instruction and an image_url object, following
        OpenAI's Chat Completions vision message format.
        """
        model = req.model

        # Normalize parameters (reuse generate-time drop behavior)
        n, normlog = self._normalize_count(req.n)

        try:
            # Resolve image source
            if len(req.images) == 0:
                raise ValueError("At least one image must be provided for editing")

            # Build guidance-augmented text
            prompt, augment_log = render_prompt_with_guidance(
                prompt=req.prompt,
                model=model,
                orientation=req.orientation,
                quality=req.quality,
                background=req.background,
                negative_prompt=req.negative_prompt,
            )
            normlog.update(augment_log)

            # Compose messages per OpenAI vision chat format
            content_parts: list[dict[str, Any]] = [
                {"type": "text", "text": prompt},
            ]
            for image_src in req.images:
                content_parts.append({"type": "image_url", "image_url": {"url": self._as_image_url(image_src)}})

            messages = [{"role": "user", "content": content_parts}]

            payload = self._build_messages_payload(model, messages, n)
            client = self._client()
            completion = await client.chat.completions.create(**payload)
            resp_json = completion.model_dump() if hasattr(completion, "model_dump") else getattr(completion, "to_dict", lambda: {})()
            images = self._extract_images_from_response(resp_json)

            if not images:
                # Include top-level keys to aid debugging
                return ImageResponse(
                    ok=False,
                    content=[],
                    model=model,
                    error=Error(code=OpenRouterErrorType.NO_IMAGES.value, message="No images found in OpenRouter response"),
                )

            return ImageResponse(ok=True, content=images, model=model)

        except Exception as e:  # pragma: no cover - network/runtime
            return ImageResponse(
                ok=False,
                content=[],
                model=model,
                error=Error(code=C.ERROR_CODE_PROVIDER_ERROR, message=augment_with_capability_tip(str(e))),
            )


__all__ = ["OpenRouterAR"]
