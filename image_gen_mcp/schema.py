from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .shard.enums import (
    Background,
    Family,
    Model,
    Orientation,
    Provider,
    Quality,
    SizeCode,
)

# ------------------------------ Error handling ------------------------------ #


class Error(BaseModel):
    """Normalized error provided on failures.

    Use short, actionable messages and stable error codes suitable for client
    handling and retries.
    """

    code: str = Field(description="Stable machine-readable error code, e.g. 'validation_error'.")
    message: str = Field(description="Human-readable error message with remediation tips when possible.")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional provider/debug details; treat as best-effort and unstable for parsing.",
    )


# ------------------------------- Image payloads ----------------------------- #


class EmbeddedResource(BaseModel):
    """Internal embedded resource with base64 blob (engine-facing only)."""

    uri: str = Field(description="Resource URI identifier (e.g., 'image://abc123').")
    name: str = Field(description="Resource name for display (e.g., 'generated-image.png').")
    mimeType: str = Field(description="MIME type of the resource content (e.g., 'image/png').")
    blob: str = Field(description="Base64-encoded resource content.")
    description: str | None = Field(default=None, description="Optional human-readable description.")
    file_path: str | None = Field(default=None, description="Absolute filesystem path where the image was saved, if applicable.")


class ResourceContent(BaseModel):
    """Internal container for embedded resources (engine-facing only)."""

    type: str = Field(default="resource", description="Content type identifier.")
    resource: EmbeddedResource = Field(description="The embedded resource data.")


# ------------------------------ Capability schema --------------------------- #


class ModelCapability(BaseModel):
    """Model-specific capability record within a provider engine.

    To meet the "only show non-shared" requirement, all fields other than
    'model' are optional. Engines should omit (leave as None) any parameter
    that is shared by all models they expose.
    """

    model: Model = Field(description="Model identifier.")
    supports_negative_prompt: bool | None = Field(default=None, description="Whether negative_prompt is honored (omit if shared).")
    supports_background: bool | None = Field(default=None, description="Whether background transparency is honored (omit if shared).")
    max_n: int | None = Field(default=None, description="Maximum images per request for this model (omit if shared).")
    supports_edit: bool | None = Field(default=None, description="Whether edit operations are supported for this model (omit if shared).")
    supports_mask: bool | None = Field(default=None, description="Whether masking is supported during edit (omit if shared).")
    supports_multi_image_edit: bool | None = Field(
        default=None,
        description="Whether multiple input images are supported during edit (omit if shared).",
    )


class CapabilityReport(BaseModel):
    """Advertises enabled engines and model-specific capabilities.

    Engines should omit per-model parameters that are shared by all models.
    """

    provider: Provider = Field(description="Provider id for routing: openai | openrouter | azure | vertex | gemini.")
    family: Family = Field(description="Engine family classification affecting knobs and defaults.")
    models: list[ModelCapability] = Field(default_factory=list, description="Capabilities per model exposed by this engine.")


class CapabilitiesResponse(BaseModel):
    """Response for get_model_capabilities tool."""

    ok: bool = Field(default=True, description="Always true when the request succeeds.")
    capabilities: list[CapabilityReport] = Field(default_factory=list, description="List of enabled engines based on credentials.")


# ------------------------------- Generate API -------------------------------- #


class ImageGenerateRequest(BaseModel):
    """Generate an image from a text prompt.

    The server routes to an enabled provider based on 'engine' and/or 'model'.
    If neither is set, the first available provider is used.
    """

    prompt: str
    provider: Provider
    model: Model
    n: int = Field(default=1, ge=1)
    # Unified controls (model-agnostic) — adapters convert to native knobs.
    size: SizeCode | None = Field(default=None)
    orientation: Orientation | None = Field(default=None)
    quality: Quality | None = Field(default=None)
    negative_prompt: str | None = Field(default=None)
    background: Background | None = Field(default=None)
    # Optional directory to save generated images
    directory: str | None = Field(default=None, description="Optional directory path to save generated images. If not provided, images will be saved to a temporary directory.")


# -------------------------- Public minimal tool output ----------------------- #


class ImageDescriptor(BaseModel):
    """Lightweight image metadata for structured tool outputs (no blobs)."""

    uri: str | None = Field(default=None, description="Opaque image URI if available (no data).")
    name: str | None = Field(default=None, description="Suggested filename (no data).")
    mimeType: str | None = Field(default=None, description="MIME type for the image.")
    file_path: str | None = Field(default=None, description="Absolute filesystem path where the image was saved, if applicable.")


class ImageToolStructured(BaseModel):
    """Public structured output for image tools without binary payloads."""

    ok: bool = Field(default=True, description="True on success; false when an error occurred.")
    model: Model = Field(description="Model used for the operation.")
    image_count: int = Field(default=0, description="Number of images returned in content blocks.")
    images: list[ImageDescriptor] = Field(default_factory=list, description="Lightweight metadata for returned images.")
    meta: dict[str, Any] = Field(default_factory=dict, description="Provider/runtime metadata (no image data).")
    error: Error | None = Field(default=None, description="Error information when ok == false.")


# Internal response type for engines (not exported)
class ImageResponse(BaseModel):
    """Internal response format used by engines before conversion to embedded resources."""

    ok: bool
    content: list[ResourceContent] = Field(default_factory=list)
    model: Model
    error: Error | None = None

    def build_structured_from_response(self) -> ImageToolStructured:
        """Convert internal ImageResponse into minimal, public structured output.

        Drops all binary payloads; includes only metadata and counts.
        """
        descriptors: list[ImageDescriptor] = []
        for part in self.content:
            try:
                if getattr(part, "type", None) != "resource":
                    continue
                res = part.resource
                descriptors.append(ImageDescriptor(uri=res.uri, name=res.name, mimeType=res.mimeType, file_path=getattr(res, "file_path", None)))
            except Exception:
                # Skip malformed entries; structured output stays minimal
                continue
        return ImageToolStructured(
            ok=self.ok,
            model=self.model,
            image_count=len(descriptors),
            images=descriptors,
            error=self.error,
        )

    def response_to_image_contents(self) -> list[Any]:
        """Convert internal ImageResponse into FastMCP ImageContent blocks.

        This emits image content only; no JSON blobs are returned to clients.
        """
        from base64 import b64decode

        from fastmcp.utilities.types import Image as FastMCPImage  # local import to avoid hard deps in type space

        def _mime_to_format(mime: str | None) -> str:
            if not mime:
                return "png"
            lower = mime.lower()
            if lower.endswith("/png"):
                return "png"
            if lower.endswith("/jpeg") or lower.endswith("/jpg"):
                return "jpeg"
            if lower.endswith("/webp"):
                return "webp"
            if lower.endswith("/gif"):
                return "gif"
            return lower.split("/")[-1] or "png"

        results: list[Any] = []
        for part in self.content:
            try:
                if getattr(part, "type", None) != "resource":
                    continue
                res = getattr(part, "resource", None)
                if not res or not getattr(res, "blob", None):
                    continue
                mime = getattr(res, "mimeType", None) or "image/png"
                fmt = _mime_to_format(mime)
                data = b64decode(res.blob)
                img = FastMCPImage(data=data, format=fmt)
                results.append(img.to_image_content(mime_type=mime))
            except Exception:
                continue
        return results


# --------------------------------- Edit API --------------------------------- #


class ImageEditRequest(BaseModel):
    """Edit an image with a text prompt and optional mask.

    Public schema exposes only `images: list[str]` for input images. Each item
    may be an HTTP(S) URL, data URL, or bare base64 string. Engines may accept
    multiple items but single-image models must use the first entry.
    """

    prompt: str
    images: list[str] = Field(..., description="Input images; first entry is used for single-image models.")
    mask: str | None = Field(default=None, description="Optional mask (URL, data URL, or base64).")

    # Routing and model selection
    provider: Provider
    model: Model
    n: int = Field(default=1, ge=1)
    # Unified controls (model-agnostic) — adapters convert to native knobs.
    size: SizeCode | None = Field(default=None)
    orientation: Orientation | None = Field(default=None)
    quality: Quality | None = Field(default=None)
    negative_prompt: str | None = Field(default=None)
    background: Background | None = Field(default=None)
    # Optional directory to save edited images
    directory: str | None = Field(default=None, description="Optional directory path to save edited images. If not provided, images will be saved to a temporary directory.")

    # Pre-process to normalize legacy inputs into images[0] without exposing them
    # in the public schema.
    @classmethod
    def _coerce_to_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            # filter out Nones and ensure all are str
            return [str(v) for v in value if v is not None]
        # Best-effort coercion
        return [str(value)]

    @classmethod
    def _normalize_images(cls, raw: dict[str, Any]) -> list[str]:
        # If images already provided, coerce and return
        if "images" in raw and raw["images"] is not None:
            return cls._coerce_to_list(raw["images"])  # type: ignore[arg-type]
        # Legacy aliases: image, image_b64
        if raw.get("image") is not None:
            return cls._coerce_to_list(raw.get("image"))
        if raw.get("image_b64") is not None:
            return cls._coerce_to_list(raw.get("image_b64"))
        return []

    @classmethod
    def _normalize_mask(cls, raw: dict[str, Any]) -> str | None:
        return raw.get("mask")

    @classmethod
    def _pop_legacy(cls, raw: dict[str, Any]) -> None:
        for k in ("image", "image_b64"):
            if k in raw:
                raw.pop(k, None)

    @classmethod
    def model_validate(cls, obj: Any, *args, **kwargs):  # type: ignore[override]
        # Intercept dict inputs to normalize before standard validation
        if isinstance(obj, dict):
            data = dict(obj)
            images = cls._normalize_images(data)
            mask = cls._normalize_mask(data)
            cls._pop_legacy(data)
            if images:
                data["images"] = images
            if mask is not None:
                data["mask"] = mask
            obj = data
        return super().model_validate(obj, *args, **kwargs)

    @field_validator("images")
    @classmethod
    def _ensure_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("images must contain at least one item")
        return v


def tool_input_schemas() -> Mapping[str, dict[str, Any]]:
    """Return JSON Schemas for tool input payloads keyed by tool name.

    Suitable for FastMCP to advertise tool parameter schemas.
    """

    return {
        "generate_image": ImageGenerateRequest.model_json_schema(),
        "edit_image": ImageEditRequest.model_json_schema(),
        # capabilities may accept an optional payload like { provider?: string }
        # Using a permissive object schema keeps it flexible.
        "get_model_capabilities": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": "Optional provider filter: openai | openrouter | azure | vertex | gemini.",
                }
            },
            "additionalProperties": False,
        },
    }


def tool_output_schemas() -> Mapping[str, dict[str, Any]]:
    """Return JSON Schemas for tool responses keyed by tool name."""

    return {
        # Public output contracts do not include binary payloads
        "generate_image": ImageToolStructured.model_json_schema(),
        "edit_image": ImageToolStructured.model_json_schema(),
        "get_model_capabilities": CapabilitiesResponse.model_json_schema(),
    }


__all__ = [
    # core
    "Error",
    # internal image carriers (engines only)
    "EmbeddedResource",
    "ResourceContent",
    # capabilities
    "CapabilityReport",
    "CapabilitiesResponse",
    # generate
    "ImageGenerateRequest",
    # edit
    "ImageEditRequest",
    # image response
    "ImageResponse",
    # public tool output
    "ImageDescriptor",
    "ImageToolStructured",
    # tools
    "tool_input_schemas",
    "tool_output_schemas",
]
