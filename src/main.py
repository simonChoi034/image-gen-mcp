from __future__ import annotations

from typing import Annotated, Any

from fastmcp import Context, FastMCP
from fastmcp.tools.tool import ToolResult
from loguru import logger
from pydantic import Field

from .engines import ModelFactory
from .schema import (
    CapabilitiesResponse,
    Error,
    ImageEditRequest,
    ImageGenerateRequest,
    ImageResponse,
)
from .settings import get_settings
from .shard.enums import (
    Background,
    Model,
    Orientation,
    Provider,
    Quality,
    SizeCode,
)
from .shard.instructions import SERVER_INSTRUCTIONS, TOOL_DESCRIPTIONS
from .utils.error_helpers import augment_with_capability_tip

app = FastMCP("image-gen-mcp", instructions=SERVER_INSTRUCTIONS)
# Obtain settings explicitly and inject into tool handlers via closure capture.
# settings = get_settings()  # Removed to allow mocking in tests


@app.tool(
    name="generate_image",
    description=TOOL_DESCRIPTIONS["generate_image"],
    annotations={
        "title": "Generate Image(s)",
        "readOnlyHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def mcp_generate_image(
    prompt: Annotated[str, Field(description="Text description of the desired image.")],
    provider: Annotated[
        Provider,
        Field(description="Provider: 'openai' | 'openrouter' | 'azure' | 'vertex' | 'gemini'."),
    ],
    model: Annotated[
        Model,
        Field(description="Model id (e.g., 'gpt-image-1', 'dall-e-3', 'imagen-4.0-generate-001')."),
    ],
    n: Annotated[int | None, Field(ge=1, description="Count of images to generate; provider limits apply.")] = 1,
    size: Annotated[SizeCode | None, Field(description="Unified size class: 'S' | 'M' | 'L'.")] = None,
    orientation: Annotated[
        Orientation | None,
        Field(description="Orientation preference: 'square' | 'portrait' | 'landscape'."),
    ] = None,
    quality: Annotated[
        Quality | None,
        Field(description="Quality preference: 'draft' | 'standard' | 'high'."),
    ] = None,
    negative_prompt: Annotated[
        str | None,
        Field(description="Optional negative prompt honored by supporting providers."),
    ] = None,
    background: Annotated[
        Background | None,
        Field(description="Optional background alpha for AR engines supporting transparency."),
    ] = None,
    extras: Annotated[
        dict[str, Any] | None,
        Field(description="Escape hatch for native provider parameters (e.g., {'style':'vivid'})."),
    ] = None,
    ctx: Context | None = None,
) -> ToolResult:
    """Generate image(s) from a prompt.

    Parameters are provider/model-agnostic. Engines normalize to native knobs.
    """
    try:
        req = ImageGenerateRequest(
            prompt=prompt,
            provider=provider,
            model=model,
            n=n or 1,
            size=size,
            orientation=orientation,
            quality=quality,
            negative_prompt=negative_prompt,
            background=background,
            extras=extras,
        )

        # Use factory's integrated validation and creation
        engine, error_resp = ModelFactory.validate_and_create(provider=req.provider, model=req.model)
        if error_resp is not None:
            # Fast-fail on provider/model validation issues
            structured = error_resp.build_structured_from_response()
            return ToolResult(content=[], structured_content=structured.model_dump())

        # Engine is guaranteed to be non-None if error_resp is None
        assert engine is not None

        resp = await engine.generate(req)

        # Convert to FastMCP ImageContent while preserving structured payload
        contents = resp.response_to_image_contents()
        structured = resp.build_structured_from_response()
        return ToolResult(content=contents, structured_content=structured.model_dump())
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        resp = ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code="generation_error", message=augment_with_capability_tip(f"Failed to generate image: {str(e)}")),
        )
        structured = resp.build_structured_from_response()
        return ToolResult(content=[], structured_content=structured.model_dump())


@app.tool(
    name="edit_image",
    description=TOOL_DESCRIPTIONS["edit_image"],
    annotations={
        "title": "Edit Image",
        "readOnlyHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def mcp_edit_image(
    prompt: Annotated[str, Field(description="Text instruction describing the edit to perform.")],
    provider: Annotated[
        Provider,
        Field(description="Provider: 'openai' | 'openrouter' | 'azure' | 'vertex' | 'gemini'."),
    ],
    model: Annotated[Model, Field(description="Model id to use for editing.")],
    images: Annotated[
        list[str],
        Field(
            description=(
                "One or more image sources; most edit-capable models use only images[0]. "
                "Accepted forms: (1) http(s) URL, (2) local file path or file:// URL (the server will read and "
                "inline it), (3) data URL 'data:image/<type>;base64,<payload>', or (4) bare base64 string. "
                "Recommended: pass a data URL or base64 for best reliability. Supported types: PNG, JPEG, WEBP, GIF. "
                "Invalid or tiny placeholder images may be rejected by providers."
            )
        ),
    ],
    mask: Annotated[
        str | None,
        Field(description="Optional mask image (same encoding forms as items in 'images')."),
    ] = None,
    n: Annotated[int | None, Field(ge=1, description="Count of images to generate; provider limits apply.")] = 1,
    size: Annotated[SizeCode | None, Field(description="Unified size class: 'S' | 'M' | 'L'.")] = None,
    orientation: Annotated[
        Orientation | None,
        Field(description="Orientation preference: 'square' | 'portrait' | 'landscape'."),
    ] = None,
    quality: Annotated[
        Quality | None,
        Field(description="Quality preference: 'draft' | 'standard' | 'high'."),
    ] = None,
    negative_prompt: Annotated[
        str | None,
        Field(description="Optional negative prompt honored by supporting providers."),
    ] = None,
    background: Annotated[
        Background | None,
        Field(description="Optional background alpha for AR engines supporting transparency."),
    ] = None,
    extras: Annotated[
        dict[str, Any] | None,
        Field(description="Escape hatch for native provider parameters (e.g., {'style':'vivid'})."),
    ] = None,
    ctx: Context | None = None,
) -> ToolResult:
    """Edit an image with a prompt and optional mask."""
    try:
        req = ImageEditRequest(
            prompt=prompt,
            images=images,
            mask=mask,
            provider=provider,
            model=model,
            n=n or 1,
            size=size,
            orientation=orientation,
            quality=quality,
            negative_prompt=negative_prompt,
            background=background,
            extras=extras,
        )

        # Use factory's integrated validation and creation
        engine, error_resp = ModelFactory.validate_and_create(provider=req.provider, model=req.model)
        if error_resp is not None:
            # Fast-fail on provider/model validation issues
            structured = error_resp.build_structured_from_response()
            return ToolResult(content=[], structured_content=structured.model_dump())

        # Engine is guaranteed to be non-None if error_resp is None
        assert engine is not None

        resp = await engine.edit(req)

        contents = resp.response_to_image_contents()
        structured = resp.build_structured_from_response()
        return ToolResult(content=contents, structured_content=structured.model_dump())
    except Exception as e:
        logger.error(f"Error editing image: {e}")
        resp = ImageResponse(
            ok=False,
            content=[],
            model=model,
            error=Error(code="edit_error", message=augment_with_capability_tip(f"Failed to edit image: {str(e)}")),
        )
        structured = resp.build_structured_from_response()
        return ToolResult(content=[], structured_content=structured.model_dump())


@app.tool(
    name="get_model_capabilities",
    description=TOOL_DESCRIPTIONS["get_model_capabilities"],
    annotations={
        "title": "List Capabilities",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def mcp_get_model_capabilities(
    provider: Annotated[
        Provider | None,
        Field(description="Optional provider filter: openai | openrouter | azure | vertex | gemini."),
    ] = None,
) -> CapabilitiesResponse:
    """Return enabled engines and supported models/knobs for current credentials."""
    try:
        settings = get_settings()
        provider_filter = provider

        capabilities = []

        # Get enabled providers using the centralized helper
        enabled_providers = ModelFactory.get_enabled_providers(settings)
        supported_providers = [p for p, enabled in enabled_providers.items() if enabled]

        if not supported_providers:
            logger.debug("No providers are enabled based on current settings.")

        # Filter by provider if specified
        if provider_filter:
            if provider_filter in supported_providers:
                supported_providers = [provider_filter]
            else:
                logger.warning(f"Requested provider {provider_filter} is not supported or enabled.")
                supported_providers = []

        # Get capability reports from each engine
        for provider in supported_providers:
            try:
                engine = ModelFactory.create(provider=provider)
                # capability report remains synchronous
                capabilities.append(engine.get_capability_report())
            except Exception as e:
                logger.warning(f"Failed to get capabilities for provider {provider}: {e}")
                continue

        return CapabilitiesResponse(capabilities=capabilities)

    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        error_response = CapabilitiesResponse(ok=False, capabilities=[])
        return error_response


def main() -> None:
    logger.info("Starting MCP image server")
    app.run(transport="sse", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
