from __future__ import annotations

import argparse
import asyncio
from typing import Annotated

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
from .utils.image_utils import save_images_from_response

app = FastMCP("image-gen-mcp", instructions=SERVER_INSTRUCTIONS)


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
    directory: Annotated[
        str | None,
        Field(description="Optional directory path to save generated images. If not provided, images will be saved to a temporary directory."),
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
            directory=directory,
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

        # Save images to disk only if response was successful
        if resp.ok:
            await asyncio.to_thread(save_images_from_response, resp, directory)

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
    directory: Annotated[
        str | None,
        Field(description="Optional directory path to save edited images. If not provided, images will be saved to a temporary directory."),
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
            directory=directory,
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

        # Save images to disk (runs in a thread to avoid blocking the event loop)
        await asyncio.to_thread(save_images_from_response, resp, directory)

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
        if provider:
            reports = ModelFactory.get_capabilities_for_provider(provider)
            capabilities = reports if reports else []
        else:
            # Get capabilities for all enabled providers
            enabled_providers = [p for p, enabled in ModelFactory.get_enabled_providers().items() if enabled]
            reports_nested = [ModelFactory.get_capabilities_for_provider(p) for p in enabled_providers]
            # Flatten nested lists and filter out empty reports
            capabilities = [r for sub in reports_nested for r in sub if r is not None]

        return CapabilitiesResponse(capabilities=capabilities)

    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        return CapabilitiesResponse(ok=False, capabilities=[])


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Gen MCP Server")
    # Only accept transports supported by FastMCP for server runs. Note: SSE
    # is legacy but still supported for backward compatibility.
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse", "streamable-http"],
        help="Transport to use (stdio, sse, http, streamable-http). Default: stdio",
    )
    parser.add_argument("--host", help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to listen on")
    args = parser.parse_args()

    # Override settings with command-line arguments if provided
    transport = args.transport
    host = args.host
    port = args.port

    logger.info(f"Starting MCP image server on {host}:{port} with {transport} transport")

    # FastMCP's stdio transport does not accept `host`/`port` kwargs. Only pass
    # `host`/`port` when using an HTTP-like transport (http, sse, streamable-http).
    http_transports = {"http", "sse", "streamable-http"}
    if transport in http_transports:
        app.run(transport=transport, host=host, port=port)
    else:
        # For stdio (the default): do not pass host/port
        app.run()


if __name__ == "__main__":
    main()
