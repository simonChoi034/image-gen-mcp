from __future__ import annotations

# ----------------------- Tool descriptions for FastMCP ----------------------- #

TOOL_DESCRIPTIONS: dict[str, str] = {
    "generate_image": ("Generate one or more images from a text prompt. Optionally specify engine/provider and model; otherwise, the server routes to an available engine."),
    "edit_image": ("Edit an image using a text prompt with optional mask. Accepts base64, data URLs, or HTTP(S) image URLs."),
    "get_model_capabilities": ("Discover enabled engines and supported models/knobs based on current credentials."),
}


# ----------------------- High-level server instructions ---------------------- #

SERVER_INSTRUCTIONS: str = """
Agent Instructions: Image Generation MCP Server

Purpose:
Enable image generation/editing via a standardized MCP interface.

Workflow:
1. Discover tools: call get_model_capabilities to list providers/models.
2. Use tools:
   • generate_image — create images via prompt.
   • edit_image — modify images using prompt + input image.

Tool Details:
- get_model_capabilities: optional `provider`; returns supported providers, models, sizes, formats.
- generate_image: requires `prompt`; optional knobs like provider, model, count, quality, negative_prompt.
- edit_image: requires `prompt`, `image`; supports mask + same knobs as generate.

Best Practices:
- Validate against capabilities before setting parameters.
- Prefer explicit model selection for consistent behavior.

Results Format:
- Success: `{ ok: true, images: [...], model, meta }`
- Failure: `{ ok: false, error: {...} }`
"""


__all__ = ["TOOL_DESCRIPTIONS", "SERVER_INSTRUCTIONS"]
