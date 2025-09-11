from __future__ import annotations

# Tool descriptions used by FastMCP when registering tools. Keep short and clear.
TOOL_DESCRIPTIONS: dict[str, str] = {
    "generate_image": "Generate image(s) from a text prompt. Prefer explicit provider+model; respect model capabilities.",
    "edit_image": "Edit an image with a prompt and optional mask. Pass images as data URLs/base64/https URLs.",
    "get_model_capabilities": "Return enabled providers and per-model capability metadata (generation/edit/mask/limits).",
}


# High-level, concise server instructions for agents.
# Guidance follows best practices: short role, concrete workflow, strict rules, and
# clear outputs. Larger examples and troubleshooting remain in README.md.
SERVER_INSTRUCTIONS: str = (
    "Image Generation MCP Server - Agent Instructions.\n"
    "Role: This server exposes three tools: get_model_capabilities, generate_image, and edit_image. "
    "It is authoritative about which providers and models are enabled at runtime.\n\n"
    "Workflow (short):\n"
    "1) Call get_model_capabilities to discover enabled providers/models and per-model flags.\n"
    "2) Pick a provider+model that supports the operation you need (check supports_generation, "
    "supports_edit, supports_mask).\n"
    "3) Call generate_image or edit_image with explicit provider and model and provider-agnostic "
    "parameters (size, orientation, quality, negative_prompt, background, n, directory).\n\n"
    "Hard rules (must follow):\n"
    "- Always discover capabilities first; do not assume a model supports generation, editing, or masks.\n"
    "- Honor max_n for the chosen model.\n"
    "- Set negative_prompt or background only if supported by the model.\n"
    "- Use mask only where supports_mask is true.\n"
    "- Prefer explicit provider+model; omitting them may produce non-deterministic routing.\n\n"
    "Outputs and failures (summary):\n"
    "- Successful calls return MCP ImageContent blocks and a structured JSON payload of type "
    "ImageToolStructured containing model, image_count, images, meta, and error.\n"
    "- Validation failures are returned as a structured error in ImageToolStructured.\n"
    "- Provider-level failures (auth, rate limit, API errors) surface as MCP ToolErrors with "
    "stable error codes. Handle both structured errors and ToolErrors.\n\n"
    "If you need more examples or troubleshooting, see README.md in the project root."
)


__all__ = ["TOOL_DESCRIPTIONS", "SERVER_INSTRUCTIONS"]
