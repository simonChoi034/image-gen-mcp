Image Gen MCP Server
=====================

Provider-agnostic MCP server for image generation and editing, built on FastMCP with clean engine routing and a stable, type‑safe schema.

- Tools: `generate_image`, `edit_image`, `get_model_capabilities`
- Providers: OpenAI, Azure OpenAI, Google Gemini, Vertex AI (Imagen, Gemini), OpenRouter
- Models: curated set mapped via a central factory for safe routing
- Output: ImageContent blocks (FastMCP-native) plus minimal structured JSON (no blobs)

> [!IMPORTANT]
> This `README.md` file is the canonical reference for the server's API, capabilities, and usage. Other design documents in the `/docs` directory may contain outdated information from earlier design phases. Please rely on this document as the single source of truth.


Installation
------------

### From PyPI

```bash
# Regular installation
pip install image-gen-mcp

# With uv
uv add image-gen-mcp

# With uvx (recommended for MCP usage)
uvx --from image-gen-mcp image-gen-mcp
```

### MCP Integration

Add to your `mcp.json`:

```json
{
  "mcpServers": {
    "image-gen-mcp": {
      "command": "uvx",
      "args": ["--from", "image-gen-mcp", "image-gen-mcp"],
      "env": {
        "OPENAI_API_KEY": "your-key-here"
      }
    }
  }
}
```


Quick Start
-----------

- Prereqs: Python 3.12+, optional `uv`
- Install deps: `uv sync` or `pip install -e .[dev]`
- Configure env: copy `.env.example` → `.env` and set keys (see Env Vars)
- Run server (stdio): `python -m src.main`
- Or via FastMCP CLI: `fastmcp run src/main.py:app`

The server exports a FastMCP app named `app`. Both entry points are supported.


What’s Included
---------------

- Unified Pydantic v2 I/O models in `src/schema.py`
- Engine adapters in `src/engines/` (AR and Diffusion families)
- Central routing with validation via `ModelFactory` (`src/engines/factory.py`)
- Capability discovery based on your credentials (`get_model_capabilities`)


Providers & Models
------------------

The server exposes a small, curated set of normalized providers and models. Routing is handled by `ModelFactory` using direct model→engine mappings.

Model Matrix

| Model | Family | Providers | Generate | Edit | Mask |
|---|---|---|---:|---:|---:|
| `gpt-image-1` | AR | `openai`, `azure` | Yes | Yes | Yes (OpenAI/Azure) |
| `dall-e-3` | Diffusion | `openai`, `azure` | Yes | No | — |
| `gemini-2.5-flash-image-preview` | AR | `gemini`, `vertex` | Yes | Yes (maskless) | No |
| `imagen-4.0-generate-001` | Diffusion | `vertex` | Yes | No (planned) | — |
| `imagen-3.0-generate-002` | Diffusion | `vertex` | Yes | No (planned) | — |
| `google/gemini-2.5-flash-image-preview` | AR | `openrouter` | Yes | Yes (maskless) | No |

Provider Model Support

| Provider | Supported Models |
|---|---|
| `openai` | `gpt-image-1`, `dall-e-3` |
| `azure` | `gpt-image-1`, `dall-e-3` |
| `gemini` | `gemini-2.5-flash-image-preview` |
| `vertex` | `imagen-4.0-generate-001`, `imagen-3.0-generate-002`, `gemini-2.5-flash-image-preview` |
| `openrouter` | `google/gemini-2.5-flash-image-preview` |

Notes

- DALL·E 3 does not support edits in this server (use the `gpt-image-1` model for edits via OpenAI/Azure).
- Imagen models do not yet support image editing; this feature will be implemented later.
- Azure OpenAI typically returns base64 only; OpenAI may return base64 or URL.
- OpenRouter adapter extracts images from varied response shapes; returns base64.
- `n` is clamped per provider (e.g., DALL·E 3 uses `n=1`).



Tool Reference
--------------

All tools are exposed by FastMCP and accept named parameters (they no longer require a single `req` object). Callers should pass the tool arguments as top-level named parameters matching the Pydantic models in `src/schema.py`.

Examples (FastMCP / programmatic clients)

`get_model_capabilities`

- Purpose: Discover enabled engines and normalized knobs given current credentials.
- Input example (call with named parameter): `{"provider": "openai"}` or omit `provider` to list all enabled engines.
- Output: `CapabilitiesResponse` (e.g., `{ "ok": true, "capabilities": [CapabilityReport, ...] }`).

`generate_image`

- Purpose: Create one or more images from a text prompt.
- Required: `prompt` (string)
- Example call (named parameters):

```
{
    "prompt": "A vibrant painting of a fox in a sunflower field",
    "provider": "openai",
    "model": "gpt-image-1",
    "n": 2,
    "size": "M",
    "orientation": "landscape"
}
```

- Optional: `n`, `size`, `orientation`, `quality`, `background`, `negative_prompt`, `extras`.
- Behavior: Unsupported knobs are dropped or normalized by the engine adapter; `n` is clamped per provider.

`edit_image`

- Purpose: Edit an image using a prompt and optional mask.
- Required: `prompt`, and at least one source image in `images` (use `images[0]` for single-image providers). Supported encodings: data URL, base64, or HTTP(S) URL.
- Example call (named parameters):

```
{
    "prompt": "Remove the background and make the subject wear a red scarf",
    "provider": "openai",
    "model": "gpt-image-1",
    "images": ["data:image/png;base64,..."],
    "mask": null
}
```

- Optional: `mask`, `n`, `size`, `orientation`, `quality`, `background`, `negative_prompt`, `extras`.
- Note: Edited images are returned as ImageContent blocks and adapter metadata; many providers return base64-encoded image blobs internally.


Request Field Guide
-------------------

Common fields (generate/edit)

| Field | Type | Description |
|---|---|---|
| `provider` | enum | **Required.** One of `openai`, `openrouter`, `azure`, `vertex`, `gemini`. |
| `model` | enum | **Required.** Specific model id (see Model Matrix). |
| `n` | int | Optional. Image count. Default 1; provider limits apply. |
| `size` | enum | Optional. Unified size class: `S` | `M` | `L`. |
| `orientation` | enum | Optional. `square` | `portrait` | `landscape`. |
| `quality` | enum | Optional. `draft` | `standard` | `high`. Maps to provider native values. |
| `background` | enum | Optional. `transparent` or `opaque` (AR engines that support transparency). |
| `negative_prompt` | str | Optional. Negative prompt if provider supports it. |
| `extras` | object | Optional. Escape hatch for provider‑specific params (e.g., `{ "style": "vivid" }`). |

Generate‑only

| Field | Type | Description |
|---|---|---|
| `prompt` | str | Text description of the desired image. |

Edit‑only

| Field | Type | Description |
|---|---|---|
| `prompt` | str | Edit instruction. |
| `images` | list[str] | One or more source images (base64, data URL, or HTTP(S) URL). Most models use only the first image. |
| `mask` | str | Optional. Mask (base64, data URL, or HTTP(S) URL). |


Examples (Python via FastMCP Client)
------------------------------------

```python
import asyncio
from fastmcp import Client


async def main():
    async with Client("src/main.py") as client:
        # 1) Capabilities (no parameters for all, or specify provider)
        caps = await client.call_tool("get_model_capabilities")
        # caps = await client.call_tool("get_model_capabilities", {"provider": "openai"})
        print("capabilities:", caps.structured_content or caps.text)

        # 2) Generate (provider/model are now required)
        gen = await client.call_tool(
            "generate_image",
            {
                "prompt": "a watercolor fox in a forest, soft light",
                "provider": "openai",
                "model": "gpt-image-1",
                "n": 1,
                "size": "M",
                "orientation": "square",
                "quality": "standard",
            },
        )
        # Structured JSON mirror (no image data, just metadata)
        print("generate:", gen.structured_content)
        # Traditional MCP content blocks include ImageContent for each image
        print("image blocks:", len(gen.content))

        # 3) Edit (provider/model are now required)
        edit = await client.call_tool(
            "edit_image",
            {
                "prompt": "add gentle sunbeams",
                "images": ["https://example.com/input.png"],
                # "mask": "data:image/png;base64,....",
                "provider": "openai",
                "model": "gpt-image-1",
                "n": 1,
                "size": "M",
                "orientation": "square",
            },
        )
        print("edit:", edit.structured_content)
        print("image blocks:", len(edit.content))


asyncio.run(main())
```

Notes

- You can also pass the in‑process server instance to `Client`: `from src.main import app as server; Client(server)`.
- FastMCP returns actual ImageContent blocks for display, and we also include a structured JSON mirror for programmatic use.


Env Vars
--------

Set only what you use. At least one provider must be configured.

| Variable | Required for | Description |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI | API key for OpenAI. |
| `AZURE_OPENAI_KEY` | Azure OpenAI | Azure OpenAI key. |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI | Azure endpoint URL. |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI | Optional; default `2024-02-15-preview`. |
| `GEMINI_API_KEY` | Gemini | Gemini Developer API key. |
| `OPENROUTER_API_KEY` | OpenRouter | OpenRouter API key. |
| `VERTEX_PROJECT` | Vertex AI | GCP project id. Required if using Vertex. |
| `VERTEX_LOCATION` | Vertex AI | GCP region (e.g., `us-central1`). |
| `VERTEX_CREDENTIALS_PATH` | Vertex AI | Optional path to GCP credentials JSON; ADC also supported. |

Tips

- Start the server and call `get_model_capabilities` to verify what’s enabled.
- For Vertex AI, either set `VERTEX_CREDENTIALS_PATH` or ensure Application Default Credentials (ADC) are available.


Running with FastMCP CLI
------------------------

FastMCP provides a CLI runner for MCP servers:

- Stdio: `fastmcp run src/main.py:app`
- SSE (HTTP): `fastmcp run src/main.py:app --transport sse --host 127.0.0.1 --port 8000`
- HTTP: `fastmcp run src/main.py:app --transport http --host 127.0.0.1 --port 8000 --path /mcp`

Point any MCP‑aware client (e.g., IDE integrations) at the script path or SSE/HTTP endpoint.


Design Notes
------------

- Pydantic models in `src/schema.py` define the public contract and are reused across tools.
- Engines are modular (`src/engines/*`) and selected via `ModelFactory` (`src/engines/factory.py`).
- Capabilities are computed from `src/settings.py` (`use_*` properties); `get_model_capabilities` advertises what’s live.
- Error shape is stable in structured JSON: `{ code, message, details? }`. Content blocks may be empty on error.


Dev Workflow
------------

### Local Development

```bash
# Setup
uv sync --all-extras --dev

# Test local build
uv build
./test-installation.sh

# Run tests
uv run pytest -v

# Linting
uv run ruff check .
uv run black --check .
uv run pyright
```

### Creating a Release

1. **Automated (Recommended)**:

   - Go to GitHub Actions
   - Run "Manual Release" workflow
   - Choose version type (patch/minor/major)
   - Wait for completion

1. **Manual**:

   ```bash
   # Create and push tag
   git tag v1.0.0
   git push origin v1.0.0

   # Create GitHub release from the web interface
   ```
