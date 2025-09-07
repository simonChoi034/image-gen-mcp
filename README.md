Image Gen MCP Server
=====================

Provider-agnostic MCP server for image generation and editing, built on FastMCP with clean engine routing and a stable, type‑safe schema.

- Tools: `generate_image`, `edit_image`, `get_model_capabilities`
- Providers: OpenAI, Azure OpenAI, Google Gemini, Vertex AI (Imagen, Gemini), OpenRouter
- Models: curated set mapped via a central factory for safe routing
- Output: ImageContent blocks (FastMCP-native) plus minimal structured JSON (no blobs)


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

The server exposes a small, curated set of normalized providers and models. Routing is handled by `ModelFactory` using direct model→engine mappings and provider defaults.

Model Matrix

| Model | Family | Providers | Generate | Edit | Mask | Output formats | Size support |
|---|---|---|---:|---:|---:|---|---|
| `gpt-image-1` | AR | `openai`, `azure` | Yes | Yes | Yes (OpenAI/Azure) | OpenAI: base64+url; Azure: base64 | Native sizes: 1024x1024, 1024x1536, 1536x1024 |
| `dall-e-3` | Diffusion | `openai`, `azure` | Yes | No | — | OpenAI: base64+url; Azure: base64 | Native sizes: 1024x1024, 1024x1792, 1792x1024 |
| `gemini-2.5-flash-image-preview` | AR | `gemini`, `vertex` | Yes | Yes (maskless) | No | base64 | Prompt‑driven sizing |
| `imagen-4.0-generate-001` | Diffusion | `vertex` | Yes | No | — | base64 | Prompt‑driven sizing |
| `imagen-3.0-generate-002` | Diffusion | `vertex` | Yes | No | — | base64 | Prompt‑driven sizing |
| `google/gemini-2.5-flash-image-preview` | AR | `openrouter` | Yes | No | — | base64 | Prompt‑driven sizing |

Provider Defaults (when `model` is omitted)

| Provider | Default engine | Supported models |
|---|---|---|
| `openai` | AR (GPT‑Image‑1) | `gpt-image-1`, `dall-e-3` |
| `azure` | AR (GPT‑Image‑1) | `gpt-image-1`, `dall-e-3` |
| `gemini` | AR (Gemini image preview) | `gemini-2.5-flash-image-preview` |
| `vertex` | Diffusion (Imagen) | `imagen-4.0-generate-001`, `imagen-3.0-generate-002`, `gemini-2.5-flash-image-preview` |
| `openrouter` | AR (Gemini via OpenRouter) | `google/gemini-2.5-flash-image-preview` |

Notes

- DALL·E 3 and Imagen do not support edits in this server.
- Azure OpenAI typically returns base64 only; OpenAI may return base64 or URL.
- OpenRouter adapter extracts images from varied response shapes; returns base64.
- `n` is clamped per provider (e.g., DALL·E 3 uses `n=1`).


Tool Reference
--------------

All tools are exposed by FastMCP and take a single argument named `req` shaped by the Pydantic models in `src/schema.py`. Wrap calls as `{ "req": { ... } }`.

get_model_capabilities

- Purpose: Discover enabled engines and normalized knobs given current credentials.
- Input: `{ "req": { "provider"?: "openai" | "openrouter" | "azure" | "vertex" | "gemini" } }`
- Output: `{ ok: true, capabilities: CapabilityReport[] }` (or `{ ok:false, error }`)

generate_image

- Purpose: Create one or more images from a text prompt.
- Required: `prompt`
- Optional: `provider`, `model`, `n`, `size`, `size_code`, `orientation`, `quality`, `background`, `negative_prompt`, `extras`
- Behavior: Unsupported knobs are dropped.

edit_image

- Purpose: Edit an image using a prompt and optional mask.
- Required: `prompt`, and one of `image` or `images[0]` (base64, data URL, or HTTP(S) URL)
- Optional: `mask`, `provider`, `model`, `n`, `size`, `size_code`, `orientation`, `quality`, `background`, `negative_prompt`, `extras`
- Note: Edited images are returned as base64 for most providers.


Request Field Guide
-------------------

Common fields (generate/edit)

| Field | Type | Description |
|---|---|---|
| `provider` | enum | One of `openai`, `openrouter`, `azure`, `vertex`, `gemini`. Optional hint for routing. |
| `model` | enum | Specific model id (see Model Matrix). Optional. |
| `n` | int | Image count. Default 1; provider limits apply. |
| `size` | str | Native size like `1024x1024` (legacy). Prefer `size_code`/`orientation` when available. |
| `size_code` | enum | Unified class: `S` | `M` | `L`. |
| `orientation` | enum | `square` | `portrait` | `landscape`. |
| `quality` | enum | `draft` | `standard` | `high`. Maps to provider native values. |
| `background` | enum | `transparent` or `opaque` (AR engines that support transparency). |
| `negative_prompt` | str | Optional negative prompt if provider supports it. |
| `extras` | object | Escape hatch for provider‑specific params (e.g., `{ "style": "vivid" }`). |

Generate‑only

| Field | Type | Description |
|---|---|---|
| `prompt` | str | Text description of the desired image. |

Edit‑only

| Field | Type | Description |
|---|---|---|
| `prompt` | str | Edit instruction. |
| `image`/`images[0]` | str | Base image (base64, data URL, or HTTP(S) URL). |
| `mask` | str | Optional mask; semantics vary by provider. |


Examples (Python via FastMCP Client)
------------------------------------

```python
import asyncio
from fastmcp import Client


async def main():
    async with Client("src/main.py") as client:
        # 1) Capabilities
        caps = await client.call_tool("get_model_capabilities", {"req": {}})
        print("capabilities:", caps.structured_content or caps.text)

        # 2) Generate (provider/model optional)
        gen = await client.call_tool(
            "generate_image",
            {
                "req": {
                    "prompt": "a watercolor fox in a forest, soft light",
                    # "provider": "openai",
                    # "model": "gpt-image-1",
                    "n": 1,
                    "size_code": "M",
                    "orientation": "square",
                    "quality": "standard",
                }
            },
        )
        # Structured JSON mirror (no image data, just metadata)
        print("generate:", gen.structured_content)
        # Traditional MCP content blocks include ImageContent for each image
        print("image blocks:", len(gen.content))

        # 3) Edit (no output_format field in schema)
        edit = await client.call_tool(
            "edit_image",
            {
                "req": {
                    "prompt": "add gentle sunbeams",
                    "image": "https://example.com/input.png",
                    # "mask": "data:image/png;base64,....",
                    # "provider": "openai",
                    # "model": "gpt-image-1",
                    "n": 1,
                    "size_code": "M",
                    "orientation": "square",
                }
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
| `USE_EMBEDDED_RESOURCES` | All | Optional; set to `true` for embedded resources, `false` (default) for resource URIs. |

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

- Lint: `ruff check .`  Format: `black .`
- Types: `pyright`
- Tests: `pytest -q`
