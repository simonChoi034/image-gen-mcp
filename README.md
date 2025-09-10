# 🎨 Image Gen MCP Server

> *"Fine. I'll do it myself."* — Thanos (and also me, after trying five different MCP servers that couldn't mix-and-match image models)  
> I wanted a single, **simple** MCP server that lets agents generate **and** edit images across OpenAI, Google (Gemini/Imagen), Azure, Vertex, and OpenRouter—without yak‑shaving. So… here it is.

[![PyPI version](https://img.shields.io/pypi/v/image-gen-mcp.svg)](https://pypi.org/project/image-gen-mcp/) ![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue) ![license](https://img.shields.io/badge/license-Apache%202.0-blue)

A multi‑provider **Model Context Protocol** (MCP) server for image **generation** and **editing** with a unified, type‑safe API. It returns MCP `ImageContent` blocks plus compact structured JSON so your client can route, log, or inspect results cleanly.

> [!IMPORTANT]
> This `README.md` is the canonical reference for API, capabilities, and usage. Some `/docs` files may lag behind.

---

## 🗺️ Table of Contents

- [Why this exists](#-why-this-exists)
- [Features](#-features)
- [Quick start (users)](#-quick-start-users)
- [Quick start (developers)](#-quick-start-developers)
- [Configure `mcp.json`](#-configure-mcpjson)
- [Tools API](#-tools-api)
  - [`generate_image`](#-generate_image)
  - [`edit_image`](#-edit_image)
  - [`get_model_capabilities`](#-get_model_capabilities)
- [Providers & Models](#-providers--models)
- [Python client example](#-python-client-example)
- [Environment Variables](#-environment-variables)
- [Running via FastMCP CLI](#-running-via-fastmcp-cli)
- [Troubleshooting & FAQ](#-troubleshooting--faq)
- [Contributing & Releases](#-contributing--releases)
- [License](#-license)

---

## 🧠 Why this exists

Because I couldn’t find an MCP server that spoke **multiple image providers** with **one sane schema**. Some only generated, some only edited, some required summoning three different CLIs at midnight.  
This one prioritizes:

- **One schema** across providers (AR & diffusion)
- **Minimal setup** (`uvx` or `pip`, drop a `mcp.json`, done)
- **Type‑safe I/O** with clear error shapes
- **Discoverability**: ask the server what models are live via `get_model_capabilities`

---

## ✨ Features

- **Unified tools**: `generate_image`, `edit_image`, `get_model_capabilities`
- **Providers**: OpenAI, Azure OpenAI, Google **Gemini**, **Vertex AI** (Imagen & Gemini), OpenRouter
- **Output**: MCP `ImageContent` blocks + small JSON metadata
- **Quality/size/orientation** normalization
- **Masking** support where engines allow it
- **Fail‑soft** errors with stable shape: `{ code, message, details? }`

---

## 🚀 Quick start (users)

Install and use as a published package.

```bash
# With uv (recommended)
uv add image-gen-mcp

# Or with pip
pip install image-gen-mcp
```

Then configure your MCP client.

### Configure `mcp.json`

Use `uvx` to run in an isolated env with correct deps:

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

### First call

```json
{
  "tool": "generate_image",
  "params": {
    "prompt": "A vibrant painting of a fox in a sunflower field",
    "provider": "openai",
    "model": "gpt-image-1"
  }
}
```

---

## 🧑‍💻 Quick start (developers)

Run from source for local development or contributions.

**Prereqs**
- Python **3.12+**
- `uv` (recommended)

**Install deps**

```bash
uv sync --all-extras --dev
```

**Environment**

```bash
cp .env.example .env
# Add your keys
```

**Run the server**

```bash
# stdio (direct)
python -m image_gen_mcp.main

# via FastMCP CLI
fastmcp run image_gen_mcp/main.py:app
```

### Local VS Code `mcp.json` for testing

If you use a VS Code extension or local tooling that reads `.vscode/mcp.json`, here's a safe example to run the local server (do NOT commit secrets):

```json
{
  "servers": {
    "image-gen-mcp": {
      "command": "python",
      "args": ["-m", "image_gen_mcp.main"],
      "env": {
        "# NOTE": "Replace with your local keys for testing; do not commit.",
        "OPENROUTER_API_KEY": "__REPLACE_WITH_YOUR_KEY__"
      }
    }
  },
  "inputs": []
}
```

Use this to run the server from your workspace instead of installing the package from PyPI. For CI or shared repos, store secrets in the environment or a secret manager and avoid checking them into git.

**Dev tasks**

```bash
uv run pytest -v
uv run ruff check .
uv run black --check .
uv run pyright
```

---

## 🧰 Tools API

All tools take **named parameters**. Outputs include structured JSON (for metadata/errors) and MCP `ImageContent` blocks (for actual images).

### `generate_image`

Create one or more images from a text prompt.

**Example**

```json
{
  "prompt": "A vibrant painting of a fox in a sunflower field",
  "provider": "openai",
  "model": "gpt-image-1",
  "n": 2,
  "size": "M",
  "orientation": "landscape"
}
```

**Parameters**

| Field | Type | Description |
|---|---|---|
| `prompt` | str | **Required.** Text description. |
| `provider` | enum | **Required.** `openai` \| `openrouter` \| `azure` \| `vertex` \| `gemini`. |
| `model` | enum | **Required.** Model id (see matrix). |
| `n` | int | Optional. Default 1; provider limits apply. |
| `size` | enum | Optional. `S` \| `M` \| `L`. |
| `orientation` | enum | Optional. `square` \| `portrait` \| `landscape`. |
| `quality` | enum | Optional. `draft` \| `standard` \| `high`. |
| `background` | enum | Optional. `transparent` \| `opaque` (when supported). |
| `negative_prompt` | str | Optional. Used when provider supports it. |
| `directory` | str | Optional. Filesystem directory where the server should save generated images. If omitted a unique temp directory is used. |

---

### `edit_image`

Edit an image with a prompt and optional mask.

**Example**

```json
{
  "prompt": "Remove the background and make the subject wear a red scarf",
  "provider": "openai",
  "model": "gpt-image-1",
  "images": ["data:image/png;base64,..."],
  "mask": null
}
```

**Parameters**

| Field | Type | Description |
|---|---|---|
| `prompt` | str | **Required.** Edit instruction. |
| `images` | list&lt;str&gt; | **Required.** One or more source images (base64, data URL, or https URL). Most models use only the first image. |
| `mask` | str | Optional. Mask as base64/data URL/https URL. |
| `provider` | enum | **Required.** See above. |
| `model` | enum | **Required.** Model id (see matrix). |
| `n` | int | Optional. Default 1; provider limits apply. |
| `size` | enum | Optional. `S` \| `M` \| `L`. |
| `orientation` | enum | Optional. `square` \| `portrait` \| `landscape`. |
| `quality` | enum | Optional. `draft` \| `standard` \| `high`. |
| `background` | enum | Optional. `transparent` \| `opaque`. |
| `negative_prompt` | str | Optional. Negative prompt. |
| `directory` | str | Optional. Filesystem directory where the server should save edited images. If omitted a unique temp directory is used. |

---

### `get_model_capabilities`

Discover which providers/models are **actually** enabled based on your environment.

**Example**

```json
{ "provider": "openai" }
```

Call with no params to list **all** enabled providers/models.

**Output**: a `CapabilitiesResponse` with providers, models, and features.

---

## 🧭 Providers & Models

Routing is handled by a `ModelFactory` that maps model → engine. A compact, curated list keeps things understandable.

### Model Matrix

| Model | Family | Providers | Generate | Edit | Mask |
|---|---|---|:---:|:---:|:---:|
| `gpt-image-1` | AR | `openai`, `azure` | ✅ | ✅ | ✅ (OpenAI/Azure) |
| `dall-e-3` | Diffusion | `openai`, `azure` | ✅ | ❌ | — |
| `gemini-2.5-flash-image-preview` | AR | `gemini`, `vertex` | ✅ | ✅ (maskless) | ❌ |
| `imagen-4.0-generate-001` | Diffusion | `vertex` | ✅ | ❌ | — |
| `imagen-3.0-generate-002` | Diffusion | `vertex` | ✅ | ❌ | — |
| `imagen-4.0-fast-generate-001` | Diffusion | `vertex` | ✅ | ❌ | — |
| `imagen-4.0-ultra-generate-001` | Diffusion | `vertex` | ✅ | ❌ | — |
| `imagen-3.0-capability-001` | Diffusion | `vertex` | ✅ | ✅ | ✅ (mask via mask config) |
| `google/gemini-2.5-flash-image-preview` | AR | `openrouter` | ✅ | ✅ (maskless) | ❌ |

### Provider Model Support

| Provider | Supported Models |
|---|---|
| `openai` | `gpt-image-1`, `dall-e-3` |
| `azure` | `gpt-image-1`, `dall-e-3` |
| `gemini` | `gemini-2.5-flash-image-preview` |
| `vertex` | `imagen-4.0-generate-001`, `imagen-3.0-generate-002`, `gemini-2.5-flash-image-preview` |
| `openrouter` | `google/gemini-2.5-flash-image-preview` |

---

## 🐍 Python client example

```python
import asyncio
from fastmcp import Client


async def main():
    # Assumes the server is running via: python -m image_gen_mcp.main
    async with Client("image_gen_mcp/main.py") as client:
        # 1) Capabilities
        caps = await client.call_tool("get_model_capabilities")
        print("Capabilities:", caps.structured_content or caps.text)

        # 2) Generate
        gen_result = await client.call_tool(
            "generate_image",
            {
                "prompt": "a watercolor fox in a forest, soft light",
                "provider": "openai",
                "model": "gpt-image-1",
            },
        )
        print("Generate Result:", gen_result.structured_content)
        print("Image blocks:", len(gen_result.content))


asyncio.run(main())
```

---

## 🔐 Environment variables

Set only what you need:

| Variable | Required for | Description |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI | API key for OpenAI. |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI | Azure OpenAI key. |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI | Azure endpoint URL. |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI | Optional; default `2024-02-15-preview`. |
| `GEMINI_API_KEY` | Gemini | Gemini Developer API key. |
| `OPENROUTER_API_KEY` | OpenRouter | OpenRouter API key. |
| `VERTEX_PROJECT` | Vertex AI | GCP project id. |
| `VERTEX_LOCATION` | Vertex AI | GCP region (e.g. `us-central1`). |
| `VERTEX_CREDENTIALS_PATH` | Vertex AI | Optional path to GCP JSON; ADC supported. |

---

## 🏃 Running via FastMCP CLI

Supports multiple transports:

- **stdio:** `fastmcp run image_gen_mcp/main.py:app`
- **SSE (HTTP):** `fastmcp run image_gen_mcp/main.py:app --transport sse --host 127.0.0.1 --port 8000`
- **HTTP:** `fastmcp run image_gen_mcp/main.py:app --transport http --host 127.0.0.1 --port 8000 --path /mcp`

**Design notes**

- **Schema:** public contract in `image_gen_mcp/schema.py` (Pydantic).
- **Engines:** modular adapters in `image_gen_mcp/engines/`, selected by `ModelFactory`.
- **Capabilities:** discovered dynamically via `image_gen_mcp/settings.py`.
- **Errors:** stable JSON error `{ code, message, details? }`.

---

## ⚠️ Testing remarks

I tested this project locally using the `openrouter`-backed model only. I could not access Gemini or OpenAI from my location (Hong Kong) due to regional restrictions — thanks, US government — so I couldn't fully exercise those providers.

Because of that limitation, the `gemini`/`vertex` and `openai` (including Azure) adapters may contain bugs or untested edge cases. If you use those providers and find issues, please open an issue or, even better, submit a pull request with a fix — contributions are welcome.

Suggested info to include when filing an issue:
- Your provider and model (e.g., `openai:gpt-image-1`, `vertex:imagen-4.0-generate-001`)
- Full stderr/server logs showing the error
- Minimal reproduction steps or a short test script

Thanks — and PRs welcome!

---

## 🤝 Contributing & Releases

PRs welcome! Please run tests and linters locally.

**Release process (GitHub Actions)**

1. **Automated (recommended)**
   - Actions → **Manual Release**
   - Pick version bump: patch / minor / major
   - The workflow tags, builds the changelog, and publishes to PyPI

2. **Manual**
   - `git tag vX.Y.Z`
   - `git push origin vX.Y.Z`
   - Create a GitHub Release from the tag

---

## 📄 License

Apache-2.0 — see `LICENSE`.
