# MCP Image Generation Server Engine Configuration

## Overview

This document reflects the credential-only configuration model for the MCP image generation server. Engines are enabled purely by the presence of valid credentials in environment variables. There are no configuration variables for default engine or default model.

## Supported Engine Types

The server supports two engine families:

- Diffusion: OpenAI DALL·E, Azure OpenAI Images, Vertex Imagen (3/4)
- Autoregressive (AR): OpenAI GPT‑Image‑1, Gemini 2.5 Flash Image (native)

### Engine IDs

Clients may pass an `engine` hint per request. Supported IDs:

- `openai` — OpenAI-compatible (supports diffusion and AR)
- `openrouter` — OpenAI-compatible via OpenRouter
- `azure` — Azure OpenAI Images
- `vertex` — Vertex Imagen and Gemini

If `engine` is omitted, the server infers the route from the requested `model` or falls back to the first available provider based on credentials.

## Environment Variables (Credentials Only)

| Variable                   | Required When      | Notes                                          |
| -------------------------- | ------------------ | ---------------------------------------------- |
| `OPENAI_API_KEY`           | Using OpenAI       | Enables OpenAI DALL·E 3 and GPT‑Image‑1.       |
| `AZURE_OPENAI_API_KEY`     | Using Azure OpenAI | Must be set with endpoint.                     |
| `AZURE_OPENAI_ENDPOINT`    | Using Azure OpenAI | Example: `https://<name>.openai.azure.com`.    |
| `AZURE_OPENAI_API_VERSION` | Optional           | Defaults to a recent stable version.           |
| `GEMINI_API_KEY`           | Using Gemini       | Optional if using ADC/workload identity.       |
| `VERTEX_PROJECT`           | Using Vertex AI    | GCP project id.                                |
| `VERTEX_LOCATION`          | Using Vertex AI    | GCP location (e.g., `us-central1`).            |
| `VERTEX_CREDENTIALS_PATH`  | Optional           | Path to service account JSON if not using ADC. |
| `OPENROUTER_API_KEY`       | Using OpenRouter   | Enables OpenRouter (OpenAI-compatible).        |

## Behavior

- Engine enablement is automatic when credentials are present.
- The per-request `engine` field is optional; when absent, routing uses the `model` or the first available provider.
- AR models are passed through to the provider; diffusion models may be translated by the adapter when needed.

## Capabilities Discovery

- `get_model_capabilities` tool: returns enabled engines and their supported models based on current credentials. Clients should call this endpoint to discover available options at runtime.
- `generate_image` and `edit_image` schemas reflect normalized parameters; unsupported knobs for a selected model are ignored or rejected with validation errors.

## Unified Request/Response Schema

This section standardizes inputs and outputs across OpenAI (DALL·E 3, GPT‑Image‑1), Azure OpenAI Images, Vertex Imagen (3/4), and Gemini 2.5 Flash Image (native image generation/editing).

### Request (Generate)

- `prompt`: Required. Text description.
- `engine`: Optional. One of `openai` | `openrouter` | `azure` | `vertex`.
- `model`: Optional. For example, `dall-e-3`, `gpt-image-1`, `imagen-4.0-generate-001`, `gemini-2.5-flash-image-preview`.
- `n`: Optional. Count of images. Defaults to `1`. Provider-specific limits apply.
- Sizing & quality knobs are model-specific (see below).

### Response (Generate)

- `ok`: boolean
- `images`: list of `{ url? , b64? , mime_type, storage_url? }`
- `engine`: engine identifier used
- `model`: model identifier used
- `meta`: additional provider-normalized details
- `error?`: normalized error if `ok=false`

### Request (Edit)

- `prompt`: Required. Text description of edit.
- `image` or `images[0]`: Required. Base image.
- `mask`: Optional. Semantics vary by provider.
- Other knobs mirror the generate request but are model-specific.

### Response (Edit)

- Same structure as generate.

## Model-Specific Notes (Quick Reference)

OpenAI DALL·E 3 (diffusion)

- Sizes: `1024x1024`, `1024x1792`, `1792x1024`.
- Style: `vivid|natural`.
- Count: `n=1` enforced.
- Response: `url` or `b64_json`.
- Edits: Use GPT‑Image‑1 (DALL·E 3 edits not exposed here).

OpenAI GPT‑Image‑1 (AR)

- Size: `1024x1024`, `1024x1536`, `1536x1024`.
- Output mime: `png|jpeg|webp`.
- Background: `transparent` supported for PNG/WebP.
- Quality: commonly `low|medium|high` (SDKs vary); response always base64.
- Edits: `image`, optional `mask` + `prompt`. One image at a time.

Azure OpenAI Images

- Mirrors OpenAI request shapes; GPT‑Image‑1 returns base64; DALL·E 3 can return `b64_json`.

Vertex Imagen (3/4)

NOTE: Imagen image editing is only supported by the `imagen-3.0-capability-001` model. Most other Imagen models do not support image editing.

- Count: `sampleCount` 1–4.
- Aspect ratio: `1:1`, `3:4`, `4:3`, `9:16`, `16:9`.
- Output options: `mimeType` + `compressionQuality`.
- Safety/controls: `personGeneration`, `safetySetting/safetyFilterLevel`, `seed`, `addWatermark`, `enhancePrompt`, `negativePrompt`.
- Response: base64 image bytes (`bytesBase64Encoded`).
- Edits: `referenceImages` with `RAW` base image and optional `MASK` with `maskImageConfig` (`maskMode`, `dilation`).

Model support notes:

- Imagen image editing: only `imagen-3.0-capability-001` supports image editing operations. Other Imagen models listed below are generation-only and do not support editing.
- Imagen generation models supported by this server:
  - `imagen-4.0-generate-001`
  - `imagen-4.0-fast-generate-001`
  - `imagen-4.0-ultra-generate-001`
  - `imagen-3.0-capability-001` (also supports editing)

Gemini 2.5 Flash Image (native)

- Conversational image generation/editing via `generate_content` with text and image parts.
- No explicit `size`/`aspect_ratio` params; express such constraints in the prompt. For precise control, use Imagen via Gemini API (`imagen‑4.0‑*`).
- Output: inline base64 parts.

## Engine Resolution & Model Handling

Routing rules:

- If `engine` is provided, route to that provider (if credentials exist).
- Otherwise, infer from `model` when possible (e.g., `dall-e-*` → OpenAI/Azure; `imagen-*` → Vertex; `gemini-*` → Gemini/Vertex).
- If neither is conclusive, use the first available enabled provider.

OpenRouter is treated as OpenAI-compatible using its own API key. AR models are passed through; diffusion models are translated by adapters when necessary. Unsupported parameters may be ignored or rejected depending on model-specific validation.

## Example Configurations

### OpenAI (DALL·E 3 and GPT‑Image‑1)

```bash
export OPENAI_API_KEY="sk-..."
```

### Azure OpenAI Images

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<name>.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview" # optional
```

### Vertex Imagen and/or Gemini

```bash
export VERTEX_PROJECT="my-gcp-project"
export VERTEX_LOCATION="us-central1"
# optionally
export VERTEX_CREDENTIALS_PATH="/path/to/service-account.json"
export GEMINI_API_KEY="..." # when calling Gemini APIs directly
```

### OpenRouter (OpenAI-compatible)

```bash
export OPENROUTER_API_KEY="or-..."
```

## Trade-offs

This model prioritizes simplicity and maintainability:

- Simplicity: only credentials are configured; no app-level engine/model envs.
- Discoverability: clients query `get_model_capabilities` to know what’s available.
- Explicitness: per-request `engine` is supported but not required.

This design keeps runtime behavior aligned with the currently provided credentials.
