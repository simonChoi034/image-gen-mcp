## Schema Overview (v2)

This document defines the public contract for the image tools and the private types used internally by engines.

- Public requests are model‑agnostic and stable across providers.
- Tool outputs never include binary data in JSON; images are returned as MCP ImageContent blocks.
- Engines use private, blob‑carrying types that are not exposed to clients.

______________________________________________________________________

## Design Principles

- Minimal inputs: unified enums and a small set of fields.
- Deterministic mapping: adapters convert unified fields to provider‑native knobs.
  Schema aims for best-effort normalization; engines drop unsupported fields.
- JSON without blobs: structured outputs contain only metadata; images are in content blocks.
- Backward compatibility: legacy aliases accepted where safe (e.g., image_b64 → image).

______________________________________________________________________

## Public Requests

The following Pydantic v2 models shape the tool input payloads.

- ImageGenerateRequest
- ImageEditRequest

Shared enums (from `src/shard/enums.py`)

- Provider, Model, SizeCode, Orientation, Quality, Background

### ImageGenerateRequest

- prompt: string (required)
- provider: Provider
- model: Model
- n: int (default 1, provider‑clamped)
- size: SizeCode | null
- orientation: Orientation | null
- quality: Quality | null
- negative_prompt: string | null
- background: Background | null
- extras: object | null (provider‑specific knobs)

### ImageEditRequest

- prompt: string (required)
- images: string[] (HTTP(S) URL, data URL, or base64 string)
  - If the provider/model only supports a single input image, engines MUST use the first entry and ignore the rest.
  - Acceptable forms per item:
    - HTTP(S) URL (e.g., https://...)
    - Data URL (e.g., data:image/png;base64,....)
    - Bare base64 string (engine infers/sets mimeType where possible)
- mask: string | null (same accepted forms as `images` items)
- provider: Provider
- model: Model
- n, size, orientation, quality, negative_prompt, background, extras (same as generate)

______________________________________________________________________

## Tool Outputs

Image tools return two complementary channels.

1. MCP content blocks (runtime)

- One ImageContent per generated/edited image
- Renderable by any MCP client

2. Structured JSON (minimal, stable)

- ImageToolStructured
  - ok: boolean
  - model: Model
  - image_count: integer
  - images: ImageDescriptor[] (uri/name/mimeType only)
  - meta: object (provider/runtime metadata)
  - error?: Error
- ImageDescriptor
  - uri?: string (opaque, no data)
  - name?: string
  - mimeType?: string

Example envelope

```
{
  "structured_content": {
    "ok": true,
    "model": "gpt-image-1",
    "image_count": 1,
    "images": [
      { "uri": "image://abc123", "name": "image-abc123.png", "mimeType": "image/png" }
    ],
    "meta": { "provider": "openai", "n": 1 }
  },
  "content": [ ImageContent(...), ... ]
}
```

Error envelope

```
{
  "structured_content": {
    "ok": false,
    "model": "gpt-image-1",
    "image_count": 0,
    "images": [],
    "meta": {},
    "error": { "code": "generation_error", "message": "Failed to generate image: ..." }
  },
  "content": []
}
```

______________________________________________________________________

## Public Schemas Advertised

Tool input/output schemas surfaced to clients.

- generate_image: input → ImageGenerateRequest, output → ImageToolStructured
- edit_image: input → ImageEditRequest, output → ImageToolStructured
- get_model_capabilities: input → { provider?: string }, output → CapabilitiesResponse

Note: no blob‑carrying types appear in any output schema.

______________________________________________________________________

## Capability Discovery

`get_model_capabilities` returns a list of provider engines, with model‑specific fields that vary only where the model differs (e.g., supports_edit, supports_mask, supports_negative_prompt, max_n). This reflects the currently enabled environment (keys, Vertex project/location, etc.).

______________________________________________________________________

## Mapping & Normalization

Every engine adapter applies the same pipeline.

- Defaults: apply defaults when fields are omitted
- Enum conversion: map SizeCode, Orientation, Quality, Background to provider‑native knobs
- Clamping: enforce provider limits (e.g., force DALL·E‑3 to n=1)
- Dropping: ignore unsupported inputs.
- Image inputs: normalize any legacy `image`/`image_b64` fields into `images[0]`; when multiple images are provided to single‑image models, use `images[0]` and drop the rest.
- Observability: record mapping/drops in response.meta for audit/debug

This design keeps the public contract stable and shifts provider variability to the adapter layer.

______________________________________________________________________

## Errors

`Error` is a stable structure included in ImageToolStructured.

- code: string (machine‑readable)
- message: string (human‑readable, actionable)
- details?: object (optional provider/debug info)

______________________________________________________________________

## Internal Types (Engines Only)

Engines exchange a private `ImageResponse` containing embedded base64 resources.

- ok: bool
- content: ResourceContent[] (EmbeddedResource with base64 blob)
- model: Model
- meta: dict
- error?: Error

The server converts `ImageResponse` into:

- MCP ImageContent blocks (for result.content)
- ImageToolStructured (for result.structured_content)

Internal carriers (EmbeddedResource, ResourceContent) never appear in public outputs.

______________________________________________________________________

## Migration Notes

- Prior versions sometimes included base64 blobs inside structured JSON. This is no longer the case. Images are returned only as MCP ImageContent.
- Clients that previously parsed embedded resources should now read `result.content` for images and use `structured_content` for metadata.
- ImageEditRequest now uses a single `images: string[]` field. The former `image` and `image_b64` aliases are deprecated. Servers SHOULD continue to accept them and normalize into `images[0]` for backward compatibility.

______________________________________________________________________

## Versioning

- The public schema is versioned with the repo. Additive changes prefer optional fields.
- Breaking changes require a major version note and a migration entry here.
