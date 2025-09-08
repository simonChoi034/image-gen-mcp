## Schema Overview (v2)

This document defines the public contract for the image tools and the private types used internally by engines.

- Public requests are model‑agnostic and stable across providers.
- Tool outputs never include binary data in JSON; images are returned as MCP ImageContent blocks.
- Engines use private, blob‑carrying types that are not exposed to clients.

______________________________________________________________________

Tool Outputs

Image tools return two complementary channels:

1. MCP content blocks (runtime)

- One ImageContent block per generated/edited image. These are the renderable image payloads delivered via FastMCP content streaming.

2. Structured JSON (minimal, stable)

- `ImageToolStructured` — a lightweight, programmatic summary that never contains binary blobs. Fields:
  - `ok`: boolean
  - `model`: `Model` (model that produced the images)
  - `image_count`: integer (number of images in the `content` blocks)
  - `images`: `ImageDescriptor[]` (each descriptor contains only `uri`, `name`, `mimeType` — no blobs)
  - `meta`: object (provider/runtime metadata and mapping notes)
  - `error?`: `Error` (present when `ok` is `false`)

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
  "content": [ /* ImageContent blocks - binary data is delivered here, not in structured JSON */ ]
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

Tool input/output schemas surfaced to clients. The input for `generate_image` and `edit_image` corresponds to the fields of their respective `Image...Request` models, passed as top-level named parameters.

- `generate_image`: input → named parameters from `ImageGenerateRequest`, output → `ImageToolStructured`
- `edit_image`: input → named parameters from `ImageEditRequest`, output → `ImageToolStructured`
- `get_model_capabilities`: input → `{ provider?: string }` (named parameter), output → `CapabilitiesResponse`

Note: no blob‑carrying types appear in any structured output schema; image binaries are carried in MCP `ImageContent` blocks.

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
