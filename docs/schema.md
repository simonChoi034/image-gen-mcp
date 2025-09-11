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
  - `model`: `Model` (model that produced the images)
  - `image_count`: integer (number of images in the `content` blocks)
  - `images`: `ImageDescriptor[]` (each descriptor contains only `uri`, `name`, `mimeType` — no blobs)
  - `meta`: object (provider/runtime metadata and mapping notes)
  - `error?`: `Error` (optional; present when an error occurred)

Example envelope

```
{
  "structured_content": {
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
  Note: errors occurring during engine execution are surfaced by raising structured exceptions inside the server. FastMCP converts those into proper MCP ToolError responses; the `ImageToolStructured` is used for successful structured summaries and may include an `error` object when engines return a non-exceptional error payload (legacy or provider-specific diagnostic). Engines in this codebase prefer exceptions for error flows.

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

## Saved file paths and directory input

Newer server versions accept an optional `directory` parameter on both `generate_image` and
`edit_image` tool inputs. When provided, the server will attempt to save generated/edited images
to that directory (creating it if necessary). When omitted, the server creates a unique temporary
directory under the system tempdir (e.g., `/tmp` on UNIX-like systems) and saves images there.

Each image descriptor in the minimal `ImageToolStructured.images` array now includes an optional
`file_path` field with the absolute filesystem path where the image was saved (when saving
was successful). This field is intended for programmatic access by clients and agents; note that
the server does not include binary blobs in structured JSON — binaries remain in MCP `ImageContent`
blocks — `file_path` is informational and points to the on-disk copy the server created.

## Security & lifecycle notes

- The server will attempt to create the target directory using the calling user's permissions. If
  directory creation fails, the server falls back to a temporary directory and surfaces an error
  message in the `structured_content.error` or the `meta` field for observability. Note that when
  failures occur during engine execution the server typically raises an exception which the MCP
  tool wrapper converts to a `ToolError` for clients. In that case clients will receive an MCP
  error response rather than a successful envelope containing an `error` field.
- The server does not automatically remove saved files. Clients are responsible for cleanup and
  lifecycle management (e.g., move to long-term storage, delete after use). Consider providing a
  `directory` when you need a predictable, durable location.

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
