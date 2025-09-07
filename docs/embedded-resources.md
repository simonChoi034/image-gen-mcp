# Embedded Resources Configuration

The image generation MCP server now supports two modes for returning generated images:

## 1. Resource URIs (Default)

- **Environment variable**: `USE_EMBEDDED_RESOURCES=false` (default)
- **Behavior**: Images are stored in memory and returned as resource URIs like `image://abc123`
- **Pros**: Most efficient, smallest response size
- **Cons**: Requires separate resource fetch by the agent
- **Use case**: When you want the most efficient approach and don't mind separate fetches

## 2. Embedded Resources (New)

- **Environment variable**: `USE_EMBEDDED_RESOURCES=true`
- **Behavior**: Images are returned as embedded resources with blob data directly in the response
- **Pros**: No separate fetch required, immediate access to image data
- **Cons**: Larger response size (but still smaller than problematic data URLs)
- **Use case**: When you want immediate access to images without additional requests

## Configuration

Set the environment variable before starting the server:

```bash
# Enable embedded resources
export USE_EMBEDDED_RESOURCES=true

# Or disable embedded resources (default)
export USE_EMBEDDED_RESOURCES=false
```

## Response Formats

### Resource URI Response (Default)

```json
{
  "ok": true,
  "images": [
    {
      "url": null,
      "b64": null,
      "mime_type": "image/png",
      "storage_url": null,
      "resource_uri": "image://fa2b6ceef90b4cb89134205fe74aa1d2"
    }
  ],
  "model": "google/gemini-2.5-flash-image-preview",
  "meta": {
    "provider": "openrouter",
    "resources": true
  }
}
```

### Embedded Resource Response (New)

```json
{
  "content": [
    {
      "type": "resource",
      "resource": {
        "uri": "image://gen-1",
        "name": "image-gen-1.png",
        "mimeType": "image/png",
        "blob": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADU...",
        "description": "Generated image with embedded data"
      }
    }
  ],
  "model": "google/gemini-2.5-flash-image-preview",
  "meta": {
    "provider": "openrouter",
    "embedded_resources": true
  }
}
```

## Context Bloat Comparison

Based on testing with a small 1x1 pixel PNG image:

- **Original problematic format** (with data URLs): 439 bytes
- **Resource URI approach**: 264 bytes (39.9% smaller)
- **Embedded resource approach**: 400 bytes (8.9% smaller than original)

The embedded resources approach provides the best balance of immediate accessibility and reasonable size.

## Migration Guide

### For Agent Developers

- **Default behavior unchanged**: Existing agents continue to work as before
- **To use embedded resources**: Set `USE_EMBEDDED_RESOURCES=true` and update your agent to handle the `content` field with embedded resources
- **Backward compatibility**: The `images` field is still present but empty when embedded resources are used

### For Server Operators

- **No breaking changes**: Default behavior is unchanged
- **To enable**: Set the environment variable and restart the server
- **Resource endpoints**: Still available for backward compatibility even with embedded resources enabled
