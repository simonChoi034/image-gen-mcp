# Image Generation Support Specification for MCP Server

## 1. Overview

This specification outlines the design and integration requirements for supporting two distinct categories of image generation within the MCP server. The goal is to enable seamless handling of both diffusion-based and auto-regressive multimodal large language model (LLM) engines, ensuring flexibility, scalability, and ease of extension. The MCP server will route image generation requests appropriately based on engine type and configuration, providing a unified interface to clients.

## 2. Supported Models & Providers

The MCP server will support a curated set of image generation models and providers, categorized as follows:

- **Diffusion-based Models:**

  - OpenAI DALL·E
  - Azure OpenAI Image Generation
  - Google Imagen via Vertex AI

- **Auto-regressive Multimodal LLMs:**

  - OpenAI GPT-Image-1
  - Google Gemini 2.5 Flash Image

These models represent leading approaches in image generation, covering both specialized diffusion techniques and integrated multimodal LLM capabilities.

## 3. Engine Types

### Diffusion-based

Diffusion models generate images by iteratively denoising a latent representation, guided by textual prompts. They typically produce high-fidelity, diverse images with fine control over style and content.

- **Examples:** OpenAI DALL·E, Azure OpenAI Image Generation, Google Imagen (via Vertex AI)
- **Characteristics:**
  - Require prompt-to-image pipelines
  - Often support multiple image sizes and variations
  - May have rate limits or cost considerations per request

### Auto-regressive Multimodal LLMs

These models extend large language models with multimodal capabilities, generating images as part of a broader generative framework. They can handle complex instructions and generate images alongside text.

- **Examples:** OpenAI GPT-Image-1, Google Gemini 2.5 Flash Image
- **Characteristics:**
  - Generate images as part of broader context understanding
  - Support conversational or multi-turn interactions
  - May require different API protocols and authentication

## 4. Environment Configuration

The MCP server uses credential-only environment variables:

- Provider credentials/endpoints:
  - `OPENAI_API_KEY`
  - `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` (and optional `AZURE_OPENAI_API_VERSION`)
  - `VERTEX_PROJECT`, `VERTEX_LOCATION` (optional `VERTEX_CREDENTIALS_PATH`)
  - `GEMINI_API_KEY`
  - `OPENROUTER_API_KEY`

There are no environment variables for default engine or default model. Engines become available when their credentials are present.

## 5. Tool Interface & Capabilities

The MCP server exposes a unified tool interface:

- Accepts prompts and model-specific options.
- Returns image URLs or base64 payloads plus normalized metadata.
- Reports actionable errors consistently across providers.
- Adds a discovery endpoint for enabled models per credentials.

Tools:

- `generate_image(...)` — generate images. Accepts named parameters corresponding to the fields in `ImageGenerateRequest`.
- `edit_image(...)` — edit images with an optional mask. Accepts named parameters corresponding to the fields in `ImageEditRequest`.
- `get_model_capabilities(provider?)` — discover enabled engines and models. Optional parameter: `provider`. Returns a `CapabilitiesResponse` JSON object with a top-level `capabilities` array.

Note: the codebase no longer relies on a top-level `ok` boolean in tool responses; successful discovery responses return `capabilities`. Engine/runtime errors are surfaced by raising structured exceptions which FastMCP converts into MCP `ToolError` responses for clients.

Example `get_model_capabilities` response:

```json
// request
// get_model_capabilities({ "provider": "openai" })

// response (structure based on src/schema.py)
{
  "capabilities": [
    {
      "provider": "openai",
      "family": "ar",
      "models": [
        {
          "model": "gpt-image-1",
          "supports_negative_prompt": null,
          "supports_background": true,
          "max_n": null,
          "supports_edit": true,
          "supports_mask": true
        }
      ]
    },
    {
      "provider": "openai",
      "family": "diffusion",
      "models": [
        {
          "model": "dall-e-3",
          "supports_negative_prompt": null,
          "supports_background": null,
          "max_n": 1,
          "supports_edit": false,
          "supports_mask": false
        }
      ]
    }
  ]
}
```

## 6. Routing Logic

The MCP server routing logic determines the appropriate engine for each image generation request based on the `provider` and `model` parameters supplied in the tool call.

### Engine Resolution & Model Handling

The `ModelFactory` is responsible for routing.

- The client must specify `provider` and `model` in the request.
- The factory validates that the requested `provider` has credentials and supports the requested `model`.
- If valid, it instantiates the correct engine adapter for the request.
- There is no fallback logic or default engine; requests must be specific.

Model handling:

- AR engines receive the model identifier directly.
- Diffusion engines may map generic model IDs to provider-specific identifiers.

## 7. Dynamic Engine Enablement

Engines are enabled and disabled by the presence or absence of credentials. This allows:

- Gradual rollout of new engines
- Temporary disabling due to outages or maintenance
- Experimentation and A/B testing of different providers

Engines appear only if their credentials are present, and discovery via `get_model_capabilities` reflects current availability.

## 8. Error Handling

Robust error handling is critical to maintain reliability:

- Detect and classify errors from provider APIs (e.g., authentication failures, rate limits, timeouts)
- Provide meaningful error messages to clients with actionable guidance
- Implement retry policies where appropriate
- Fall back to alternative engines if configured and available
- Log errors with context for operational monitoring and troubleshooting

Client-facing errors should be standardized regardless of underlying engine.

## 9. Future Extensions

The architecture will accommodate future enhancements including:

- Support for additional image generation models and emerging providers
- Integration of custom or on-premise image generation engines
- Advanced prompt engineering features and parameter tuning
- Multi-modal output support combining images with text, audio, or video
- Enhanced analytics and usage tracking per engine and provider
- User preference learning to optimize engine selection

This extensible design ensures the MCP server remains adaptable to evolving AI image generation technologies.

## 10. Implementation Guide

The MCP server implements an abstract base class `BaseImageEngine` defining the core interface for all image generation engines. This class includes methods for `generate` and `edit`.

Canonical Pydantic models such as `ImageGenerateRequest` and `ImageEditRequest` standardize request formats across engines.

The codebase is organized with separate directories for Auto-regressive (AR) and Diffusion engines inside `src/engines/`. Each provider has its own module within these directories.

Engine selection, validation, and capability listing are managed by the `ModelFactory` (`src/engines/factory.py`). It is the central routing component.

Model handling differs by engine type: AR adapters generally pass the model identifier directly to the underlying API, whereas diffusion adapters may use an internal mapping to convert generic model names to provider-specific identifiers.

Testing employs contract tests to verify adapter compliance with the base interface, golden tests to validate output consistency, and routing tests to ensure correct engine selection.

### Project Folder Structure

The actual folder structure is as follows:

```
src/
├── __init__.py
├── main.py
├── schema.py
├── settings.py
├── engines/
│   ├── __init__.py
│   ├── base_engine.py
│   ├── factory.py
│   ├── ar/
│   │   ├── gemini.py
│   │   ├── openai.py
│   │   └── openrouter.py
│   └── diffusion/
│       ├── dalle_diffusion.py
│       └── vertex_imagen.py
├── shard/
│   ├── constants.py
│   ├── enums.py
│   └── instructions.py
└── utils/
    ├── error_helpers.py
    ├── image_utils.py
    └── prompt.py
```
