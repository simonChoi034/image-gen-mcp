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
  - `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT` (and optional `AZURE_OPENAI_API_VERSION`)
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

Endpoints:

- `generate_image(payload)` — generate images.
- `edit_image(payload)` — edit images with an optional mask.
- `get_model_capabilities(payload?)` — discover enabled engines and models. Optional payload: `{ provider?: "openai" | "azure_openai" | "vertex" | "gemini" | "openrouter" }`. Returns `{ ok: true, capabilities: CapabilityReport[] }` where each report includes `engine`, `provider`, `family`, `models`, and selected parameter limits.

Example:

```
// request
get_model_capabilities({ provider: "openai" })

// response (excerpt)
{
  ok: true,
  capabilities: [
    {
      engine: "ar:openai",
      provider: "openai",
      family: "ar",
      models: ["gpt-image-1"],
      response_formats: ["url", "base64"],
      sizes: ["1024x1024", "1024x1536", "1536x1024"]
    },
    {
      engine: "diffusion:openai",
      provider: "openai",
      family: "diffusion",
      models: ["dall-e-3"],
      sizes: ["1024x1024", "1024x1792", "1792x1024"]
    }
  ]
}
```

## 6. Routing Logic

The MCP server routing logic will determine the appropriate engine for each image generation request based on:

- Explicit client specification of engine or provider, if provided
- Default engine preferences configured per deployment or user profile
- Capability matching (e.g., diffusion engines for high-fidelity images, multimodal LLMs for conversational contexts)
- Load balancing and rate limit considerations

Routing decisions will be logged for auditing and debugging.

### Engine Resolution & Model Handling

Routing precedence for engine selection:

- If the request specifies `engine`, route there (if credentials exist).
- Else, infer from `model` when possible.
- Else, route to the first available credentialed provider.

OpenRouter is treated as OpenAI-compatible using its own API key.

Model handling:

- AR engines receive the model identifier directly.
- Diffusion engines map generic model IDs to provider-specific identifiers when needed.

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

The MCP server implements an abstract base class `BaseImageEngine` defining the core interface for all image generation engines. This class includes methods:

- `capabilities()`: returns supported features and parameters.
- `generate()`: handles image generation requests.
- `edit()`: supports image editing operations.

Canonical dataclasses such as `ImageRequest`, `ImageEditRequest`, and `ImageBatch` standardize request and response formats across engines. Standardized error classes unify error handling and reporting.

The codebase is organized with separate directories for Auto-regressive (AR) and Diffusion engines. Each provider resides in its own file within these directories. AR engines share a common LangGraph agent implementation (`agent_graph.py`), facilitating conversational and multi-turn interactions. Diffusion engines implement provider-specific logic and translation layers.

Engine registration and capability listing are managed in `registry.py`, while `selector.py` handles resolution of engine selection based on requested `engine` and optional `family` parameters, applying routing logic as specified.

Model handling differs by engine type: AR adapters pass the model identifier directly to the underlying API, whereas diffusion adapters use a translator mapping to convert generic model names to provider-specific identifiers.

Per-request overrides allow clients to specify engine and model preferences, which take precedence over environment or configuration defaults.

Testing employs contract tests to verify adapter compliance with the base interface, golden tests to validate output consistency, routing tests to ensure correct engine selection, and documentation tests to maintain correctness of usage examples and interface definitions.

### Proposed Folder Structure

```
image_gen_mcp/
  src/
    engines/
      base_engine.py
      registry.py
      ar/
        agent_graph.py
        openai_ar.py
        gemini_ar.py
      diffusion/
        translator.py
        openai_diffusion.py
        azure_diffusion.py
        vertex_imagen.py
    routing/
      selector.py
    mcp/
      tools.py
      schema.py
    config/
      env.py
    utils/
      http.py
      logging.py
      provenance.py
  tests/
    engines/
    routing/
    mcp/
```
