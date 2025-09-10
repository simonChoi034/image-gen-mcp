from __future__ import annotations

import importlib

from loguru import logger

from ..schema import CapabilityReport, Error, ImageResponse
from ..settings import get_settings, Settings
from ..shard import constants as C
from ..shard.enums import Model, Provider
from ..utils.error_helpers import augment_with_capability_tip, EngineResolutionError, ProviderUnavailableError
from .base_engine import ImageEngine

# Model-engine mapping constants

# Direct model-to-engine mappings
MODEL_ENGINE_MAP: dict[Model, type[ImageEngine] | str] = {
    # AR Models - store import paths to avoid circular imports at module import time
    Model.GPT_IMAGE_1: "image_gen_mcp.engines.ar.openai.OpenAIAR",
    Model.GEMINI_IMAGE_PREVIEW: "image_gen_mcp.engines.ar.gemini.GeminiAR",
    Model.OPENROUTER_GOOGLE_GEMINI_IMAGE: "image_gen_mcp.engines.ar.openrouter.OpenRouterAR",
    # Diffusion Models
    Model.DALL_E_3: "image_gen_mcp.engines.diffusion.dalle_diffusion.DalleDiffusion",
    Model.IMAGEN_4_STANDARD: "image_gen_mcp.engines.diffusion.vertex_imagen.VertexImagen",
    Model.IMAGEN_4_FAST: "image_gen_mcp.engines.diffusion.vertex_imagen.VertexImagen",
    Model.IMAGEN_4_ULTRA: "image_gen_mcp.engines.diffusion.vertex_imagen.VertexImagen",
    Model.IMAGEN_3_GENERATE: "image_gen_mcp.engines.diffusion.vertex_imagen.VertexImagen",
    Model.IMAGEN_3_CAPABILITY: "image_gen_mcp.engines.diffusion.vertex_imagen.VertexImagen",
}

# Provider-to-default-engine mappings (when model is not specified)
PROVIDER_DEFAULT_ENGINE_MAP: dict[Provider, type[ImageEngine] | str] = {
    Provider.OPENAI: "image_gen_mcp.engines.ar.openai.OpenAIAR",
    Provider.AZURE_OPENAI: "image_gen_mcp.engines.ar.openai.OpenAIAR",
    Provider.GEMINI: "image_gen_mcp.engines.ar.gemini.GeminiAR",
    Provider.VERTEX: "image_gen_mcp.engines.ar.gemini.GeminiAR",
    Provider.OPENROUTER: "image_gen_mcp.engines.ar.openrouter.OpenRouterAR",
}

# Provider-to-supported-models mappings
PROVIDER_MODELS_MAP: dict[Provider, list[Model]] = {
    Provider.OPENAI: [Model.GPT_IMAGE_1, Model.DALL_E_3],
    Provider.AZURE_OPENAI: [Model.GPT_IMAGE_1, Model.DALL_E_3],
    Provider.VERTEX: [Model.IMAGEN_4_STANDARD, Model.IMAGEN_4_FAST, Model.IMAGEN_4_ULTRA, Model.IMAGEN_3_GENERATE, Model.IMAGEN_3_CAPABILITY, Model.GEMINI_IMAGE_PREVIEW],
    Provider.GEMINI: [Model.GEMINI_IMAGE_PREVIEW],
    Provider.OPENROUTER: [Model.OPENROUTER_GOOGLE_GEMINI_IMAGE],
}


# Model capability definitions

# Models that support image editing
EDIT_CAPABLE_MODELS: set[Model] = {
    Model.GPT_IMAGE_1,  # OpenAI AR
    Model.GEMINI_IMAGE_PREVIEW,  # Gemini AR (maskless)
    Model.OPENROUTER_GOOGLE_GEMINI_IMAGE,  # OpenRouter Gemini AR (maskless)
    Model.IMAGEN_3_CAPABILITY,  # Only Imagen model that supports editing
}

# Models that support masking during editing
MASK_CAPABLE_MODELS: set[Model] = {
    Model.GPT_IMAGE_1,  # OpenAI AR
    Model.IMAGEN_3_CAPABILITY,  # Imagen with mask config support
}

# Models that only support generation (no editing)
GENERATION_ONLY_MODELS: set[Model] = {
    Model.DALL_E_3,
    Model.IMAGEN_4_STANDARD,
    Model.IMAGEN_4_FAST,
    Model.IMAGEN_4_ULTRA,
    Model.IMAGEN_3_GENERATE,
}


# Capability validation functions


def _model_supports_editing(model: Model) -> bool:
    """Check if a model supports image editing operations."""
    return model in EDIT_CAPABLE_MODELS


def _model_supports_masking(model: Model) -> bool:
    """Check if a model supports masking during editing."""
    return model in MASK_CAPABLE_MODELS


def _validate_edit_capability(model: Model) -> Error | None:
    """Validate if a model supports editing. Returns Error if not supported, None if valid."""
    if not _model_supports_editing(model):
        return Error(code="unsupported_operation", message=f"Model {model.value} does not support image editing. Only {', '.join(m.value for m in EDIT_CAPABLE_MODELS)} support editing.")
    return None


def _validate_mask_capability(model: Model, has_mask: bool) -> Error | None:
    """Validate if a model supports masking when a mask is provided. Returns Error if not supported, None if valid."""
    if has_mask and not _model_supports_masking(model):
        return Error(code="unsupported_operation", message=f"Model {model.value} does not support masking. Models with mask support: {', '.join(m.value for m in MASK_CAPABLE_MODELS)}")
    return None


# Internal resolution & validation logic


def _get_supported_providers_for_model(model: Model) -> list[Provider]:
    """Get list of providers that support a model."""
    return [provider for provider, models in PROVIDER_MODELS_MAP.items() if model in models]


def _load_engine_class(path_or_cls: type[ImageEngine] | str) -> type[ImageEngine]:
    """Resolve an engine class from either a direct class or an import path string.

    If a string is provided, import and return the class. Cache back into
    the mapping callers supply to avoid repeated imports.
    """
    if isinstance(path_or_cls, str):
        module_path, class_name = path_or_cls.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    return path_or_cls


def _create_resolution_error_message(provider: Provider | None, model: Model | None) -> str:
    """Create a helpful error message for resolution failures."""
    parts = []
    if provider:
        parts.append(f"provider={provider.value}")
    if model:
        parts.append(f"model={model.value}")
    hint = " " + ", ".join(parts) if parts else " for the given inputs"
    message = f"No engine available{hint}"

    if model and not provider:
        supported_providers = _get_supported_providers_for_model(model)
        if supported_providers:
            names = [p.value for p in supported_providers]
            message += f". Model {model.value} is supported by: {', '.join(names)}"
    elif provider and not model:
        supported_models = PROVIDER_MODELS_MAP.get(provider, [])
        if supported_models:
            names = [m.value for m in supported_models]
            message += f". Provider {provider.value} supports: {', '.join(names)}"
    elif model and provider:
        supported_providers = _get_supported_providers_for_model(model)
        names = [p.value for p in supported_providers]
        message += f". Model {model.value} is not supported by {provider.value}. Supported providers: {', '.join(names)}"

    return message


def _resolve_engine_spec(provider: Provider | None, model: Model | None) -> tuple[type[ImageEngine], Provider]:
    """
    Determines the engine class and the effective provider to use.
    Raises `EngineResolutionError` if no suitable engine can be found.
    """
    # Enforce that both provider and model must be specified.
    if not provider or not model:
        raise EngineResolutionError("Both 'provider' and 'model' must be specified to resolve an engine.")

    engine_class = MODEL_ENGINE_MAP.get(model)
    if not engine_class:
        raise EngineResolutionError(_create_resolution_error_message(provider, model))

    # Resolve lazy import if needed
    engine_class = _load_engine_class(engine_class)

    supported_providers = _get_supported_providers_for_model(model)
    if provider not in supported_providers:
        raise EngineResolutionError(_create_resolution_error_message(provider, model))

    # If we reach here, both are valid and compatible.
    return engine_class, provider


def _get_enabled_providers(settings: Settings = get_settings()) -> dict[Provider, bool]:
    """Get mapping of providers to their enabled status based on credentials."""
    return {
        Provider.OPENAI: settings.use_openai,
        Provider.AZURE_OPENAI: settings.use_azure_openai,
        Provider.GEMINI: settings.use_gemini,
        Provider.OPENROUTER: settings.use_openrouter,
        Provider.VERTEX: settings.use_vertex,
    }


# Public API - ModelFactory


class ModelFactory:
    """
    Provides a clean, maintainable interface for creating image engines
    based on model and provider specifications.
    """

    @classmethod
    def create(cls, provider: Provider | None = None, model: Model | None = None) -> ImageEngine:
        """
        Create an engine instance for the given model/provider.

        Raises:
            EngineResolutionError: If no suitable engine can be found.
            ProviderUnavailableError: If the required provider is not enabled.
        """
        engine_class, effective_provider = _resolve_engine_spec(provider, model)

        if not cls.is_provider_enabled(effective_provider):
            raise ProviderUnavailableError(effective_provider)

        return engine_class(provider=effective_provider)

    @classmethod
    def validate_and_create(cls, provider: Provider, model: Model) -> tuple[ImageEngine | None, ImageResponse | None]:
        """
        Validate and create an engine, returning either the engine or a formatted error response.
        """
        try:
            engine = cls.create(provider=provider, model=model)
            return engine, None
        except ProviderUnavailableError as e:
            err = Error(
                code=C.ERROR_CODE_PROVIDER_UNAVAILABLE,
                message=augment_with_capability_tip(str(e)),
            )
            return None, ImageResponse(ok=False, content=[], model=model, error=err)
        except EngineResolutionError as e:
            err = Error(code="validation_error", message=augment_with_capability_tip(str(e)))
            return None, ImageResponse(ok=False, content=[], model=model, error=err)

    @classmethod
    def get_supported_models(cls, provider: Provider | None = None) -> list[Model]:
        """Get all supported models, optionally filtered by provider."""
        if provider:
            return PROVIDER_MODELS_MAP.get(provider, [])
        return list(MODEL_ENGINE_MAP.keys())

    @classmethod
    def get_supported_providers(cls, model: Model | None = None) -> list[Provider]:
        """Get all supported providers, optionally filtered by model."""
        if model:
            return _get_supported_providers_for_model(model)
        return list(PROVIDER_DEFAULT_ENGINE_MAP.keys())

    @classmethod
    def get_default_engine_class(cls, provider: Provider) -> type[ImageEngine] | None:
        """Get the engine class for a given provider."""
        val = PROVIDER_DEFAULT_ENGINE_MAP.get(provider)
        if not val:
            return None
        return _load_engine_class(val)

    @classmethod
    def get_capabilities_for_provider(cls, provider: Provider) -> list[CapabilityReport]:
        """Get capabilities for a single enabled provider.

        This will attempt to instantiate every distinct engine class referenced by
        models that the provider advertises (via `PROVIDER_MODELS_MAP`) and
        collect each engine's `CapabilityReport`. This ensures we expose multiple
        engine families (e.g., AR Gemini + Imagen) under the same provider when
        appropriate.
        """
        if not cls.get_enabled_providers().get(provider):
            logger.warning(f"Provider {provider.value} is not enabled, skipping capabilities.")
            return []

        reports: list[CapabilityReport] = []

        # Determine all models this provider claims to support and resolve the
        # associated engine classes. Use a set to avoid duplicate engine classes.
        models = PROVIDER_MODELS_MAP.get(provider, [])
        engine_paths: set[str | type[ImageEngine]] = set()
        for m in models:
            val = MODEL_ENGINE_MAP.get(m)
            if val:
                engine_paths.add(val)

        # If no specific models are listed, fall back to the provider's default
        # engine class (keeps previous behavior intact).
        if not engine_paths:
            default_cls = cls.get_default_engine_class(provider)
            if default_cls:
                engine_paths.add(default_cls)

        for path_or_cls in engine_paths:
            try:
                engine_cls = _load_engine_class(path_or_cls)
                engine = engine_cls(provider=provider)
                report = engine.get_capability_report()
                reports.append(report)
            except Exception as e:
                logger.warning(f"Failed to get capabilities for engine {path_or_cls} under provider {provider.value}: {e}")
                continue

        return reports

    @classmethod
    def is_combination_supported(cls, provider: Provider, model: Model) -> bool:
        """Validate if a provider/model combination is supported."""
        return model in PROVIDER_MODELS_MAP.get(provider, [])

    @classmethod
    def get_enabled_providers(cls) -> dict[Provider, bool]:
        """Get mapping of providers to their enabled status."""
        return _get_enabled_providers()

    @classmethod
    def is_provider_enabled(cls, provider: Provider) -> bool:
        """Check if a provider is enabled (has required credentials)."""
        return _get_enabled_providers().get(provider, False)

    @classmethod
    def validate_edit_request(cls, model: Model, has_mask: bool = False) -> Error | None:
        """
        Validate if a model supports editing operations and optional masking.
        Returns Error if validation fails, None if valid.
        """
        # Check if model supports editing at all
        edit_error = _validate_edit_capability(model)
        if edit_error:
            return edit_error

        # Check masking capability if a mask is provided
        mask_error = _validate_mask_capability(model, has_mask)
        if mask_error:
            return mask_error

        return None

    @classmethod
    def model_supports_editing(cls, model: Model) -> bool:
        """Check if a model supports image editing operations."""
        return _model_supports_editing(model)

    @classmethod
    def model_supports_masking(cls, model: Model) -> bool:
        """Check if a model supports masking during editing."""
        return _model_supports_masking(model)


__all__ = ["ModelFactory"]
