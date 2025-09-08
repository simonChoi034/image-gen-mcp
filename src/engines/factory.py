from __future__ import annotations

from loguru import logger

from ..schema import CapabilityReport, Error, ImageResponse
from ..settings import get_settings, Settings
from ..shard import constants as C
from ..shard.enums import Model, Provider
from ..utils.error_helpers import augment_with_capability_tip, EngineResolutionError, ProviderUnavailableError
from .ar.gemini import GeminiAR
from .ar.openai import OpenAIAR
from .ar.openrouter import OpenRouterAR
from .base_engine import ImageEngine
from .diffusion.dalle_diffusion import DalleDiffusion
from .diffusion.vertex_imagen import VertexImagen

# ============================================================================
# MODEL-ENGINE MAPPING CONSTANTS
# ============================================================================

# Direct model-to-engine mappings
MODEL_ENGINE_MAP: dict[Model, type[ImageEngine]] = {
    # AR Models
    Model.GPT_IMAGE_1: OpenAIAR,
    Model.GEMINI_IMAGE_PREVIEW: GeminiAR,
    Model.OPENROUTER_GOOGLE_GEMINI_IMAGE: OpenRouterAR,
    # Diffusion Models
    Model.DALL_E_3: DalleDiffusion,
    Model.IMAGEN_4_STANDARD: VertexImagen,
    Model.IMAGEN_3_GENERATE: VertexImagen,
}

# Provider-to-default-engine mappings (when model is not specified)
PROVIDER_DEFAULT_ENGINE_MAP: dict[Provider, type[ImageEngine]] = {
    Provider.OPENAI: OpenAIAR,
    Provider.AZURE_OPENAI: OpenAIAR,
    Provider.GEMINI: GeminiAR,
    Provider.VERTEX: VertexImagen,
    Provider.OPENROUTER: OpenRouterAR,
}

# Provider-to-supported-models mappings
PROVIDER_MODELS_MAP: dict[Provider, list[Model]] = {
    Provider.OPENAI: [Model.GPT_IMAGE_1, Model.DALL_E_3],
    Provider.AZURE_OPENAI: [Model.GPT_IMAGE_1, Model.DALL_E_3],
    Provider.VERTEX: [Model.IMAGEN_4_STANDARD, Model.IMAGEN_3_GENERATE, Model.GEMINI_IMAGE_PREVIEW],
    Provider.GEMINI: [Model.GEMINI_IMAGE_PREVIEW],
    Provider.OPENROUTER: [Model.OPENROUTER_GOOGLE_GEMINI_IMAGE],
}


# ============================================================================
# INTERNAL RESOLUTION & VALIDATION LOGIC
# ============================================================================


def _get_supported_providers_for_model(model: Model) -> list[Provider]:
    """Get list of providers that support a model."""
    return [provider for provider, models in PROVIDER_MODELS_MAP.items() if model in models]


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
        Provider.VERTEX: bool(settings.vertex_project and settings.vertex_location),
    }


# ============================================================================
# PUBLIC API - ModelFactory
# ============================================================================


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
        return PROVIDER_DEFAULT_ENGINE_MAP.get(provider)

    @classmethod
    def get_capabilities_for_provider(cls, provider: Provider) -> CapabilityReport | None:
        """Get capabilities for a single enabled provider."""
        if not cls.get_enabled_providers().get(provider):
            logger.warning(f"Provider {provider.value} is not enabled, skipping capabilities.")
            return None

        engine_class = cls.get_default_engine_class(provider)
        if not engine_class:
            logger.warning(f"No engine for provider {provider.value}, skipping capabilities.")
            return None

        try:
            engine = engine_class(provider=provider)
            return engine.get_capability_report()
        except Exception as e:
            logger.warning(f"Failed to get capabilities for {provider.value}: {e}")
            return None

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


__all__ = ["ModelFactory"]
