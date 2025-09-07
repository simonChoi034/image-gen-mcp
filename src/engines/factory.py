from __future__ import annotations

from ..schema import Error, ImageResponse
from ..settings import get_settings
from ..shard import constants as C
from ..shard.enums import Model, Provider
from ..utils.error_helpers import augment_with_capability_tip
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
MODEL_ENGINE_MAP = {
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
PROVIDER_DEFAULT_ENGINE_MAP = {
    Provider.OPENAI: OpenAIAR,  # Default to AR (GPT-Image-1) for OpenAI
    Provider.AZURE_OPENAI: OpenAIAR,  # Default to AR (GPT-Image-1) for Azure
    Provider.GEMINI: GeminiAR,
    Provider.VERTEX: VertexImagen,  # Default to Imagen for Vertex
    Provider.OPENROUTER: OpenRouterAR,  # OpenRouter Gemini image preview
}

# Provider-to-supported-models mappings
PROVIDER_MODELS_MAP = {
    Provider.OPENAI: [Model.GPT_IMAGE_1, Model.DALL_E_3],
    Provider.AZURE_OPENAI: [Model.GPT_IMAGE_1, Model.DALL_E_3],
    Provider.VERTEX: [Model.IMAGEN_4_STANDARD, Model.IMAGEN_3_GENERATE, Model.GEMINI_IMAGE_PREVIEW],
    Provider.GEMINI: [Model.GEMINI_IMAGE_PREVIEW],
    Provider.OPENROUTER: [Model.OPENROUTER_GOOGLE_GEMINI_IMAGE],
}


# ============================================================================
# MODEL FACTORY CLASS
# ============================================================================


class ModelFactory:
    """Resolve and construct the appropriate engine for a request.

    This factory provides a clean, maintainable interface for creating image engines
    based on model and provider specifications. Key features:

    - Direct model-to-engine mapping for unambiguous routing
    - Provider-based fallbacks when model is unspecified
    - Comprehensive validation with specific error types
    - Support for all implemented engines (AR and Diffusion)
    - Clear error messages for debugging unsupported combinations

    Architecture:
    - Uses enum-based mapping constants for maintainability
    - Modular validation and error handling methods
    - Consistent error response format
    - Comprehensive provider/model compatibility checking
    """

    # ========================================================================
    # PROVIDER VALIDATION
    # ========================================================================

    @staticmethod
    def _get_enabled_providers(settings=None) -> dict[Provider, bool]:
        """Get mapping of providers to their enabled status based on credentials."""
        if settings is None:
            settings = get_settings()

        return {
            Provider.OPENAI: settings.use_openai,
            Provider.AZURE_OPENAI: settings.use_azure_openai,
            Provider.GEMINI: settings.use_gemini,
            Provider.OPENROUTER: settings.use_openrouter,
            Provider.VERTEX: bool(settings.vertex_project and settings.vertex_location),
        }

    @classmethod
    def _validate_provider_enabled(cls, provider: Provider, settings=None) -> bool:
        """Check if a provider is enabled (has required credentials)."""
        enabled_providers = cls._get_enabled_providers(settings)
        return enabled_providers.get(provider, False)

    @staticmethod
    def _create_provider_unavailable_error(provider: Provider, model: Model) -> ValueError:
        """Create error for when provider is not enabled."""
        message = augment_with_capability_tip(f"Requested provider '{provider.value}' is not enabled (missing credentials).")
        return ValueError(message)

    @staticmethod
    def create_provider_unavailable_response(provider: Provider, model: Model) -> ImageResponse:
        """Create a standardized error response for unavailable providers."""
        err = Error(
            code=C.ERROR_CODE_PROVIDER_UNAVAILABLE,
            message=augment_with_capability_tip(f"Requested provider '{provider.value}' is not enabled (missing credentials)."),
        )
        return ImageResponse(ok=False, content=[], model=model, error=err)

    # ========================================================================
    # VALIDATION AND MAPPING
    # ========================================================================

    # --------------------------- validation helpers ----------------------- #
    @staticmethod
    def _validate_model_provider_compatibility(model: Model, provider: Provider) -> bool:
        """Check if a model is compatible with the given provider."""
        supported_models = PROVIDER_MODELS_MAP.get(provider, [])
        return model in supported_models

    @staticmethod
    def _is_model_supported_by_provider(model: Model, provider: Provider) -> bool:
        """Optimized check if a model is compatible with the given provider."""
        return model in PROVIDER_MODELS_MAP.get(provider, [])

    @staticmethod
    def _get_supported_models_for_provider(provider: Provider) -> list[Model]:
        """Get list of models supported by a provider."""
        return PROVIDER_MODELS_MAP.get(provider, [])

    @staticmethod
    def _get_supported_providers_for_model(model: Model) -> list[Provider]:
        """Get list of providers that support a model."""
        providers = []
        for provider, models in PROVIDER_MODELS_MAP.items():
            if model in models:
                providers.append(provider)
        return providers

    # --------------------------- engine resolution ------------------------- #
    @staticmethod
    def _resolve_engine_from_model(model: Model | None) -> type[ImageEngine] | None:
        """Return engine class for a specific model when known."""
        if model is None:
            return None
        return MODEL_ENGINE_MAP.get(model)

    @staticmethod
    def _resolve_engine_from_provider(provider: Provider | None) -> type[ImageEngine] | None:
        """Return a default engine class given only a provider hint."""
        if provider is None:
            return None
        return PROVIDER_DEFAULT_ENGINE_MAP.get(provider)

    # --------------------------- error handling --------------------------- #
    @staticmethod
    def _create_no_engine_error(provider: Provider | None, model: Model | None) -> ValueError:
        """Create error for when no suitable engine can be found."""
        parts = []
        if provider:
            parts.append(f"provider={provider.value}")
        if model:
            parts.append(f"model={model.value}")

        hint = " " + ", ".join(parts) if parts else " given inputs"
        message = f"No engine available for{hint}"

        # Add helpful suggestions
        if model and not provider:
            supported_providers = ModelFactory._get_supported_providers_for_model(model)
            if supported_providers:
                provider_names = [p.value for p in supported_providers]
                message += f". Model {model.value} is supported by: {', '.join(provider_names)}"
        elif provider and not model:
            supported_models = ModelFactory._get_supported_models_for_provider(provider)
            if supported_models:
                model_names = [m.value for m in supported_models]
                message += f". Provider {provider.value} supports: {', '.join(model_names)}"
        elif model and provider:
            if not ModelFactory._validate_model_provider_compatibility(model, provider):
                supported_providers = ModelFactory._get_supported_providers_for_model(model)
                provider_names = [p.value for p in supported_providers]
                message += f". Model {model.value} is not supported by {provider.value}. Supported providers: {', '.join(provider_names)}"

        return ValueError(message)

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    # ------------------------------ factory method ------------------------ #
    @classmethod
    def create(
        cls,
        provider: Provider | None = None,
        model: Model | None = None,
    ) -> ImageEngine:
        """Create an engine instance for the given model/provider.

        Resolution priority:
        1. Validate provider credentials (fail fast if not enabled)
        2. Direct model mapping (if model is specified and supported)
        3. Provider default mapping (if provider is specified)
        4. Raises ValueError with helpful suggestions

        Args:
            provider: Optional provider hint for engine selection
            model: Optional specific model for engine selection

        Returns:
            ImageEngine: Configured engine instance

        Raises:
            ValueError: When no suitable engine exists for the inputs,
                       or when provider is not enabled (missing credentials)
        """
        # Determine the provider to validate
        provider_to_validate = provider
        if provider_to_validate is None and model is not None:
            # If only model is specified, get the first supported provider
            supported_providers = cls._get_supported_providers_for_model(model)
            if supported_providers:
                provider_to_validate = supported_providers[0]

        # Early validation: fail fast if provider not enabled
        if provider_to_validate and not cls._validate_provider_enabled(provider_to_validate):
            raise cls._create_provider_unavailable_error(provider_to_validate, model or Model.GPT_IMAGE_1)

        # Try direct model mapping first (highest priority)
        if model is not None:
            engine_class = MODEL_ENGINE_MAP.get(model)
            if engine_class is not None:
                # If provider is also specified, validate compatibility
                if provider is not None and not cls._is_model_supported_by_provider(model, provider):
                    raise cls._create_no_engine_error(provider, model)
                # Use the specified provider, or determine from model
                engine_provider = provider or cls._get_supported_providers_for_model(model)[0]
                return engine_class(provider=engine_provider)

        # Fallback to provider default
        if provider is not None:
            engine_class = PROVIDER_DEFAULT_ENGINE_MAP.get(provider)
            if engine_class is not None:
                return engine_class(provider=provider)

        # No resolution possible
        raise cls._create_no_engine_error(provider, model)

    # --------------------------- utility methods --------------------------- #
    @classmethod
    def get_supported_models(cls, provider: Provider | None = None) -> list[Model]:
        """Get all supported models, optionally filtered by provider."""
        if provider is not None:
            return cls._get_supported_models_for_provider(provider)

        # Return all supported models
        return list(MODEL_ENGINE_MAP.keys())

    @classmethod
    def get_supported_providers(cls, model: Model | None = None) -> list[Provider]:
        """Get all supported providers, optionally filtered by model."""
        if model is not None:
            return cls._get_supported_providers_for_model(model)

        # Return all providers with default engines
        return list(PROVIDER_DEFAULT_ENGINE_MAP.keys())

    @classmethod
    def validate_combination(cls, provider: Provider, model: Model) -> bool:
        """Validate if a provider/model combination is supported."""
        return cls._validate_model_provider_compatibility(model, provider)

    @classmethod
    def get_enabled_providers(cls, settings=None) -> dict[Provider, bool]:
        """Get mapping of providers to their enabled status based on credentials."""
        return cls._get_enabled_providers(settings)

    @classmethod
    def validate_provider_enabled(cls, provider: Provider, settings=None) -> bool:
        """Check if a provider is enabled (has required credentials)."""
        return cls._validate_provider_enabled(provider, settings)

    @classmethod
    def validate_and_create(
        cls,
        provider: Provider | None = None,
        model: Model | None = None,
    ) -> tuple[ImageEngine | None, ImageResponse | None]:
        """Validate and create engine, returning either engine or error response.

        Returns:
            Tuple of (engine, error_response) where exactly one will be None.
        """
        try:
            engine = cls.create(provider=provider, model=model)
            return engine, None
        except ValueError as e:
            # Check if this is a provider unavailable error
            error_msg = str(e)
            if "not enabled" in error_msg and "missing credentials" in error_msg:
                # Extract provider from error or use the passed provider
                error_provider = provider or (cls._get_supported_providers_for_model(model)[0] if model else Provider.OPENAI)
                error_model = model or Model.GPT_IMAGE_1
                error_response = cls.create_provider_unavailable_response(error_provider, error_model)
                return None, error_response
            else:
                # Other validation errors - wrap in generic response
                error_model = model or Model.GPT_IMAGE_1
                err = Error(code="validation_error", message=augment_with_capability_tip(error_msg))
                error_response = ImageResponse(ok=False, content=[], model=error_model, error=err)
                return None, error_response


__all__ = ["ModelFactory"]
