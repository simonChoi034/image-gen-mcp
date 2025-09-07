from __future__ import annotations

from unittest.mock import patch

import pytest

from src.engines.ar.openai import OpenAIAR
from src.engines.ar.openrouter import OpenRouterAR
from src.engines.diffusion.vertex_imagen import VertexImagen
from src.engines.factory import ModelFactory
from src.shard.enums import Model, Provider


@patch.object(ModelFactory, "_validate_provider_enabled", return_value=True)
def test_create_by_model_openai_ar(mock_validate) -> None:
    engine = ModelFactory.create(model=Model.GPT_IMAGE_1)
    assert isinstance(engine, OpenAIAR)
    # When model is specified, factory chooses a compatible provider
    assert engine.provider in {Provider.OPENAI, Provider.AZURE_OPENAI}


@patch.object(ModelFactory, "_validate_provider_enabled", return_value=True)
def test_create_by_provider_vertex_defaults_imagen(mock_validate) -> None:
    engine = ModelFactory.create(provider=Provider.VERTEX)
    assert isinstance(engine, VertexImagen)
    assert engine.provider == Provider.VERTEX


def test_invalid_combination_raises() -> None:
    # Gemini provider does not support GPT-Image-1 per PROVIDER_MODELS_MAP
    with pytest.raises(ValueError):
        ModelFactory.create(provider=Provider.GEMINI, model=Model.GPT_IMAGE_1)


@patch.object(ModelFactory, "_validate_provider_enabled", return_value=True)
def test_openrouter_defaults_to_openrouter_ar(mock_validate) -> None:
    engine = ModelFactory.create(provider=Provider.OPENROUTER)
    assert isinstance(engine, OpenRouterAR)
    assert engine.provider == Provider.OPENROUTER


@patch.object(ModelFactory, "_validate_provider_enabled", return_value=True)
def test_openrouter_model_maps_to_openrouter_ar(mock_validate) -> None:
    engine = ModelFactory.create(model=Model.OPENROUTER_GOOGLE_GEMINI_IMAGE)
    assert isinstance(engine, OpenRouterAR)


def test_provider_validation_fails_appropriately() -> None:
    """Test that provider validation works correctly."""
    # This will fail because no credentials are set in test environment
    with pytest.raises(ValueError, match="not enabled"):
        ModelFactory.create(provider=Provider.OPENAI)
