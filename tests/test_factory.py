from __future__ import annotations

from unittest.mock import patch

import pytest

from image_gen_mcp.engines.ar.openai import OpenAIAR
from image_gen_mcp.engines.ar.openrouter import OpenRouterAR
from image_gen_mcp.engines.diffusion.vertex_imagen import VertexImagen
from image_gen_mcp.engines.factory import ModelFactory
from image_gen_mcp.shard.enums import Model, Provider


@patch.object(ModelFactory, "is_provider_enabled", return_value=True)
def test_create_by_model_openai_ar(mock_validate) -> None:
    engine = ModelFactory.create(model=Model.GPT_IMAGE_1, provider=Provider.OPENAI)
    assert isinstance(engine, OpenAIAR)
    assert engine.provider == Provider.OPENAI


@patch.object(ModelFactory, "is_provider_enabled", return_value=True)
def test_create_by_provider_vertex_defaults_imagen(mock_validate) -> None:
    engine = ModelFactory.create(provider=Provider.VERTEX, model=Model.IMAGEN_4_STANDARD)
    assert isinstance(engine, VertexImagen)
    assert engine.provider == Provider.VERTEX


def test_invalid_combination_raises() -> None:
    # Test that both provider and model are required
    with pytest.raises(Exception):  # Could be EngineResolutionError or ValueError
        ModelFactory.create(provider=Provider.GEMINI, model=Model.GPT_IMAGE_1)


@patch.object(ModelFactory, "is_provider_enabled", return_value=True)
def test_openrouter_defaults_to_openrouter_ar(mock_validate) -> None:
    engine = ModelFactory.create(provider=Provider.OPENROUTER, model=Model.OPENROUTER_GOOGLE_GEMINI_IMAGE)
    assert isinstance(engine, OpenRouterAR)
    assert engine.provider == Provider.OPENROUTER


@patch.object(ModelFactory, "is_provider_enabled", return_value=True)
def test_openrouter_model_maps_to_openrouter_ar(mock_validate) -> None:
    engine = ModelFactory.create(model=Model.OPENROUTER_GOOGLE_GEMINI_IMAGE, provider=Provider.OPENROUTER)
    assert isinstance(engine, OpenRouterAR)


def test_provider_validation_fails_appropriately() -> None:
    """Test that provider validation works correctly."""
    # Test with both provider and model but provider not enabled
    with pytest.raises(Exception, match="not enabled|unavailable"):
        ModelFactory.create(provider=Provider.OPENAI, model=Model.GPT_IMAGE_1)
