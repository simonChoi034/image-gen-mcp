from __future__ import annotations

import pytest

from image_gen_mcp.engines.ar.openrouter import OpenRouterAR
from image_gen_mcp.engines.diffusion.dalle_diffusion import DalleDiffusion
from image_gen_mcp.engines.diffusion.vertex_imagen import VertexImagen
from image_gen_mcp.exceptions import (
    ConfigurationError,
    ProviderError,
)
from image_gen_mcp.schema import ImageEditRequest, ImageGenerateRequest
from image_gen_mcp.shard.enums import Model, Provider


@pytest.mark.asyncio
async def test_provider_error_openrouter(monkeypatch):
    engine = OpenRouterAR(provider=Provider.OPENROUTER)

    class DummyClient:
        class images:
            @staticmethod
            async def generate(**kwargs):  # pragma: no cover - forced path
                raise RuntimeError("network down")

    monkeypatch.setattr(engine, "_client", lambda: DummyClient())

    req = ImageGenerateRequest(prompt="test", provider=Provider.OPENROUTER, model=Model.OPENROUTER_GOOGLE_GEMINI_IMAGE)
    with pytest.raises(ProviderError):
        await engine.generate(req)


@pytest.mark.asyncio
async def test_validation_error_vertex_edit_missing_image():
    # Missing images handled by pydantic before engine call
    with pytest.raises(Exception):  # Pydantic ValidationError
        ImageEditRequest(prompt="edit", images=[], provider=Provider.VERTEX, model=Model.IMAGEN_3_CAPABILITY)


@pytest.mark.asyncio
async def test_configuration_error_vertex_client(monkeypatch):
    engine = VertexImagen(provider=Provider.VERTEX)
    # Force settings missing by patching settings object attributes to None
    from image_gen_mcp.engines.diffusion import vertex_imagen as vi

    monkeypatch.setattr(vi.settings, "vertex_project", None)
    req = ImageGenerateRequest(prompt="x", provider=Provider.VERTEX, model=Model.IMAGEN_4_STANDARD)
    with pytest.raises(ConfigurationError):
        await engine.generate(req)


@pytest.mark.asyncio
async def test_no_images_generated_error(monkeypatch):
    engine = DalleDiffusion(provider=Provider.OPENAI)

    class DummyClient:
        class images:
            @staticmethod
            async def generate(**kwargs):
                class R:
                    data = []  # empty data triggers no images

                return R()

    monkeypatch.setattr(engine, "_get_client", lambda: DummyClient())
    req = ImageGenerateRequest(prompt="a", provider=Provider.OPENAI, model=Model.DALL_E_3)
    # Provider layer wraps NoImagesGeneratedError into ProviderError in catch-all
    from image_gen_mcp.exceptions import ProviderError

    with pytest.raises(ProviderError):
        await engine.generate(req)
