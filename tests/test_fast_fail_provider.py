from __future__ import annotations

import asyncio

from image_gen_mcp.engines.factory import ModelFactory
from image_gen_mcp.main import mcp_edit_image, mcp_generate_image
from image_gen_mcp.shard.enums import Model, Provider


def test_provider_validation_helper(monkeypatch):
    """Test the provider validation helper directly."""
    # Ensure no credentials present
    for var in ["OPENAI_API_KEY", "AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "VERTEX_PROJECT", "VERTEX_LOCATION"]:
        monkeypatch.delenv(var, raising=False)

    # Test that validation properly detects missing providers
    assert not ModelFactory.is_provider_enabled(Provider.OPENAI)
    assert not ModelFactory.is_provider_enabled(Provider.AZURE_OPENAI)
    assert not ModelFactory.is_provider_enabled(Provider.GEMINI)


def test_generate_fast_fail_provider_unavailable(monkeypatch):
    # Ensure no credentials present
    for var in ["OPENAI_API_KEY", "AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "VERTEX_PROJECT", "VERTEX_LOCATION"]:
        monkeypatch.delenv(var, raising=False)

    # FastMCP tool wrapper stores original coroutine in _fn
    async def _run():
        return await mcp_generate_image.fn(  # type: ignore[attr-defined]
            prompt="test",
            provider=Provider.OPENAI,
            model=Model.GPT_IMAGE_1,
            n=1,
            size=None,
            orientation=None,
            quality=None,
            negative_prompt=None,
            background=None,
            extras=None,
            ctx=None,
        )

    result = asyncio.run(_run())
    structured = result.structured_content
    assert structured["ok"] is False
    assert structured["error"]["code"] == "provider_unavailable"
    assert "not enabled" in structured["error"]["message"].lower()


def test_edit_fast_fail_provider_unavailable(monkeypatch):
    # Ensure no credentials present
    for var in ["OPENAI_API_KEY", "AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "VERTEX_PROJECT", "VERTEX_LOCATION"]:
        monkeypatch.delenv(var, raising=False)

    # minimal 1x1 png base64
    img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB/9k3WQAAAABJRU5ErkJggg=="
    data_url = f"data:image/png;base64,{img_b64}"

    async def _run():
        return await mcp_edit_image.fn(  # type: ignore[attr-defined]
            prompt="edit",
            provider=Provider.OPENAI,
            model=Model.GPT_IMAGE_1,
            images=[data_url],
            mask=None,
            n=1,
            size=None,
            orientation=None,
            quality=None,
            negative_prompt=None,
            background=None,
            extras=None,
            ctx=None,
        )

    result = asyncio.run(_run())
    structured = result.structured_content
    assert structured["ok"] is False
    assert structured["error"]["code"] == "provider_unavailable"
    assert "not enabled" in structured["error"]["message"].lower()
