from __future__ import annotations

import base64
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from image_gen_mcp.engines.diffusion.vertex_imagen import VertexImagen
from image_gen_mcp.schema import ImageEditRequest
from image_gen_mcp.shard.enums import Model, Provider


def _make_temp_png() -> str:
    """Create a temporary PNG file for testing."""
    # 1x1 transparent PNG
    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB/9k3WQAAAABJRU5ErkJggg=="
    data = base64.b64decode(b64)
    fd, path = tempfile.mkstemp(suffix=".png")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def test_vertex_imagen_capabilities():
    """Test that Vertex Imagen reports correct capabilities including edit support."""
    engine = VertexImagen(provider=Provider.VERTEX)
    caps = engine.get_capability_report()

    assert caps.provider == Provider.VERTEX
    assert len(caps.models) == 5  # Now we have 5 Imagen models

    # Check generation-only models don't support editing
    imagen_4 = next(m for m in caps.models if m.model == Model.IMAGEN_4_STANDARD)
    assert imagen_4.supports_edit is False
    assert imagen_4.supports_mask is False
    assert imagen_4.supports_negative_prompt is True
    assert imagen_4.max_n == 4

    # Check that only imagen-3.0-capability-001 supports editing
    imagen_3_capability = next(m for m in caps.models if m.model == Model.IMAGEN_3_CAPABILITY)
    assert imagen_3_capability.supports_edit is True
    assert imagen_3_capability.supports_mask is True
    assert imagen_3_capability.supports_negative_prompt is True
    assert imagen_3_capability.max_n == 4


def test_image_utils_inheritance():
    """Test that the engine properly inherits image utility methods."""
    engine = VertexImagen(provider=Provider.VERTEX)
    path = _make_temp_png()
    try:
        # Test read_image_bytes_and_mime method (inherited from base)
        image_bytes, mime = engine.read_image_bytes_and_mime(path)
        assert len(image_bytes) > 0
        assert mime == "image/png"
        # Test to_image_data_url method (inherited from base)
        data_url = engine.to_image_data_url(path)
        assert data_url.startswith("data:image/png;base64,")
    finally:
        os.remove(path)


@patch("google.oauth2.service_account.Credentials.from_service_account_file")
@patch("google.genai.Client")
@patch("PIL.Image.open")
async def test_edit_with_base_image(mock_pil_open, mock_client_class, mock_credentials):
    """Test edit method with base image (no mask)."""
    # Setup mocks
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_credentials.return_value = MagicMock()  # Mock service account credentials

    # Mock PIL image
    mock_pil_image = MagicMock()
    mock_pil_open.return_value = mock_pil_image

    # No need to mock types - let them work naturally

    # Mock response with generated_images
    mock_response = MagicMock()
    mock_generated_image = MagicMock()
    mock_generated_image.image.image_bytes = b"fake_image_bytes"
    mock_response.generated_images = [mock_generated_image]

    mock_client.aio.models.edit_image = AsyncMock(return_value=mock_response)

    # Create engine and test data
    engine = VertexImagen(provider=Provider.VERTEX)

    # Mock settings for client creation
    with patch("image_gen_mcp.engines.diffusion.vertex_imagen.settings") as mock_settings:
        mock_settings.vertex_project = "test-project"
        mock_settings.vertex_location = "us-central1"
        mock_settings.vertex_credentials_path = "/fake/path"

        # Create test image
        test_image = _make_temp_png()
        try:
            # Create edit request using model that supports editing
            req = ImageEditRequest(prompt="Edit this image", images=[test_image], provider=Provider.VERTEX, model=Model.IMAGEN_3_CAPABILITY, n=1)

            # Call edit method
            response = await engine.edit(req)

            # Verify response
            assert response.ok is True
            assert response.model == Model.IMAGEN_3_CAPABILITY
            assert len(response.content) == 1
            assert response.content[0].type == "resource"
            # Engine now returns PNG images (output_mime_type set to image/png)
            assert response.content[0].resource.mimeType == "image/png"

            # Verify SDK was called correctly
            mock_client.aio.models.edit_image.assert_called_once()

            # Verify the call arguments
            call_args = mock_client.aio.models.edit_image.call_args
            assert call_args[1]["model"] == str(Model.IMAGEN_3_CAPABILITY)
            assert call_args[1]["prompt"] == "Edit this image"
            # With fallback mask behavior, we now have 2 reference images: base + fallback mask
            assert len(call_args[1]["reference_images"]) == 2

        finally:
            os.remove(test_image)


@patch("google.oauth2.service_account.Credentials.from_service_account_file")
@patch("google.genai.Client")
@patch("PIL.Image.open")
async def test_edit_with_mask(mock_pil_open, mock_client_class, mock_credentials):
    """Test edit method with base image and mask."""
    # Setup mocks
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_credentials.return_value = MagicMock()  # Mock service account credentials

    # Mock PIL images
    mock_pil_image = MagicMock()
    mock_mask_image = MagicMock()
    mock_pil_open.side_effect = [mock_pil_image, mock_mask_image]

    # No need to mock types - let them work naturally

    # Mock response with generated_images
    mock_response = MagicMock()
    mock_generated_image = MagicMock()
    mock_generated_image.image.image_bytes = b"fake_image_bytes"
    mock_response.generated_images = [mock_generated_image]

    mock_client.aio.models.edit_image = AsyncMock(return_value=mock_response)

    # Create engine and test data
    engine = VertexImagen(provider=Provider.VERTEX)

    # Mock settings for client creation
    with patch("image_gen_mcp.engines.diffusion.vertex_imagen.settings") as mock_settings:
        mock_settings.vertex_project = "test-project"
        mock_settings.vertex_location = "us-central1"
        mock_settings.vertex_credentials_path = "/fake/path"

        # Create test images
        test_image = _make_temp_png()
        test_mask = _make_temp_png()
        try:
            # Create edit request with mask using model that supports editing and masking
            req = ImageEditRequest(prompt="Edit this image with mask", images=[test_image], mask=test_mask, provider=Provider.VERTEX, model=Model.IMAGEN_3_CAPABILITY, n=1)

            # Call edit method
            response = await engine.edit(req)

            # Verify response
            assert response.ok is True
            assert response.model == Model.IMAGEN_3_CAPABILITY
            assert len(response.content) == 1

            # Verify SDK was called correctly
            mock_client.aio.models.edit_image.assert_called_once()

            # Verify the call arguments
            call_args = mock_client.aio.models.edit_image.call_args
            assert call_args[1]["model"] == str(Model.IMAGEN_3_CAPABILITY)
            assert call_args[1]["prompt"] == "Edit this image with mask"
            assert len(call_args[1]["reference_images"]) == 2  # base image + mask

        finally:
            os.remove(test_image)
            os.remove(test_mask)


async def test_edit_missing_image():
    """Test edit method error handling when no image is provided."""
    # Attempt to create edit request without images should raise ValidationError
    with pytest.raises(Exception) as exc_info:
        ImageEditRequest(
            prompt="Edit this image",
            images=[],  # Empty images list
            provider=Provider.VERTEX,
            model=Model.IMAGEN_3_CAPABILITY,
        )

    # Verify it's a validation error
    assert "images must contain at least one item" in str(exc_info.value)


async def test_edit_unsupported_model():
    """Test edit method properly rejects models that don't support editing."""
    engine = VertexImagen(provider=Provider.VERTEX)

    # Create test image
    test_image = _make_temp_png()
    try:
        # Create edit request with generation-only model
        req = ImageEditRequest(prompt="Edit this image", images=[test_image], provider=Provider.VERTEX, model=Model.IMAGEN_4_STANDARD, n=1)  # This model doesn't support editing

        # Call edit method
        response = await engine.edit(req)

        # Verify response is an error
        assert response.ok is False
        assert response.error is not None
        assert response.error.code == "unsupported_operation"
        assert "does not support image editing" in response.error.message
        assert "imagen-3.0-capability-001" in response.error.message

    finally:
        os.remove(test_image)
