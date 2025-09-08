from __future__ import annotations

import base64
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from image_gen_mcp.engines.diffusion.vertex_imagen import VertexImagen
from image_gen_mcp.schema import ImageEditRequest, ImageGenerateRequest, ImageResponse, ResourceContent, EmbeddedResource
from image_gen_mcp.shard.enums import Provider, Model


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
    assert len(caps.models) == 2
    
    # Check Imagen 4 capabilities
    imagen_4 = next(m for m in caps.models if m.model == Model.IMAGEN_4_STANDARD)
    assert imagen_4.supports_edit is True
    assert imagen_4.supports_mask is True
    assert imagen_4.supports_negative_prompt is True
    assert imagen_4.max_n == 4
    
    # Check Imagen 3 capabilities  
    imagen_3 = next(m for m in caps.models if m.model == Model.IMAGEN_3_GENERATE)
    assert imagen_3.supports_edit is True
    assert imagen_3.supports_mask is True
    assert imagen_3.supports_negative_prompt is True
    assert imagen_3.max_n == 4


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


@patch('image_gen_mcp.engines.diffusion.vertex_imagen.genai_sdk')
@patch('image_gen_mcp.engines.diffusion.vertex_imagen.genai_types')
async def test_edit_with_base_image(mock_genai_types, mock_genai_sdk):
    """Test edit method with base image (no mask)."""
    # Setup mocks
    mock_client = MagicMock()
    mock_genai_sdk.Client.return_value = mock_client
    
    mock_part = MagicMock()
    mock_genai_types.Part.from_bytes.return_value = mock_part
    
    # Mock response with image data
    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_content = MagicMock()
    mock_inline_part = MagicMock()
    mock_inline_data = MagicMock()
    
    mock_inline_data.data = b"fake_image_bytes"
    mock_inline_data.mime_type = "image/png"
    mock_inline_part.inline_data = mock_inline_data
    mock_content.parts = [mock_inline_part]
    mock_candidate.content = mock_content
    mock_response.candidates = [mock_candidate]
    
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    # Create engine and test data
    engine = VertexImagen(provider=Provider.VERTEX)
    
    # Mock settings for client creation
    with patch('image_gen_mcp.engines.diffusion.vertex_imagen.settings') as mock_settings:
        mock_settings.vertex_project = "test-project"
        mock_settings.vertex_location = "us-central1"
        
        # Create test image
        test_image = _make_temp_png()
        try:
            # Create edit request
            req = ImageEditRequest(
                prompt="Edit this image",
                images=[test_image],
                provider=Provider.VERTEX,
                model=Model.IMAGEN_4_STANDARD,
                n=1
            )
            
            # Call edit method
            response = await engine.edit(req)
            
            # Verify response
            assert response.ok is True
            assert response.model == Model.IMAGEN_4_STANDARD
            assert len(response.content) == 1
            assert response.content[0].type == "resource"
            assert response.content[0].resource.mimeType == "image/png"
            
            # Verify SDK was called correctly
            mock_genai_types.Part.from_bytes.assert_called_once()
            mock_client.aio.models.generate_content.assert_called_once()
            
            # Verify the call had the right structure (image part + prompt)
            call_args = mock_client.aio.models.generate_content.call_args
            assert call_args[1]['model'] == str(Model.IMAGEN_4_STANDARD)
            assert len(call_args[1]['contents']) == 2  # image part + prompt
            
        finally:
            os.remove(test_image)


@patch('image_gen_mcp.engines.diffusion.vertex_imagen.genai_sdk')
@patch('image_gen_mcp.engines.diffusion.vertex_imagen.genai_types')
async def test_edit_with_mask(mock_genai_types, mock_genai_sdk):
    """Test edit method with base image and mask."""
    # Setup mocks
    mock_client = MagicMock()
    mock_genai_sdk.Client.return_value = mock_client
    
    mock_part = MagicMock()
    mock_genai_types.Part.from_bytes.return_value = mock_part
    
    # Mock response with image data
    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_content = MagicMock()
    mock_inline_part = MagicMock()
    mock_inline_data = MagicMock()
    
    mock_inline_data.data = b"fake_image_bytes"
    mock_inline_data.mime_type = "image/png"
    mock_inline_part.inline_data = mock_inline_data
    mock_content.parts = [mock_inline_part]
    mock_candidate.content = mock_content
    mock_response.candidates = [mock_candidate]
    
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    # Create engine and test data
    engine = VertexImagen(provider=Provider.VERTEX)
    
    # Mock settings for client creation
    with patch('image_gen_mcp.engines.diffusion.vertex_imagen.settings') as mock_settings:
        mock_settings.vertex_project = "test-project"
        mock_settings.vertex_location = "us-central1"
        
        # Create test images
        test_image = _make_temp_png()
        test_mask = _make_temp_png()
        try:
            # Create edit request with mask
            req = ImageEditRequest(
                prompt="Edit this image with mask",
                images=[test_image],
                mask=test_mask,
                provider=Provider.VERTEX,
                model=Model.IMAGEN_4_STANDARD,
                n=1
            )
            
            # Call edit method
            response = await engine.edit(req)
            
            # Verify response
            assert response.ok is True
            assert response.model == Model.IMAGEN_4_STANDARD
            assert len(response.content) == 1
            
            # Verify SDK was called correctly
            assert mock_genai_types.Part.from_bytes.call_count == 2  # base image + mask
            mock_client.aio.models.generate_content.assert_called_once()
            
            # Verify the call had the right structure (image part + mask part + prompt)
            call_args = mock_client.aio.models.generate_content.call_args
            assert call_args[1]['model'] == str(Model.IMAGEN_4_STANDARD)
            assert len(call_args[1]['contents']) == 3  # image part + mask part + prompt
            
        finally:
            os.remove(test_image)
            os.remove(test_mask)


async def test_edit_missing_image():
    """Test edit method error handling when no image is provided."""
    engine = VertexImagen(provider=Provider.VERTEX)
    
    # Create edit request without images
    req = ImageEditRequest(
        prompt="Edit this image",
        images=[],  # Empty images list
        provider=Provider.VERTEX,
        model=Model.IMAGEN_4_STANDARD,
    )
    
    # Mock settings for client creation
    with patch('image_gen_mcp.engines.diffusion.vertex_imagen.settings') as mock_settings:
        mock_settings.vertex_project = "test-project"
        mock_settings.vertex_location = "us-central1"
        
        # Call edit method
        response = await engine.edit(req)
        
        # Verify error response
        assert response.ok is False
        assert response.error is not None
        assert "images[0] is required for edit" in response.error.message


@patch('image_gen_mcp.engines.diffusion.vertex_imagen.genai_sdk', None)
async def test_edit_sdk_missing():
    """Test edit method error handling when SDK is not installed."""
    engine = VertexImagen(provider=Provider.VERTEX)
    
    # Create test image
    test_image = _make_temp_png()
    try:
        # Create edit request
        req = ImageEditRequest(
            prompt="Edit this image",
            images=[test_image],
            provider=Provider.VERTEX,
            model=Model.IMAGEN_4_STANDARD,
        )
        
        # Mock settings for client creation
        with patch('image_gen_mcp.engines.diffusion.vertex_imagen.settings') as mock_settings:
            mock_settings.vertex_project = "test-project"
            mock_settings.vertex_location = "us-central1"
            
            # Call edit method
            response = await engine.edit(req)
            
            # Verify error response
            assert response.ok is False
            assert response.error is not None
            assert "google-genai SDK is required" in response.error.message
            
    finally:
        os.remove(test_image)