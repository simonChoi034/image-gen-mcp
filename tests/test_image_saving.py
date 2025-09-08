from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from unittest.mock import patch

import pytest

from image_gen_mcp.schema import EmbeddedResource, ImageResponse, ResourceContent
from image_gen_mcp.shard.enums import Model
from image_gen_mcp.utils.image_utils import (
    ensure_directory,
    guess_extension_from_mime,
    save_image_bytes,
    save_image_from_data_url,
    save_images_from_response,
)

# Sample 1x1 PNG image as base64
SAMPLE_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI" "9Ecf1UQAAAABJRU5ErkJggg=="

SAMPLE_PNG_BYTES = base64.b64decode(SAMPLE_PNG_B64)


class TestImageSavingUtils:
    """Test the image saving utility functions."""

    def test_guess_extension_from_mime(self):
        """Test MIME type to extension mapping."""
        assert guess_extension_from_mime("image/png") == ".png"
        assert guess_extension_from_mime("image/jpeg") == ".jpg"
        assert guess_extension_from_mime("image/jpg") == ".jpg"
        assert guess_extension_from_mime("image/webp") == ".webp"
        assert guess_extension_from_mime("image/gif") == ".gif"
        assert guess_extension_from_mime("image/unknown") == ".png"  # fallback

    def test_ensure_directory_creates_temp_when_none(self):
        """Test that ensure_directory creates temp dir when None is passed."""
        temp_dir = ensure_directory(None)
        assert os.path.exists(temp_dir)
        assert os.path.isdir(temp_dir)
        assert "image_gen_mcp_" in os.path.basename(temp_dir)
        # Cleanup
        os.rmdir(temp_dir)

    def test_ensure_directory_creates_specified_directory(self):
        """Test that ensure_directory creates the specified directory."""
        with tempfile.TemporaryDirectory() as temp_base:
            test_dir = os.path.join(temp_base, "test_images")
            result_dir = ensure_directory(test_dir)
            assert os.path.exists(result_dir)
            assert os.path.isdir(result_dir)
            assert os.path.abspath(test_dir) == result_dir

    def test_ensure_directory_handles_existing_directory(self):
        """Test that ensure_directory works with existing directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = ensure_directory(temp_dir)
            assert os.path.abspath(temp_dir) == result_dir

    def test_save_image_bytes(self):
        """Test saving image bytes to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = save_image_bytes(SAMPLE_PNG_BYTES, temp_dir, "image/png")

            # Check file was created
            assert os.path.exists(file_path)
            assert file_path.endswith(".png")
            assert temp_dir in file_path

            # Check file content
            with open(file_path, "rb") as f:
                saved_bytes = f.read()
            assert saved_bytes == SAMPLE_PNG_BYTES

    def test_save_image_bytes_with_temp_directory(self):
        """Test saving image bytes with no specified directory."""
        file_path = save_image_bytes(SAMPLE_PNG_BYTES, None, "image/png")

        # Check file was created
        assert os.path.exists(file_path)
        assert file_path.endswith(".png")

        # Check file content
        with open(file_path, "rb") as f:
            saved_bytes = f.read()
        assert saved_bytes == SAMPLE_PNG_BYTES

        # Cleanup
        os.unlink(file_path)
        # Also cleanup the temp directory (get parent dir)
        temp_dir = os.path.dirname(file_path)
        if "image_gen_mcp_" in os.path.basename(temp_dir):
            os.rmdir(temp_dir)

    def test_save_image_from_data_url(self):
        """Test saving image from data URL."""
        data_url = f"data:image/png;base64,{SAMPLE_PNG_B64}"

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = save_image_from_data_url(data_url, temp_dir)

            # Check file was created
            assert os.path.exists(file_path)
            assert file_path.endswith(".png")

            # Check file content
            with open(file_path, "rb") as f:
                saved_bytes = f.read()
            assert saved_bytes == SAMPLE_PNG_BYTES

    def test_save_image_from_data_url_invalid(self):
        """Test saving from invalid data URL raises error."""
        with pytest.raises(ValueError, match="Invalid data URL format"):
            save_image_from_data_url("not-a-data-url", None)


class TestSaveImagesToDisk:
    """Test the _save_images_to_disk function."""

    def create_sample_response(self, num_images: int = 1) -> ImageResponse:
        """Create a sample ImageResponse with mock images."""
        content = []
        for i in range(num_images):
            resource = EmbeddedResource(
                uri=f"image://test-{i}",
                name=f"test-image-{i}.png",
                mimeType="image/png",
                blob=SAMPLE_PNG_B64,
                description=f"Test image {i}",
            )
            content.append(ResourceContent(type="resource", resource=resource))

        return ImageResponse(
            ok=True,
            content=content,
            model=Model.DALL_E_3,
        )

    @pytest.mark.asyncio
    async def test_save_images_to_disk_with_directory(self):
        """Test saving images with specified directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            response = self.create_sample_response(2)

            await asyncio.to_thread(save_images_from_response, response, temp_dir)

            # Check that file_path was set for each resource
            for part in response.content:
                resource = part.resource
                assert hasattr(resource, "file_path")
                assert resource.file_path is not None
                assert os.path.exists(resource.file_path)
                assert temp_dir in resource.file_path
                assert resource.file_path.endswith(".png")

    @pytest.mark.asyncio
    async def test_save_images_to_disk_with_temp_directory(self):
        """Test saving images with temp directory (directory=None)."""
        response = self.create_sample_response(1)

        await asyncio.to_thread(save_images_from_response, response, None)

        # Check that file_path was set
        resource = response.content[0].resource
        assert hasattr(resource, "file_path")
        assert resource.file_path is not None
        assert os.path.exists(resource.file_path)
        assert resource.file_path.endswith(".png")

        # Cleanup
        os.unlink(resource.file_path)
        # Also cleanup the temp directory
        temp_dir = os.path.dirname(resource.file_path)
        if "image_gen_mcp_" in os.path.basename(temp_dir):
            os.rmdir(temp_dir)

    @pytest.mark.asyncio
    async def test_save_images_to_disk_with_invalid_base64(self):
        """Test handling of invalid base64 data."""
        # Create response with invalid base64
        resource = EmbeddedResource(
            uri="image://test",
            name="test-image.png",
            mimeType="image/png",
            blob="invalid-base64!!!",
            description="Test image with invalid data",
        )
        response = ImageResponse(
            ok=True,
            content=[ResourceContent(type="resource", resource=resource)],
            model=Model.DALL_E_3,
        )

        # Should not raise error, but should log error and continue
        with patch("image_gen_mcp.utils.image_utils.logger") as mock_logger:
            await asyncio.to_thread(save_images_from_response, response, None)
            mock_logger.error.assert_called_once()

        # file_path should not be set for invalid data
        assert not hasattr(resource, "file_path") or resource.file_path is None

    @pytest.mark.asyncio
    async def test_save_images_to_disk_ignores_non_resource_content(self):
        """Test that non-resource content is ignored."""
        # Create dummy resource for non-resource content
        dummy_resource = EmbeddedResource(
            uri="dummy://test",
            name="dummy.txt",
            mimeType="text/plain",
            blob="dummy",
            description="Dummy resource",
        )

        # Create response with mixed content types
        response = ImageResponse(
            ok=True,
            content=[
                ResourceContent(type="text", resource=dummy_resource),  # Will be ignored
                ResourceContent(
                    type="resource",
                    resource=EmbeddedResource(
                        uri="image://test",
                        name="test-image.png",
                        mimeType="image/png",
                        blob=SAMPLE_PNG_B64,
                        description="Test image",
                    ),
                ),
            ],
            model=Model.DALL_E_3,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            await asyncio.to_thread(save_images_from_response, response, temp_dir)

            # Only the resource content should have file_path set
            assert not hasattr(response.content[0], "file_path")
            resource = response.content[1].resource
            assert hasattr(resource, "file_path")
            assert resource.file_path is not None
            assert os.path.exists(resource.file_path)


class TestImageDescriptorWithFilePath:
    """Test that ImageDescriptor includes file_path in structured responses."""

    def test_structured_response_includes_file_path(self):
        """Test that build_structured_from_response includes file_path."""
        # Create a response with file_path set
        resource = EmbeddedResource(
            uri="image://test",
            name="test-image.png",
            mimeType="image/png",
            blob=SAMPLE_PNG_B64,
            description="Test image",
            file_path="/tmp/test/image.png",
        )
        response = ImageResponse(
            ok=True,
            content=[ResourceContent(type="resource", resource=resource)],
            model=Model.DALL_E_3,
        )

        structured = response.build_structured_from_response()

        assert len(structured.images) == 1
        image_desc = structured.images[0]
        assert image_desc.file_path == "/tmp/test/image.png"
        assert image_desc.uri == "image://test"
        assert image_desc.name == "test-image.png"
        assert image_desc.mimeType == "image/png"

    def test_structured_response_handles_none_file_path(self):
        """Test that build_structured_from_response handles None file_path."""
        # Create a response without file_path set
        resource = EmbeddedResource(
            uri="image://test",
            name="test-image.png",
            mimeType="image/png",
            blob=SAMPLE_PNG_B64,
            description="Test image",
            # file_path=None (default)
        )
        response = ImageResponse(
            ok=True,
            content=[ResourceContent(type="resource", resource=resource)],
            model=Model.DALL_E_3,
        )

        structured = response.build_structured_from_response()

        assert len(structured.images) == 1
        image_desc = structured.images[0]
        assert image_desc.file_path is None
        assert image_desc.uri == "image://test"
