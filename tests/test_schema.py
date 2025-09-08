from __future__ import annotations

import pytest
from pydantic import ValidationError

from image_gen_mcp.schema import (
    CapabilitiesResponse,
    CapabilityReport,
    EmbeddedResource,
    Error,
    ImageDescriptor,
    ImageEditRequest,
    ImageGenerateRequest,
    ImageResponse,
    ImageToolStructured,
    ModelCapability,
    ResourceContent,
)
from image_gen_mcp.shard.enums import Family, Model, Orientation, Provider, Quality, SizeCode


def test_image_generate_request_valid():
    """Test that ImageGenerateRequest validates correctly with valid data."""
    req = ImageGenerateRequest(
        prompt="A beautiful sunset",
        provider=Provider.OPENAI,
        model=Model.GPT_IMAGE_1,
        n=2,
        size=SizeCode.M,
        orientation=Orientation.LANDSCAPE,
        quality=Quality.HIGH,
    )

    assert req.prompt == "A beautiful sunset"
    assert req.provider == Provider.OPENAI
    assert req.model == Model.GPT_IMAGE_1
    assert req.n == 2
    assert req.size == SizeCode.M
    assert req.orientation == Orientation.LANDSCAPE
    assert req.quality == Quality.HIGH


def test_image_generate_request_minimal():
    """Test ImageGenerateRequest with minimal required fields."""
    req = ImageGenerateRequest(
        prompt="Test prompt",
        provider=Provider.OPENAI,
        model=Model.GPT_IMAGE_1,
    )

    assert req.prompt == "Test prompt"
    assert req.provider == Provider.OPENAI
    assert req.model == Model.GPT_IMAGE_1
    # Defaults should be set
    assert req.n == 1


def test_image_generate_request_invalid_n():
    """Test that ImageGenerateRequest validates n parameter."""
    with pytest.raises(ValidationError):
        ImageGenerateRequest(prompt="Test", provider=Provider.OPENAI, model=Model.GPT_IMAGE_1, n=0)  # Invalid, must be >= 1


def test_image_edit_request_valid():
    """Test that ImageEditRequest validates correctly."""
    req = ImageEditRequest(
        prompt="Edit this image",
        images=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB/9k3WQAAAABJRU5ErkJggg=="],
        provider=Provider.OPENAI,
        model=Model.GPT_IMAGE_1,
    )

    assert req.prompt == "Edit this image"
    assert len(req.images) == 1
    assert req.provider == Provider.OPENAI
    assert req.model == Model.GPT_IMAGE_1


def test_image_edit_request_with_mask():
    """Test ImageEditRequest with mask."""
    mask = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB/9k3WQAAAABJRU5ErkJggg=="
    req = ImageEditRequest(
        prompt="Edit with mask",
        images=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB/9k3WQAAAABJRU5ErkJggg=="],
        mask=mask,
        provider=Provider.OPENAI,
        model=Model.GPT_IMAGE_1,
    )

    assert req.mask == mask


def test_image_response_success():
    """Test ImageResponse for successful response."""
    resp = ImageResponse(ok=True, content=[], model=Model.GPT_IMAGE_1)

    assert resp.ok is True
    assert resp.content == []
    assert resp.model == Model.GPT_IMAGE_1
    assert resp.error is None


def test_image_response_error():
    """Test ImageResponse for error response."""
    error = Error(code="provider_unavailable", message="OpenAI provider not configured")
    resp = ImageResponse(ok=False, content=[], model=Model.GPT_IMAGE_1, error=error)

    assert resp.ok is False
    assert resp.content == []
    assert resp.error is not None
    assert resp.error.code == "provider_unavailable"
    assert resp.error.message == "OpenAI provider not configured"


def test_model_capability():
    """Test ModelCapability structure."""
    model_cap = ModelCapability(model=Model.GPT_IMAGE_1, supports_negative_prompt=False, supports_background=True, max_n=1, supports_edit=True, supports_mask=True)

    assert model_cap.model == Model.GPT_IMAGE_1
    assert model_cap.supports_negative_prompt is False
    assert model_cap.supports_background is True
    assert model_cap.max_n == 1
    assert model_cap.supports_edit is True
    assert model_cap.supports_mask is True


def test_capability_report():
    """Test CapabilityReport structure."""
    model_cap = ModelCapability(model=Model.GPT_IMAGE_1, supports_negative_prompt=False, max_n=1)

    report = CapabilityReport(provider=Provider.OPENAI, family=Family.AR, models=[model_cap])

    assert report.provider == Provider.OPENAI
    assert report.family == Family.AR
    assert len(report.models) == 1
    assert report.models[0].model == Model.GPT_IMAGE_1


def test_capabilities_response():
    """Test CapabilitiesResponse structure."""
    model_cap = ModelCapability(model=Model.GPT_IMAGE_1)
    report = CapabilityReport(provider=Provider.OPENAI, family=Family.AR, models=[model_cap])

    resp = CapabilitiesResponse(ok=True, capabilities=[report])

    assert resp.ok is True
    assert len(resp.capabilities) == 1
    assert resp.capabilities[0].provider == Provider.OPENAI


def test_error_model():
    """Test Error model validation."""
    error = Error(code="test_error", message="Test error message")

    assert error.code == "test_error"
    assert error.message == "Test error message"
    assert error.details is None


def test_error_with_details():
    """Test Error model with details."""
    details = {"additional_info": "Some extra context"}
    error = Error(code="detailed_error", message="Error with details", details=details)

    assert error.code == "detailed_error"
    assert error.message == "Error with details"
    assert error.details == details


def test_image_response_structured_content():
    """Test ImageResponse structured content generation."""
    resp = ImageResponse(ok=True, content=[], model=Model.GPT_IMAGE_1)

    structured = resp.build_structured_from_response()

    assert structured.ok is True
    assert structured.model == Model.GPT_IMAGE_1
    assert structured.error is None


def test_image_response_structured_content_with_error():
    """Test ImageResponse structured content with error."""
    error = Error(code="test", message="Test error")
    resp = ImageResponse(ok=False, content=[], model=Model.GPT_IMAGE_1, error=error)

    structured = resp.build_structured_from_response()

    assert structured.ok is False
    assert structured.error is not None
    assert structured.error.code == "test"
    assert structured.error.message == "Test error"


def test_embedded_resource():
    """Test EmbeddedResource structure."""
    resource = EmbeddedResource(uri="image://test123", name="test-image.png", mimeType="image/png", blob="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB/9k3WQAAAABJRU5ErkJggg==", description="Test image")

    assert resource.uri == "image://test123"
    assert resource.name == "test-image.png"
    assert resource.mimeType == "image/png"
    assert resource.description == "Test image"


def test_resource_content():
    """Test ResourceContent structure."""
    resource = EmbeddedResource(uri="image://test123", name="test.png", mimeType="image/png", blob="test-blob")

    content = ResourceContent(resource=resource)

    assert content.type == "resource"
    assert content.resource.uri == "image://test123"


def test_image_descriptor():
    """Test ImageDescriptor structure."""
    descriptor = ImageDescriptor(uri="image://test123", name="test.png", mimeType="image/png")

    assert descriptor.uri == "image://test123"
    assert descriptor.name == "test.png"
    assert descriptor.mimeType == "image/png"


def test_image_tool_structured():
    """Test ImageToolStructured structure."""
    descriptor = ImageDescriptor(uri="image://test", name="test.png")
    structured = ImageToolStructured(ok=True, model=Model.GPT_IMAGE_1, image_count=1, images=[descriptor])

    assert structured.ok is True
    assert structured.model == Model.GPT_IMAGE_1
    assert structured.image_count == 1
    assert len(structured.images) == 1
