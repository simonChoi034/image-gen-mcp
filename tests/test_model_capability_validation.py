"""Test model capability validation functionality."""

from image_gen_mcp.engines.factory import ModelFactory
from image_gen_mcp.shard.enums import Model


def test_editing_capability_validation():
    """Test that ModelFactory correctly validates editing capability."""
    # Models that support editing
    editing_models = [
        Model.GPT_IMAGE_1,
        Model.GEMINI_IMAGE_PREVIEW,
        Model.OPENROUTER_GOOGLE_GEMINI_IMAGE,
        Model.IMAGEN_3_CAPABILITY,
    ]

    for model in editing_models:
        assert ModelFactory.model_supports_editing(model), f"{model.value} should support editing"
        # These models should pass validation
        error = ModelFactory.validate_edit_request(model, has_mask=False)
        assert error is None, f"{model.value} should pass edit validation"

    # Models that don't support editing
    generation_only_models = [
        Model.DALL_E_3,
        Model.IMAGEN_4_STANDARD,
        Model.IMAGEN_4_FAST,
        Model.IMAGEN_4_ULTRA,
        Model.IMAGEN_3_GENERATE,
    ]

    for model in generation_only_models:
        assert not ModelFactory.model_supports_editing(model), f"{model.value} should not support editing"
        # These models should fail validation
        error = ModelFactory.validate_edit_request(model, has_mask=False)
        assert error is not None, f"{model.value} should fail edit validation"
        assert error.code == "unsupported_operation"
        assert "does not support image editing" in error.message


def test_masking_capability_validation():
    """Test that ModelFactory correctly validates masking capability."""
    # Models that support masking
    masking_models = [
        Model.GPT_IMAGE_1,
        Model.IMAGEN_3_CAPABILITY,
    ]

    for model in masking_models:
        assert ModelFactory.model_supports_masking(model), f"{model.value} should support masking"
        # These models should pass mask validation
        error = ModelFactory.validate_edit_request(model, has_mask=True)
        assert error is None, f"{model.value} should pass mask validation"

    # Models that support editing but not masking
    maskless_editing_models = [
        Model.GEMINI_IMAGE_PREVIEW,
        Model.OPENROUTER_GOOGLE_GEMINI_IMAGE,
    ]

    for model in maskless_editing_models:
        assert not ModelFactory.model_supports_masking(model), f"{model.value} should not support masking"
        # These models should pass edit validation without mask
        error = ModelFactory.validate_edit_request(model, has_mask=False)
        assert error is None, f"{model.value} should pass edit validation without mask"
        # But fail with mask
        error = ModelFactory.validate_edit_request(model, has_mask=True)
        assert error is not None, f"{model.value} should fail mask validation"
        assert error.code == "unsupported_operation"
        assert "does not support masking" in error.message


def test_validate_edit_request_comprehensive():
    """Test comprehensive edit request validation scenarios."""

    # Valid editing model without mask
    error = ModelFactory.validate_edit_request(Model.IMAGEN_3_CAPABILITY, has_mask=False)
    assert error is None

    # Valid editing model with mask
    error = ModelFactory.validate_edit_request(Model.IMAGEN_3_CAPABILITY, has_mask=True)
    assert error is None

    # Invalid model for editing (generation-only)
    error = ModelFactory.validate_edit_request(Model.IMAGEN_4_STANDARD, has_mask=False)
    assert error is not None
    assert error.code == "unsupported_operation"
    assert "imagen-4.0-generate-001 does not support image editing" in error.message

    # Editing model that doesn't support masking, with mask
    error = ModelFactory.validate_edit_request(Model.GEMINI_IMAGE_PREVIEW, has_mask=True)
    assert error is not None
    assert error.code == "unsupported_operation"
    assert "gemini-2.5-flash-image-preview does not support masking" in error.message


def test_model_capability_constants():
    """Test that capability constants are correctly defined."""
    from image_gen_mcp.engines.factory import (
        EDIT_CAPABLE_MODELS,
        GENERATION_ONLY_MODELS,
        MASK_CAPABLE_MODELS,
    )

    # Verify no overlap between editing and generation-only models
    assert len(EDIT_CAPABLE_MODELS & GENERATION_ONLY_MODELS) == 0, "Models can't be both edit-capable and generation-only"

    # Verify mask-capable models are subset of edit-capable models
    assert MASK_CAPABLE_MODELS.issubset(EDIT_CAPABLE_MODELS), "All mask-capable models must be edit-capable"

    # Verify expected models are in the right sets
    assert Model.IMAGEN_3_CAPABILITY in EDIT_CAPABLE_MODELS
    assert Model.IMAGEN_3_CAPABILITY in MASK_CAPABLE_MODELS
    assert Model.IMAGEN_4_STANDARD in GENERATION_ONLY_MODELS
    assert Model.IMAGEN_4_STANDARD not in EDIT_CAPABLE_MODELS
    assert Model.GPT_IMAGE_1 in EDIT_CAPABLE_MODELS
    assert Model.GPT_IMAGE_1 in MASK_CAPABLE_MODELS
