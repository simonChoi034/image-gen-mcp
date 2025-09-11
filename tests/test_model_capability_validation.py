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


def test_generation_capability_validation():
    """Test that ModelFactory correctly validates generation capability."""
    # Models that support generation
    generation_models = [
        Model.GPT_IMAGE_1,
        Model.DALL_E_3,
        Model.GEMINI_IMAGE_PREVIEW,
        Model.OPENROUTER_GOOGLE_GEMINI_IMAGE,
        Model.IMAGEN_4_STANDARD,
        Model.IMAGEN_4_FAST,
        Model.IMAGEN_4_ULTRA,
        Model.IMAGEN_3_GENERATE,
    ]

    for model in generation_models:
        assert ModelFactory.model_supports_generation(model), f"{model.value} should support generation"
        # These models should pass validation
        error = ModelFactory.validate_generation_request(model)
        assert error is None, f"{model.value} should pass generation validation"

    # Models that don't support generation (edit-only)
    edit_only_models = [
        Model.IMAGEN_3_CAPABILITY,
    ]

    for model in edit_only_models:
        assert not ModelFactory.model_supports_generation(model), f"{model.value} should not support generation"
        # These models should fail validation
        error = ModelFactory.validate_generation_request(model)
        assert error is not None, f"{model.value} should fail generation validation"
        assert error.code == "unsupported_operation"
        assert "does not support standalone image generation" in error.message


def test_dynamic_capability_discovery():
    """Test that dynamic capability discovery works correctly."""
    # Test specific model capabilities by querying the factory

    # IMAGEN_3_CAPABILITY should be edit-only (no generation)
    assert not ModelFactory.model_supports_generation(Model.IMAGEN_3_CAPABILITY), "IMAGEN_3_CAPABILITY should not support generation"
    assert ModelFactory.model_supports_editing(Model.IMAGEN_3_CAPABILITY), "IMAGEN_3_CAPABILITY should support editing"
    assert ModelFactory.model_supports_masking(Model.IMAGEN_3_CAPABILITY), "IMAGEN_3_CAPABILITY should support masking"

    # IMAGEN_4_STANDARD should be generation-only (no editing)
    assert ModelFactory.model_supports_generation(Model.IMAGEN_4_STANDARD), "IMAGEN_4_STANDARD should support generation"
    assert not ModelFactory.model_supports_editing(Model.IMAGEN_4_STANDARD), "IMAGEN_4_STANDARD should not support editing"
    assert not ModelFactory.model_supports_masking(Model.IMAGEN_4_STANDARD), "IMAGEN_4_STANDARD should not support masking"

    # GPT_IMAGE_1 should support both generation and editing
    assert ModelFactory.model_supports_generation(Model.GPT_IMAGE_1), "GPT_IMAGE_1 should support generation"
    assert ModelFactory.model_supports_editing(Model.GPT_IMAGE_1), "GPT_IMAGE_1 should support editing"
    assert ModelFactory.model_supports_masking(Model.GPT_IMAGE_1), "GPT_IMAGE_1 should support masking"

    # DALL_E_3 should be generation-only
    assert ModelFactory.model_supports_generation(Model.DALL_E_3), "DALL_E_3 should support generation"
    assert not ModelFactory.model_supports_editing(Model.DALL_E_3), "DALL_E_3 should not support editing"
    assert not ModelFactory.model_supports_masking(Model.DALL_E_3), "DALL_E_3 should not support masking"
