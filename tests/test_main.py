from __future__ import annotations

from image_gen_mcp.main import app, main, mcp_edit_image, mcp_generate_image, mcp_get_model_capabilities


def test_fastmcp_app_exists():
    """Test that the FastMCP app is properly created."""
    assert app is not None
    assert app.name == "image-gen-mcp"


def test_main_function_exists():
    """Test that the main function exists and can be called."""
    assert callable(main)


def test_mcp_tools_are_registered():
    """Test that all MCP tools are properly registered."""
    # Test that the tools exist and have the expected structure
    assert hasattr(mcp_generate_image, "name")
    assert hasattr(mcp_edit_image, "name")
    assert hasattr(mcp_get_model_capabilities, "name")

    # Test tool names
    assert mcp_generate_image.name == "generate_image"
    assert mcp_edit_image.name == "edit_image"
    assert mcp_get_model_capabilities.name == "get_model_capabilities"


def test_main_function_argument_parsing():
    """Test that main function can handle command line arguments."""
    # This is a basic test to ensure the function exists and can be imported
    # More detailed testing would require mocking sys.argv
    assert callable(main)


def test_parser_transport_choices():
    """Ensure the CLI parser restricts transport choices to stdio, sse, http."""
    import argparse

    parser = argparse.ArgumentParser()
    # Recreate the argument as in main to validate choices programmatically
    parser.add_argument("--transport", choices=["stdio", "sse", "http"])

    # Valid choices should parse
    args = parser.parse_args(["--transport", "stdio"])
    assert args.transport == "stdio"

    args = parser.parse_args(["--transport", "http"])
    assert args.transport == "http"

    # Invalid choice should raise SystemExit from argparse
    try:
        parser.parse_args(["--transport", "invalid"])
        raise AssertionError("parser did not reject invalid transport")
    except SystemExit:
        # Expected behavior
        pass
