"""
Image Gen MCP Server

Provider-agnostic MCP server for image generation and editing,
built on FastMCP with clean engine routing and a stable, type-safe schema.
"""

__version__ = "0.1.0"

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("image-gen-mcp")
except (importlib.metadata.PackageNotFoundError, ImportError):
    # Fallback for development mode
    pass

__all__ = ["__version__"]
