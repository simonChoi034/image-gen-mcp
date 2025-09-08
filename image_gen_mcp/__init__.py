"""
Image Gen MCP Server

Provider-agnostic MCP server for image generation and editing,
built on FastMCP with clean engine routing and a stable, type-safe schema.
"""

__version__ = "0.1.0"

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("image-gen-mcp")
except (ImportError, Exception):
    # Fallback for development mode or when package is not installed
    pass

__all__ = ["__version__"]
