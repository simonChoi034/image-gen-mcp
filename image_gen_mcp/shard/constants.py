"""Project constants for image generation adapters and routing.

This module centralizes normalized parameter names and defaults shared across
both diffusion-style and autoregressive (AR) image models. Keep these values
provider-agnostic and small in scope; provider-specific translations belong in
individual engine adapters.
"""

from __future__ import annotations

from typing import Final

# Normalized enums live in ``src/shard/enums.py``. Import is safe here to expose
# sensible defaults for unified controls without creating circular imports.
from .enums import Orientation, Quality, SizeCode

# ----------------------------- General defaults ----------------------------- #

# Default requested output size for providers that accept ``WIDTHxHEIGHT``
# descriptors (OpenAI, Azure) or that infer an aspect from size (some adapters
# may translate this to aspect ratios when required).
DEFAULT_SIZE: Final[str] = "1024x1024"

# Unified control defaults (model-agnostic)
DEFAULT_SIZE_CODE: Final[SizeCode] = SizeCode.M
DEFAULT_ORIENTATION: Final[Orientation] = Orientation.SQUARE
DEFAULT_QUALITY: Final[Quality] = Quality.STANDARD

# Default number of images to request when ``n`` is omitted.
DEFAULT_N: Final[int] = 1

# Recommended upper-bound for ``n`` per unified schema (adapters will clamp
# further to provider-specific limits when needed).
MAX_N: Final[int] = 4

# Normalized default mime type for returned images when not explicitly set by
# a provider or when the provider returns only bytes/base64.
DEFAULT_MIME: Final[str] = "image/png"


# General error codes used across engines
ERROR_CODE_UNSUPPORTED_FIELDS: Final[str] = "UNSUPPORTED_FIELDS"
ERROR_CODE_PROVIDER_ERROR: Final[str] = "provider_error"
ERROR_CODE_PROVIDER_UNAVAILABLE: Final[str] = "provider_unavailable"
