from __future__ import annotations

import base64
import mimetypes
import os
import tempfile
import urllib.request
from datetime import datetime
from urllib.parse import unquote, urlparse
from uuid import uuid4

from loguru import logger

from ..shard import constants as C


# --------------------------- source classifiers --------------------------- #
def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def is_data_url(value: str) -> bool:
    return value.startswith("data:")


def is_file_url(value: str) -> bool:
    return value.startswith("file://")


def file_url_to_path(url: str) -> str:
    parsed = urlparse(url)
    return unquote(parsed.path)


def guess_mime_from_path(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or C.DEFAULT_MIME


# --------------------------- validation ---------------------------------- #
def validate_image_bytes(data: bytes, mime: str | None = None) -> None:
    """Basic validation to ensure the bytes look like an image.

    Checks common magic numbers for PNG/JPEG/WEBP/GIF and a sane minimum length.
    Raises ValueError if validation fails.
    """
    if not data or len(data) < 16:
        raise ValueError("Image data is empty or too small")

    # Magic headers
    png_sig = b"\x89PNG\r\n\x1a\x0a"
    jpeg_sig = b"\xff\xd8\xff"
    riff = b"RIFF"
    webp = b"WEBP"
    gif87a = b"GIF87a"
    gif89a = b"GIF89a"

    valid = False
    if data.startswith(png_sig):
        valid = True
    elif data.startswith(jpeg_sig):
        valid = True
    elif data.startswith(gif87a) or data.startswith(gif89a):
        valid = True
    elif data.startswith(riff) and webp in data[:32]:
        valid = True

    if not valid:
        raise ValueError("Unsupported or corrupt image data; expected PNG/JPEG/GIF/WEBP")


# --------------------------- IO + conversion ------------------------------ #
def read_image_bytes_and_mime(source: str, *, validate: bool = True) -> tuple[bytes, str]:
    """Read image content from various source forms.

    Accepts http(s) URLs, data URLs, local file paths, file:// URLs, or bare base64.
    Returns (bytes, mime_type).
    """
    if is_url(source):
        with urllib.request.urlopen(source) as resp:  # nosec - controlled by caller
            mime = resp.headers.get_content_type() or C.DEFAULT_MIME
            data = resp.read()
        if validate:
            validate_image_bytes(data, mime)
        return data, mime

    if is_data_url(source):
        try:
            header, payload = source.split(",", 1)
        except ValueError:
            raise ValueError("Invalid data URL for image")
        mime = C.DEFAULT_MIME
        try:
            mime = header.split(";")[0].split(":", 1)[1] or C.DEFAULT_MIME
        except Exception:
            pass
        data = base64.b64decode(payload)
        if validate:
            validate_image_bytes(data, mime)
        return data, mime

    # file:// URL
    if is_file_url(source):
        path = file_url_to_path(source)
        if not os.path.exists(path):
            raise ValueError(f"File not found: {path}")
        with open(path, "rb") as f:
            data = f.read()
        mime = guess_mime_from_path(path)
        if validate:
            validate_image_bytes(data, mime)
        return data, mime

    # Local path
    if os.path.exists(source):
        with open(source, "rb") as f:
            data = f.read()
        mime = guess_mime_from_path(source)
        if validate:
            validate_image_bytes(data, mime)
        return data, mime

    # Bare base64
    try:
        data = base64.b64decode(source, validate=True)
        if validate:
            validate_image_bytes(data, C.DEFAULT_MIME)
        return data, C.DEFAULT_MIME
    except Exception:
        raise ValueError("Unsupported image source: must be URL, data URL, local file path, or base64 string")


def to_image_data_url(source: str, *, validate_remote: bool = False) -> str:
    """Return a data URL for the image; passthrough if already URL/data URL.

    - http(s) and data: URLs are returned unchanged unless validate_remote=True
    - file paths, file:// URLs, and bare base64 are converted to data URLs
    """
    if is_url(source):
        if not validate_remote:
            return source
        data, mime = read_image_bytes_and_mime(source, validate=True)
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    if is_data_url(source):
        data, mime = read_image_bytes_and_mime(source, validate=True)
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    data, mime = read_image_bytes_and_mime(source, validate=True)
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def guess_extension_from_mime(mime: str) -> str:
    """Guess file extension from MIME type."""
    if mime.endswith("/png"):
        return ".png"
    elif mime.endswith("/jpeg") or mime.endswith("/jpg"):
        return ".jpg"
    elif mime.endswith("/webp"):
        return ".webp"
    elif mime.endswith("/gif"):
        return ".gif"
    else:
        return ".png"  # Default fallback


def ensure_directory(directory: str | None) -> str:
    """Ensure directory exists, creating if necessary. Returns absolute path.

    If directory is None, creates a temporary directory.
    """
    if directory is None:
        # Create a temporary directory
        return tempfile.mkdtemp(prefix="image_gen_mcp_", dir=tempfile.gettempdir())

    # Convert to absolute path
    abs_directory = os.path.abspath(directory)

    # Create directory if it doesn't exist
    try:
        os.makedirs(abs_directory, exist_ok=True)
    except (OSError, PermissionError) as e:
        # Fallback to temp directory on error
        temp_dir = tempfile.mkdtemp(prefix="image_gen_mcp_fallback_", dir=tempfile.gettempdir())
        logger.warning(f"Cannot create directory {abs_directory}: {e}. Using temp directory: {temp_dir}")
        return temp_dir

    return abs_directory


def save_image_bytes(image_bytes: bytes, directory: str | None, mime_type: str = C.DEFAULT_MIME) -> str:
    """Save image bytes to disk and return the absolute path.

    Args:
        image_bytes: Raw image data
        directory: Target directory (None for temp directory)
        mime_type: MIME type for determining file extension

    Returns:
        Absolute path to the saved file

    Raises:
        ValueError: If directory creation fails or image saving fails
    """
    # Ensure target directory exists
    target_dir = ensure_directory(directory)

    # Generate unique filename with timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid4())[:8]  # Short UUID for readability
    extension = guess_extension_from_mime(mime_type)
    filename = f"{timestamp}_{unique_id}{extension}"

    # Full path
    file_path = os.path.join(target_dir, filename)

    # Write image bytes to file
    try:
        with open(file_path, "wb") as f:
            f.write(image_bytes)
    except (OSError, PermissionError) as e:
        raise ValueError(f"Cannot write image to {file_path}: {e}") from e

    return os.path.abspath(file_path)


def save_image_from_data_url(data_url: str, directory: str | None) -> str:
    """Save image from data URL to disk and return the absolute path.

    Args:
        data_url: Data URL containing image data
        directory: Target directory (None for temp directory)

    Returns:
        Absolute path to the saved file

    Raises:
        ValueError: If data URL is invalid or saving fails
    """
    try:
        # Parse data URL
        header, payload = data_url.split(",", 1)
    except ValueError:
        raise ValueError("Invalid data URL format")

    # Extract MIME type
    mime_type = C.DEFAULT_MIME
    try:
        mime_type = header.split(";")[0].split(":", 1)[1] or C.DEFAULT_MIME
    except Exception:
        pass

    # Decode base64 content
    try:
        image_bytes = base64.b64decode(payload)
    except Exception as e:
        raise ValueError(f"Cannot decode base64 data: {e}") from e

    return save_image_bytes(image_bytes, directory, mime_type)


def save_images_from_response(resp: object, directory: str | None) -> None:
    """Save images embedded in an ImageResponse to disk and update their file_path.

    This inspects `resp.content` for ResourceContent entries with `resource.blob` set
    (base64-encoded), decodes and writes them to `directory` (or a tempdir when None),
    and sets `resource.file_path` to the absolute path on success. Errors for individual
    images are logged and do not stop processing of other images.
    """

    # logger is module-level; local reference kept for tests
    # (we avoid importing ImageResponse here to prevent circular imports)

    for part in getattr(resp, "content", []):
        if getattr(part, "type", None) != "resource":
            continue

        resource = getattr(part, "resource", None)
        if not resource or not getattr(resource, "blob", None):
            continue

        try:
            # Decode base64 image data
            image_bytes = base64.b64decode(resource.blob)
            mime_type = getattr(resource, "mimeType", None) or C.DEFAULT_MIME

            # Save to disk using existing helper
            file_path = save_image_bytes(image_bytes, directory, mime_type)

            # Update the resource with the file path
            setattr(resource, "file_path", file_path)

        except Exception as e:
            logger.error(f"Failed to save image {getattr(resource, 'name', 'unknown')}: {e}")
            # Continue processing other images even if one fails
            continue


__all__ = [
    "is_url",
    "is_data_url",
    "is_file_url",
    "file_url_to_path",
    "guess_mime_from_path",
    "validate_image_bytes",
    "read_image_bytes_and_mime",
    "to_image_data_url",
    "guess_extension_from_mime",
    "ensure_directory",
    "save_image_bytes",
    "save_image_from_data_url",
    "save_images_from_response",
]
