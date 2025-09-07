from __future__ import annotations

import base64
import mimetypes
import os
import urllib.request
from urllib.parse import unquote, urlparse

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


__all__ = [
    "is_url",
    "is_data_url",
    "is_file_url",
    "file_url_to_path",
    "guess_mime_from_path",
    "validate_image_bytes",
    "read_image_bytes_and_mime",
    "to_image_data_url",
]
