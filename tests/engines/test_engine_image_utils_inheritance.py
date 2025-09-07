from __future__ import annotations

import base64
import os
import tempfile

from src.engines.diffusion.dalle_diffusion import DalleDiffusion
from src.engines.diffusion.vertex_imagen import VertexImagen
from src.shard.enums import Provider


def _temp_png_path() -> str:
    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB/9k3WQAAAABJRU5ErkJggg=="
    data = base64.b64decode(b64)
    fd, path = tempfile.mkstemp(suffix=".png")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def test_dalle_diffusion_inherits_image_readers():
    engine = DalleDiffusion(provider=Provider.OPENAI)
    path = _temp_png_path()
    try:
        data, mime = engine.read_image_bytes_and_mime(path)
        assert isinstance(data, (bytes, bytearray)) and len(data) > 0
        assert mime.startswith("image/")
        data_url = engine.to_image_data_url(path)
        assert data_url.startswith("data:image/")
    finally:
        os.remove(path)


def test_vertex_imagen_inherits_image_readers():
    engine = VertexImagen(provider=Provider.VERTEX)
    path = _temp_png_path()
    try:
        data, mime = engine.read_image_bytes_and_mime(path)
        assert isinstance(data, (bytes, bytearray)) and len(data) > 0
        assert mime.startswith("image/")
        data_url = engine.to_image_data_url(path)
        assert data_url.startswith("data:image/")
    finally:
        os.remove(path)
