from __future__ import annotations

import base64
import os
import tempfile

from src.engines.ar.openrouter import OpenRouterAR
from src.shard.enums import Provider


def _make_temp_png() -> str:
    # 1x1 transparent PNG
    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB/9k3WQAAAABJRU5ErkJggg=="
    data = base64.b64decode(b64)
    fd, path = tempfile.mkstemp(suffix=".png")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def test_as_image_url_from_path_wraps_with_data_url():
    engine = OpenRouterAR(provider=Provider.OPENROUTER)
    path = _make_temp_png()
    try:
        url = engine._as_image_url(path)
        assert url.startswith("data:image/png;base64,")
    finally:
        os.remove(path)


def test_as_image_url_passthrough_data_url():
    engine = OpenRouterAR(provider=Provider.OPENROUTER)
    data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB/9k3WQAAAABJRU5ErkJggg=="
    assert engine._as_image_url(data_url) == data_url


def test_as_image_url_bare_base64_wraps():
    engine = OpenRouterAR(provider=Provider.OPENROUTER)
    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB/9k3WQAAAABJRU5ErkJggg=="
    url = engine._as_image_url(b64)
    assert url.startswith("data:image/png;base64,")


def test_base_engine_validation_rejects_non_image(tmp_path):
    engine = OpenRouterAR(provider=Provider.OPENROUTER)
    p = tmp_path / "not_image.txt"
    p.write_text("hello world")
    try:
        engine.to_image_data_url(str(p))
        assert False, "Expected validation to fail for non-image"
    except ValueError:
        pass
