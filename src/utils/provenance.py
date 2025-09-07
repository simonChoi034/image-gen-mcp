from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_provenance(engine: str, model: str | None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    p = {
        "engine": engine,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        p.update(extra)
    return p
