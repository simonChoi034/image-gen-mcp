from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    text: str


def http_post(url: str, body: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> HttpResponse:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:  # nosec - controlled by env
        content = resp.read().decode("utf-8")
        return HttpResponse(status=resp.status, headers=dict(resp.headers.items()), text=content)
