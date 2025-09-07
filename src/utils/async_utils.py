from __future__ import annotations

import inspect
from typing import Any


async def maybe_await(value: Any) -> Any:
    """Await a value if it's awaitable; otherwise return it as-is.

    Useful for handling engines that may implement sync or async methods.
    """
    if inspect.isawaitable(value):
        return await value  # type: ignore[no-any-return]
    return value


__all__ = ["maybe_await"]
