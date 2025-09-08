from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..shard.enums import Provider


_CAPABILITY_TIP = " Tip: Use the 'get_model_capabilities' tool to check enabled providers and models for your current credentials."


def _looks_like_auth_or_capability_issue(text: str) -> bool:
    """Best-effort detection for auth/capability issues from provider errors."""
    if not text:
        return False
    lower = text.lower()

    keywords = [
        # auth/credentials
        "api key",
        "apikey",
        "invalid key",
        "missing key",
        "no api key",
        "unauthorized",
        "forbidden",
        "permission",
        "not allowed",
        "access denied",
        "access is denied",
        "credentials",
        "auth",
        "401",
        "403",
        # billing/quota
        "billing",
        "quota",
        # capability/provider disabled
        "provider not enabled",
        "provider disabled",
        "not enabled",
        "disabled",
        "no providers are enabled",
        # routing/model mapping
        "no engine available",
        "not supported by",
        "unsupported provider",
        "unsupported model",
    ]

    return any(k in lower for k in keywords)


def augment_with_capability_tip(message: str) -> str:
    """Append a capability tip to the message when appropriate."""
    if not message:
        return message
    if _CAPABILITY_TIP.strip() in message:
        return message
    if _looks_like_auth_or_capability_issue(message):
        return message.rstrip() + _CAPABILITY_TIP
    return message


# ============================================================================
# Engine Factory Errors
# ============================================================================


class EngineFactoryError(ValueError):
    """Base exception for engine factory errors."""


class ProviderUnavailableError(EngineFactoryError):
    """Raised when a provider is not enabled (missing credentials)."""

    def __init__(self, provider: Provider):
        self.provider = provider
        super().__init__(f"Provider '{provider.value}' is not enabled (missing credentials).")


class EngineResolutionError(EngineFactoryError):
    """Raised when no suitable engine can be found for the inputs."""

    def __init__(self, message: str):
        super().__init__(message)


__all__ = [
    "augment_with_capability_tip",
    "EngineFactoryError",
    "ProviderUnavailableError",
    "EngineResolutionError",
]
