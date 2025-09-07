from __future__ import annotations

_CAPABILITY_TIP = " Tip: Use the 'get_model_capabilities' tool to check enabled providers and models for your current credentials."


def _looks_like_auth_or_capability_issue(text: str) -> bool:
    """Best-effort detection for auth/capability issues from provider errors.

    We avoid importing SDK-specific exception types and instead match common
    substrings that appear in provider error messages when credentials are
    missing/invalid or when a model/provider is unavailable or not enabled.
    """
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
    """Append a capability tip to the message when appropriate.

    Ensures we don't duplicate the tip on repeated calls.
    """
    if not message:
        return message
    if _CAPABILITY_TIP.strip() in message:
        return message
    if _looks_like_auth_or_capability_issue(message):
        return message.rstrip() + _CAPABILITY_TIP
    return message


__all__ = ["augment_with_capability_tip"]
