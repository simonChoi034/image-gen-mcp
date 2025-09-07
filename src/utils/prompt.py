from __future__ import annotations

from typing import Any

import jinja2

from ..shard.enums import Background, Model, Orientation, Quality, SizeCode

# ---------------------------------------------------------------------------
# Jinja2 template to fold unsupported knobs into prompt guidance
# ---------------------------------------------------------------------------

# This guidance format matches adapters in docs/schema.md, e.g.:
# "[guidance: orientation: portrait; approx size tier: S; quality: draft; avoid: no text]"
_GUIDANCE_TEMPLATE = jinja2.Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True).from_string(
    """
<task>
{{ prompt | trim }}
</task>

{% if fold.negative_prompt %}
<avoid>
{{ fold.negative_prompt }}
</avoid>
{% endif %}

{% if fold.orientation or fold.size or fold.quality or fold.background %}
<guidance>
    {% if fold.orientation %}<orientation>{{ fold.orientation }}</orientation>{% endif %}
    {% if fold.size %}<size>{{ fold.size }}</size>{% endif %}
    {% if fold.quality %}<quality>{{ fold.quality }}</quality>{% endif %}
    {% if fold.background %}<background>{{ fold.background }}</background>{% endif %}
</guidance>
{% endif %}
"""
)


def _as_str(val: Any | None) -> str | None:
    return None if val is None else str(val)


def _unsupported_knobs_for_model(model: Model) -> set[str]:
    """Return unified fields that should be folded for the given model.

    Only returns fields among: size, orientation, quality, background, negative_prompt.
    """
    # Defaults: conservative — fold nothing
    unsupported: set[str] = set()

    if model == Model.GPT_IMAGE_1:
        # OpenAI AR supports size/orientation, quality, background; negative_prompt is not native.
        unsupported = {"negative_prompt"}
    elif model in {Model.GEMINI_IMAGE_PREVIEW, Model.OPENROUTER_GOOGLE_GEMINI_IMAGE}:
        # Gemini image preview (native or via OpenRouter) exposes no explicit knobs.
        unsupported = {"size", "orientation", "quality", "background", "negative_prompt"}
    else:
        # Diffusion models aren't AR; callers typically won't use this path for them.
        unsupported = set()

    return unsupported


def render_prompt_with_guidance(
    *,
    prompt: str,
    model: Model,
    size: SizeCode | None = None,
    orientation: Orientation | None = None,
    quality: Quality | None = None,
    background: Background | None = None,
    negative_prompt: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Render a prompt augmented with guidance for fields unsupported by the model.

    Returns a tuple of (rendered_prompt, normalization_log) where normalization_log
    includes whether augmentation occurred and which fields were folded.
    """
    to_fold = _unsupported_knobs_for_model(model)

    fold: dict[str, Any] = {}
    used: list[str] = []

    if "orientation" in to_fold and orientation is not None:
        fold["orientation"] = orientation.value
        used.append("orientation")
    if "size" in to_fold and size is not None:
        fold["size"] = size.value
        used.append("size")
    if "quality" in to_fold and quality is not None:
        fold["quality"] = quality.value
        used.append("quality")
    if "background" in to_fold and background is not None:
        fold["background"] = background.value
        used.append("background")
    if "negative_prompt" in to_fold and negative_prompt:
        fold["negative_prompt"] = negative_prompt
        used.append("negative_prompt")

    rendered = _GUIDANCE_TEMPLATE.render(prompt=prompt, fold=fold)

    normlog = {
        "prompt_augmented": bool(used),
        "guidance_fields": used,
    }
    return rendered, normlog


__all__ = [
    "render_prompt_with_guidance",
]
