from __future__ import annotations

from typing import Any

import jinja2

from ..shard.enums import Background, Model, Orientation, Quality

# ---------------------------------------------------------------------------
# Jinja2 template to fold unsupported knobs into prompt guidance
# ---------------------------------------------------------------------------

# This guidance format matches adapters in docs/schema.md, e.g.:
# "[guidance: orientation: portrait; approx size tier: S; quality: draft; avoid: no text]"
_GUIDANCE_TEMPLATE = jinja2.Environment(
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
).from_string(
    """
{{ prompt | trim }}

{% if fold.negative_prompt %}
AVOID (must not appear):
{{ fold.negative_prompt }}
{% endif %}

{% if fold.orientation %}
{% if fold.orientation == "square" %}
Orientation: Square (1:1). Center the subject with comfortable padding.
If a seed canvas or input image is supplied, do not change the input aspect ratio.
Prefer phrasing like: "Square format (1:1); centered composition; avoid cropping key features."
{% elif fold.orientation == "portrait" %}
Orientation: Portrait 9:16 (vertical). Favor full‑height framing for mobile views.
If a seed canvas or input image is supplied, do not change the input aspect ratio.
Suggest phrasing: "Portrait 9:16 — vertical framing; keep the full subject within frame."
{% elif fold.orientation == "landscape" %}
Orientation: Landscape 16:9 (horizontal). Use a wide composition with clear foreground/midground/background.
If a seed canvas or input image is supplied, do not change the input aspect ratio.
Suggest phrasing: "Landscape 16:9 — wide composition; maintain a stable horizon."
{% else %}
Orientation: {{ fold.orientation }} (unrecognized). Prefer: square (1:1) / portrait (9:16) / landscape (16:9).
{% endif %}
{% endif %}

{% if fold.quality %}
{% if fold.quality == "draft" %}
Detail level: Draft — prioritize speed and overall composition over micro‑detail; keep the description brief and concrete.
{% elif fold.quality == "standard" %}
Detail level: Standard — balanced fidelity suitable for production assets; avoid verbose style tags.
{% elif fold.quality == "high" %}
Detail level: High — emphasize fine textures and realistic lighting; allow longer render time.
{% else %}
Detail level: {{ fold.quality }} (unrecognized). Prefer: draft / standard / high.
{% endif %}
{% endif %}

{% if fold.background %}
{% if fold.background == "transparent" %}
Background: Transparent (alpha). Keep edges clean for compositing; avoid halos and stray pixels.
{% elif fold.background == "opaque" %}
Background: Opaque. Use a simple neutral backdrop or a coherent environment as specified in the task.
{% else %}
Background: {{ fold.background }} (unrecognized). Prefer: transparent / opaque.
{% endif %}
{% endif %}

Guidance: Write a short, natural sentence (not a tag list) that covers subject, composition, lighting, mood, background, and style.
Prefer plain language over comma‑separated keywords.
If strict orientation is required and a seed canvas is supplied at 1:1, 9:16, or 16:9, do not change the input aspect ratio.
"""
)


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
