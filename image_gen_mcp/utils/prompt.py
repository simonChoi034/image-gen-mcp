from __future__ import annotations

from typing import Any

import jinja2

from ..shard.enums import Background, Model, Orientation, Quality, SizeCode

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
<task>
{{ prompt | trim }}
</task>

{% if fold.negative_prompt %}
<avoid>
{{ fold.negative_prompt }}
</avoid>
{% endif %}

{# Only render guidance if there is at least one folded field #}
{% if fold.orientation or fold.size or fold.quality or fold.background %}
<guidance>
{# Orientation guidance #}
{% if fold.orientation %}
<orientation>
{% if fold.orientation == "square" %}
- Orientation: Square (1:1). Use when you want centered compositions, avatars, icons, or product thumbnails.
    Example phrasing: "Square format (1:1) — crop to a centered subject."
{% elif fold.orientation == "portrait" %}
- Orientation: Portrait (taller than wide; e.g., 3:4). Use for portraits, posters, or mobile vertical displays.
    Example phrasing: "Portrait orientation — taller than wide; focus on full body or head-and-shoulders."
{% elif fold.orientation == "landscape" %}
- Orientation: Landscape (wider than tall; e.g., 4:3 or wider). Use for landscapes, banners, or cinematic compositions.
    Example phrasing: "Landscape orientation — horizontally wide composition, broad field of view."
{% else %}
- Orientation: {{ fold.orientation }} (unrecognized). Prefer: square / portrait / landscape.
{% endif %}
</orientation>
{% endif %}

{# Size guidance #}
{% if fold.size %}
<size>
{% if fold.size == "S" %}
- Size: Small (S). Fast generation, lower detail. Approx: up to ~512 px on long side. Use for thumbnails or quick drafts.
    Example phrasing: "Approx size tier: S (small, fast, lower detail)."
{% elif fold.size == "M" %}
- Size: Medium (M). Balanced quality vs speed. Approx: ~1024 px on long side. Good default for most use-cases.
    Example phrasing: "Approx size tier: M (~1024 px long side; balanced quality)."
{% elif fold.size == "L" %}
- Size: Large (L). Highest detail, slower. Approx: 2048 px+ on long side; suitable for print or high-res output.
    Example phrasing: "Approx size tier: L (large, highest detail — slower)."
{% else %}
- Size: {{ fold.size }} (unrecognized). Prefer: S / M / L.
{% endif %}
</size>
{% endif %}

{# Quality guidance #}
{% if fold.quality %}
<quality>
{% if fold.quality == "draft" %}
- Quality: Draft. Prioritize speed and rough composition over fine detail and texture.
    Example phrasing: "Quality: draft — coarse detail, faster render."
{% elif fold.quality == "standard" %}
- Quality: Standard. Balanced settings for production-quality images.
    Example phrasing: "Quality: standard — balanced detail and generation time."
{% elif fold.quality == "high" %}
- Quality: High. Prioritize fine textures, realistic lighting, and high fidelity; expect longer render times.
    Example phrasing: "Quality: high — prioritize photoreal detail and fine textures."
{% else %}
- Quality: {{ fold.quality }} (unrecognized). Prefer: draft / standard / high.
{% endif %}
</quality>
{% endif %}

{# Background guidance #}
{% if fold.background %}
<background>
{% if fold.background == "transparent" %}
- Background: Transparent (alpha). Suitable for overlays and compositing; keep edges clean for cutouts.
    Example phrasing: "Background: transparent — preserve alpha where subject is not present."
{% elif fold.background == "opaque" %}
- Background: Opaque. Render with a solid/background environment (default). Specify a desired color or neutral backdrop if needed.
    Example phrasing: "Background: opaque — use plain neutral backdrop (e.g., white or light grey)."
{% else %}
- Background: {{ fold.background }} (unrecognized). Prefer: transparent / opaque.
{% endif %}
</background>
{% endif %}

{# Short summary line to suggest how to merge guidance into the task prompt #}
<note>
If folding these knobs into the textual instruction, append a short guidance phrase to the task
such as: "[guidance: orientation: portrait; approx size tier: M; quality: standard; background: transparent; avoid: no text]".
Keep guidance concise and prioritized (avoid long paragraphs).
</note>
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
