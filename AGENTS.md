# Repository Guidelines

## Project Structure & Module Organization

- `src/` — main package (MCP server). Key modules: `main.py` (entry), `schema.py` (Pydantic I/O), `engines/` (providers: OpenAI, Gemini, Vertex/Imagen, DALL·E), `utils/`, `shard/` (enums/constants), `settings.py` (env config).
- `tests/` — pytest tests (add new tests here).
- `scripts/` — helper workflows (feature scaffolding, plan checks).
- `templates/`, `docs/`, `memory/` — specs, docs, and working notes.

## Build, Test, and Development Commands

- Run server: `python -m src.main`
- With uv: `uv run python -m src.main` (installs/uses lockfile).
- Lint: `ruff check .` Format: `black .`
- Type-check: `pyright`
- Tests: `pytest -q`
- Feature scaffolding: `./scripts/create-new-feature.sh "image upscaling"`

## Coding Style & Naming Conventions

- Python 3.12, 4-space indent, type hints required for new/changed code.
- Modules: `lower_snake_case.py`; Classes: `CapWords`; Functions/vars: `snake_case`.
- Pydantic v2 models in `schema.py` define the public contract; keep field names and enums stable.
- Run `ruff` and `black` before pushing; keep diffs minimal and cohesive.

## Testing Guidelines

- Use `pytest`; place tests under `tests/` mirroring `src/` paths (e.g., `tests/engines/test_openai_ar.py`).
- Prefer unit tests around engines and factory routing; mock network calls.
- No coverage gate yet; include tests for new engines and bug fixes.

## Commit & Pull Request Guidelines

- Branches: `NNN-short-title` (e.g., `003-gemini-edits`). Scripts in `scripts/` assume this.
- Commits: follow Conventional Commits (e.g., `feat(engines): add vertex imagen`).
- PRs: include a clear description, linked issues/spec (`specs/NNN-*/`), test plan (`pytest` output), and screenshots or sample payloads if UI/tooling changes.

## Security & Configuration Tips

- Set only what you use. Common env vars:
  - `OPENAI_API_KEY`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`
  - Vertex: `VERTEX_PROJECT`, `VERTEX_LOCATION`, `VERTEX_CREDENTIALS_PATH`
- Validate with capabilities: call MCP tool `get_model_capabilities` to confirm enabled providers.

## Agent-Specific Notes

- Exposed tools: `generate_image`, `edit_image`, `get_model_capabilities`.
- Call tools with named parameters (e.g., `generate_image(prompt="...", provider="openai", model="dall-e-3", ...)`). Arguments are not wrapped in a single request object.
- The `provider` and `model` parameters are required for `generate_image` and `edit_image`.
- Use `get_model_capabilities` to discover available providers and models before calling generation or editing tools.
- Prefer provider/model-agnostic fields where possible.
