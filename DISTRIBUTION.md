# Distribution Setup Guide

This document describes the complete pipeline setup for distributing the image-gen-mcp package to PyPI.

## Overview

The project now includes:

- ✅ Dynamic versioning using `uv-dynamic-versioning`
- ✅ GitHub Actions CI/CD pipeline
- ✅ Automated PyPI publishing on releases
- ✅ Multi-version Python testing (3.12+)
- ✅ Package build and installation verification

## Features Implemented

### 1. Dynamic Versioning

- **Tool**: `uv-dynamic-versioning` with hatch backend
- **Configuration**: Automatic version from git tags
- **Format**: Semantic versioning (v1.2.3)
- **Fallback**: 0.1.0 for development

### 2. GitHub Actions Workflows

#### CI Pipeline (`.github/workflows/ci.yml`)

- **Triggers**: Push to main/develop, pull requests
- **Python versions**: 3.12, 3.13
- **Checks**: Linting, type checking, testing, security scan
- **Artifacts**: Built packages for further use

#### Release Pipeline (`.github/workflows/release.yml`)

- **Trigger**: GitHub release published
- **Process**: Build → Test → TestPyPI → PyPI
- **Verification**: Package installation and import tests
- **Assets**: Wheel and source distribution attached to release

#### Manual Release (`.github/workflows/manual-release.yml`)

- **Trigger**: Manual workflow dispatch
- **Options**: patch/minor/major/prerelease version bumps
- **Features**: Dry run mode, automatic changelog generation

#### Dependency Updates (`.github/workflows/dependencies.yml`)

- **Trigger**: Weekly schedule + manual
- **Process**: Update uv.lock, run tests, create PR

### 3. Package Configuration

#### pyproject.toml Updates

- Proper PyPI metadata (author, classifiers, keywords)
- Dynamic versioning configuration
- Hatch build system with src layout
- Console script entry point

#### Build System

- **Backend**: hatchling with uv-dynamic-versioning
- **Structure**: `image_gen_mcp/` package directory
- **Exclusions**: Tests, docs, development files

## Usage Instructions

### For Users

#### Installation from PyPI

```bash
# Regular installation
pip install image-gen-mcp

# With uv
uv add image-gen-mcp

# With uvx (recommended for MCP usage)
uvx --from image-gen-mcp image-gen-mcp
```

#### MCP Integration

Add to your `mcp.json`:

```json
{
  "mcpServers": {
    "image-gen-mcp": {
      "command": "uvx",
      "args": ["--from", "image-gen-mcp", "image-gen-mcp"],
      "env": {
        "OPENAI_API_KEY": "your-key-here"
      }
    }
  }
}
```

### For Maintainers

#### Creating a Release

1. **Automated (Recommended)**:

   - Go to GitHub Actions
   - Run "Manual Release" workflow
   - Choose version type (patch/minor/major)
   - Wait for completion

1. **Manual**:

   ```bash
   # Create and push tag
   git tag v1.0.0
   git push origin v1.0.0

   # Create GitHub release from the web interface
   ```

#### Local Development

```bash
# Setup
uv sync --all-extras --dev

# Test local build
uv build
./test-installation.sh

# Run tests
uv run pytest -v

# Linting
uv run ruff check .
uv run black --check .
uv run pyright
```

## Security Configuration

### GitHub Secrets Required

None! The pipeline uses PyPI trusted publishing via OIDC tokens.

### Trusted Publishing Setup

1. Go to PyPI → Account settings → Publishing
1. Add trusted publisher:
   - **Owner**: simonChoi034
   - **Repository**: image-gen-mcp
   - **Workflow**: release.yml
   - **Environment**: pypi

## Pipeline Validation

The complete pipeline has been tested and verified:

- ✅ Dynamic versioning works correctly
- ✅ Package builds successfully
- ✅ Installation and imports work
- ✅ Console script is functional
- ✅ CI pipeline passes all checks
- ✅ Release workflow is configured

## Troubleshooting

### Common Issues

1. **Build fails with "Unable to determine which files to ship"**

   - Ensure `[tool.hatch.build.targets.wheel]` has correct package path

1. **Version not updating**

   - Check git tags are formatted as `v1.2.3`
   - Ensure full git history is available (fetch-depth: 0)

1. **Import errors after installation**

   - Verify package name consistency (hyphens vs underscores)
   - Check entry point configuration

### Testing Locally

```bash
# Test complete pipeline
./test-installation.sh

# Test specific version
git tag v0.2.0
uv build
```

## Next Steps

The pipeline is ready for production use. Future enhancements could include:

- Code coverage reporting integration
- Automated security scanning with alerts
- Release notes automation from commit messages
- Multi-architecture builds if needed
- Container image publishing

For questions or issues, refer to the GitHub repository or create an issue.
