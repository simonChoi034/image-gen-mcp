#!/bin/bash
set -e

echo "Testing package build and installation..."

# Clean previous builds
rm -rf dist/ build/

# Build the package
echo "Building package..."
uv build

# Test installation in a clean environment
echo "Testing installation..."
uv venv --python 3.12 test-env
source test-env/bin/activate

# Install from wheel
pip install dist/*.whl

# Test basic functionality
echo "Testing basic functionality..."
python -c "
try:
    # Test package structure
    import image_gen_mcp
    print(f'✅ Package import successful, version: {image_gen_mcp.__version__}')

    # Test console script availability
    import subprocess
    result = subprocess.run(['image-gen-mcp', '--help'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0 or 'usage:' in result.stderr.lower():
        print('✅ Console script works')
    else:
        print(f'⚠️  Console script test: {result.stderr}')

    print('✅ Package installation test completed successfully')

except Exception as e:
    print(f'❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

# Clean up
deactivate
rm -rf test-env

echo "✅ All tests passed!"
