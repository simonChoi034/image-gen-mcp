from __future__ import annotations

import os
import sys

# Add repository root to sys.path for `import image_gen_mcp.*` in tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
