"""Pytest lifecycle hooks for deterministic resource cleanup."""

import sys
from pathlib import Path


PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def pytest_sessionfinish(session, exitstatus):
    """Close PyTorch's import-time temporary directory before interpreter exit.

    PyTorch creates this directory as a module global.  With warnings promoted to
    errors, leaving it to Python's interpreter shutdown emits a ResourceWarning.
    """
    instantiator = sys.modules.get("torch.distributed.nn.jit.instantiator")
    if instantiator is None:
        return

    temp_dir = getattr(instantiator, "_TEMP_DIR", None)
    if temp_dir is not None:
        temp_dir.cleanup()
