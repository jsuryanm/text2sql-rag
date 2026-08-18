"""
conftest.py (Shared pytest setup)
Pytest automatically loads this file before it collects/runs test. 
Its a standard place: 
    - fixtures shared across multiple test files.
    - one-time setup that must happen before you test files import the application code they're testing 
"""

import sys 
import types 
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def _register_fake_module(dotted_name: str, **attrs) -> types.ModuleType:
    """Insert a fake module into sys.modules """
