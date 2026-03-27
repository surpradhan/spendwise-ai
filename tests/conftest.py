"""Add repo root to sys.path so `scripts.*` imports work from tests/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
