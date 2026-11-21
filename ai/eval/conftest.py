"""Make ai/ imports consistent when regression tests run from the repository root."""
import sys
from pathlib import Path

AI = Path(__file__).resolve().parents[1]
if str(AI) not in sys.path:
    sys.path.insert(0, str(AI))
