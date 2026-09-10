# Adds the project root to sys.path so tests can import engine/scripts
# modules the same way the CLI scripts do (see e.g. scripts/predict.py's
# own sys.path.insert), without needing the project installed as a package.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
