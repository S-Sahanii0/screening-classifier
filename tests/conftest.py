import sys
from pathlib import Path

# let tests import the fake client and the package without installing
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
