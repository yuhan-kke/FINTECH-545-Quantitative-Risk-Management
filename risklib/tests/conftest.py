# Lets pytest import risklib straight from src/ without pip install
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
