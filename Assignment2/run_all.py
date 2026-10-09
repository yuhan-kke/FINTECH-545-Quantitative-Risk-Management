# Runs every problem and saves its printout to output/problemN.txt
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
(HERE / "output").mkdir(exist_ok=True)
for i in range(1, 6):
    out = subprocess.run([sys.executable, str(HERE / f"problem{i}.py")],
                         capture_output=True, text=True, check=True).stdout
    (HERE / "output" / f"problem{i}.txt").write_text(out)
    print(out)
