from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from seq_common import run_baseline


if __name__ == "__main__":
    run_baseline("transformer", univariate=False)
