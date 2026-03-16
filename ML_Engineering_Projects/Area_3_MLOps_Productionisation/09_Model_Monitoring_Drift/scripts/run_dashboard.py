"""Launch the Streamlit monitoring dashboard.

Usage:
    uv run python scripts/run_dashboard.py
    uv run streamlit run src/dashboard.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    dashboard_path = Path(__file__).parent.parent / "src" / "dashboard.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        check=True,
    )


if __name__ == "__main__":
    main()
