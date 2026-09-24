"""Run all tests."""
import subprocess, sys
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=str(root),
    )
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
