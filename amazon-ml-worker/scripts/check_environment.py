"""Check environment: Python, packages, hardware."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hardware.detect import detect_hardware, detect_environment_type, print_report, save_report
from src.utils.logging import setup_logging

def main():
    setup_logging()
    report = detect_hardware()
    report["environment_type"] = detect_environment_type()
    print_report(report)
    save_report(report, "outputs/logs/environment_report.json")
    print(f"\nEnvironment type: {report['environment_type']}")
    print("Report saved to: outputs/logs/environment_report.json")

if __name__ == "__main__":
    main()
