"""QLoRA smoke test runner."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.qlora_smoke import run_qlora_smoke_test, check_qlora_dependencies
from src.utils.logging import setup_logging, get_logger

def main():
    setup_logging()
    log = get_logger("qlora_smoke")

    deps = check_qlora_dependencies()
    log.info("QLoRA dependencies: %s", json.dumps(deps, indent=2))

    missing = [k for k, v in deps.items() if not v]
    if missing:
        log.warning("Missing: %s. Smoke test will be skipped.", missing)

    report = run_qlora_smoke_test()
    print(f"\nQLoRA smoke test status: {report['status']}")
    if report.get("reason"):
        print(f"Reason: {report['reason']}")

if __name__ == "__main__":
    main()
