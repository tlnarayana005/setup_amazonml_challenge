"""Check GPU capabilities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging, get_logger

def main():
    setup_logging()
    log = get_logger("gpu")
    try:
        import torch
        log.info("PyTorch version: %s", torch.__version__)
        log.info("CUDA available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            log.info("CUDA version: %s", torch.version.cuda)
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                log.info("GPU %d: %s (%.2f GB VRAM)", i, props.name, props.total_mem / (1024**3))
            # Quick test
            x = torch.randn(100, 100, device="cuda")
            y = torch.matmul(x, x)
            log.info("GPU compute test PASSED (matmul on 100x100).")
        else:
            log.info("No CUDA GPU available. CPU-only mode.")
    except ImportError:
        log.warning("PyTorch not installed.")

if __name__ == "__main__":
    main()
