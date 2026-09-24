# Quick Start Guide

Get running in 5 minutes on any machine.

## 1. Prerequisites

- Python 3.9+
- pip

## 2. Install Base Dependencies

```bash
pip install -r requirements.txt
```

**Optional** (install only what your worker needs):
```bash
pip install -r requirements-ocr.txt    # OCR workers
pip install -r requirements-cv.txt     # CV workers
pip install -r requirements-nlp.txt    # NLP workers
pip install -r requirements-gpu.txt    # GPU workers
```

## 3. Verify Environment

```bash
python scripts/check_environment.py
python scripts/check_gpu.py  # if you have a GPU
```

## 4. Generate Synthetic Data (Pre-Competition)

```bash
python scripts/generate_synthetic_data.py
```

## 5. Run Tests

```bash
python -m pytest tests/ -v
```

## 6. Run Your First Worker

```bash
# ETL on synthetic data
python scripts/run_worker.py --task etl --input-path synthetic_data

# Baseline on synthetic data
python scripts/run_worker.py --task baseline --input-path synthetic_data --enable-baseline
```

## 7. Kaggle Setup

```python
# In a Kaggle notebook cell:
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                       "pyyaml", "tqdm", "pyarrow"])

# Then import and use:
sys.path.insert(0, "/kaggle/working/amazon-ml-worker")
from src.config.loader import WorkerConfig
```

## What's Next?

- Read [WORKER_GUIDE.md](WORKER_GUIDE.md) for detailed operations
- Read [DAY1_CHECKLIST.md](DAY1_CHECKLIST.md) when the competition starts
