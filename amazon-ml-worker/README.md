# Amazon ML Worker

> **One codebase. Many workers. Configuration-driven behavior.**

A reusable ML worker repository for the Amazon ML Challenge 2026. The **same source code** runs on ~20 independent machines, with each machine's behavior controlled entirely through configuration.

## Architecture

```
SAME SOURCE CODE
       |
 worker configuration (YAML + CLI)
       |
 +-----+-----+-----+-----+-----+-----+
 |     |     |     |     |     |     |
ETL   OCR   CV    NLP  EMB   ML   INF
 |     |     |     |     |     |     |
 +-----+-----+-----+-----+-----+-----+
               |
         worker outputs
               |
           merge later
               |
        complete dataset
               |
       heavy training repo
               |
          final model
               |
          submission
```

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Check environment
python scripts/check_environment.py

# 3. Generate synthetic data (for testing before real data)
python scripts/generate_synthetic_data.py

# 4. Run tests
python -m pytest tests/ -v

# 5. Run a worker
python scripts/run_worker.py --task etl --input-path synthetic_data
```

## Worker Examples

### OCR Worker (Machine 1)
```bash
python scripts/run_worker.py \
  --task ocr --worker-id 0 --total-workers 8 \
  --enable-ocr --ocr-backend tesseract
```

### OCR Worker (Machine 2)
```bash
python scripts/run_worker.py \
  --task ocr --worker-id 1 --total-workers 8 \
  --enable-ocr --ocr-backend tesseract
```

### CV Worker
```bash
python scripts/run_worker.py \
  --task cv --worker-id 0 --total-workers 4 \
  --enable-cv --enable-embeddings
```

### NLP Worker
```bash
python scripts/run_worker.py \
  --task nlp --worker-id 0 --total-workers 3 \
  --enable-nlp
```

### ETL Worker
```bash
python scripts/run_worker.py --task etl
```

### Baseline Worker
```bash
python scripts/run_worker.py --task baseline --enable-baseline \
  --model-type lightgbm --input-path data/raw
```

**All examples use exactly the same codebase.** Only configuration differs.

## Configuration

Configuration is layered (later overrides earlier):

1. `configs/default.yaml` — defaults
2. `configs/tasks/{task}.yaml` — task-specific
3. `configs/worker.yaml` — per-worker overrides
4. CLI arguments
5. Environment variables (`WORKER_ID`, `TOTAL_WORKERS`, `WORKER_TASK`)

### Key Configuration Fields

| Field | Description | Default |
|-------|-------------|---------|
| `worker_id` | Logical worker index (0-based) | 0 |
| `total_workers` | Total number of workers for sharding | 1 |
| `task` | Worker task: etl, ocr, cv, nlp, embeddings, baseline, inference | etl |
| `id_column` | ID column name (set on Day-1) | id |
| `target_column` | Target column name (set on Day-1) | target |
| `metric` | Evaluation metric (set on Day-1) | f1_macro |
| `max_rows` | Row limit (-1 = unlimited) | -1 |
| `seed` | Random seed | 42 |

## Project Structure

```
amazon-ml-worker/
├── configs/            # YAML configuration files
│   ├── default.yaml    # Global defaults
│   ├── worker.yaml     # Per-worker overrides
│   └── tasks/          # Task-specific configs
├── src/                # Source code
│   ├── config/         # Configuration loader
│   ├── hardware/       # Hardware detection
│   ├── data/           # Data inspection, validation, split, shard, merge
│   ├── etl/            # ETL pipeline, preprocessing, caching
│   ├── ocr/            # OCR abstraction + backends
│   ├── cv/             # Image utilities + embeddings
│   ├── nlp/            # Text utilities + embeddings
│   ├── features/       # Feature builder + merger
│   ├── models/         # Baseline, lightweight, QLoRA smoke
│   ├── evaluation/     # Metrics + threshold search
│   ├── inference/      # Batch prediction
│   ├── experiments/    # Experiment tracker
│   └── utils/          # Logging, seeds, paths, timing
├── scripts/            # Entry-point scripts
├── tests/              # Test suite
├── synthetic_data/     # Synthetic data generator
├── notebooks/          # Jupyter/Kaggle notebooks
├── data/               # Raw → interim → processed
├── features/           # Feature outputs by type
├── outputs/            # Predictions, metrics, logs, worker results
├── cache/              # Processing cache
└── experiments/        # Experiment tracking CSV
```

## Sharding

Workers process non-overlapping data shards deterministically:

```
worker_id=0, total_workers=3 → processes shard 0 of 3
worker_id=1, total_workers=3 → processes shard 1 of 3
worker_id=2, total_workers=3 → processes shard 2 of 3
```

Sharding is based on `hash(id) % total_workers`. This guarantees:
- **Deterministic**: Same input → same shards
- **No overlap**: Each row assigned to exactly one worker
- **Complete coverage**: All rows covered
- **Stable across restarts**: Can resume safely

## Merging Worker Outputs

```bash
python scripts/merge_workers.py --input-dir outputs/worker_results
```

Verifies: schema compatibility, no duplicate IDs, no missing IDs, correct row counts.

## Parallelizable Tasks

| Task | Shardable? | Description |
|------|-----------|-------------|
| OCR | ✅ | Extract text from images |
| CV/Image features | ✅ | Image preprocessing, embeddings |
| NLP/Text features | ✅ | Text cleaning, embeddings |
| Embeddings | ✅ | Text/image embedding extraction |
| Feature generation | ✅ | Per-row feature engineering |
| Inference | ✅ | Batch prediction |
| ETL | ❌ | Run once, share output |
| Baseline training | ❌ | Train on full data |

## Caching

All expensive operations (OCR, embeddings, image processing) are cached:
- Workers skip already-completed IDs on restart
- Cache is versioned by config hash and data version
- Safe to restart after crashes/Kaggle session termination

## Dependencies

| File | Purpose |
|------|---------|
| `requirements.txt` | Base (all workers) |
| `requirements-ocr.txt` | OCR workers only |
| `requirements-cv.txt` | CV workers only |
| `requirements-nlp.txt` | NLP workers only |
| `requirements-gpu.txt` | GPU workers only |

## Security

- **Never commit**: API keys, Kaggle tokens, HF tokens, cloud credentials
- Use `.env` file (gitignored) or platform secrets
- See `.env.example` for template

## Documentation

- [QUICKSTART.md](QUICKSTART.md) — 5-minute setup guide
- [WORKER_GUIDE.md](WORKER_GUIDE.md) — Detailed worker operations
- [DAY1_CHECKLIST.md](DAY1_CHECKLIST.md) — Competition day-1 action plan
