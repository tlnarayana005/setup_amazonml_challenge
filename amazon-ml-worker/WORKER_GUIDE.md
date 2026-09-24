# Worker Guide

Detailed guide for operating workers in the Amazon ML Challenge 2026.

## Core Concept

**Same codebase, different configuration.** Every machine runs the identical source code. Worker behavior is controlled entirely through:

1. YAML config files
2. CLI arguments  
3. Environment variables

## Worker Roles

### ETL Worker
Runs once to clean and preprocess the raw dataset.

```bash
python scripts/run_worker.py --task etl --input-path data/raw
```

### OCR Worker (Shardable)
Extracts text from images. Distribute across N machines.

```bash
# Machine 1
python scripts/run_worker.py --task ocr --worker-id 0 --total-workers 8 --enable-ocr

# Machine 2
python scripts/run_worker.py --task ocr --worker-id 1 --total-workers 8 --enable-ocr
```

### CV Worker (Shardable)
Image processing and optional embedding extraction.

```bash
python scripts/run_worker.py --task cv --worker-id 0 --total-workers 4 --enable-cv
```

### NLP Worker (Shardable)
Text cleaning, normalization, and feature extraction.

```bash
python scripts/run_worker.py --task nlp --worker-id 0 --total-workers 3 --enable-nlp
```

### Embedding Worker (Shardable)
Text and/or image embedding extraction.

```bash
python scripts/run_worker.py --task embeddings --worker-id 0 --total-workers 5 --enable-embeddings
```

### Baseline Worker
Trains and evaluates a classical ML model.

```bash
python scripts/run_worker.py --task baseline --enable-baseline --model-type lightgbm
```

### Inference Worker (Shardable)
Runs batch prediction with a trained model.

```bash
python scripts/run_worker.py --task inference --worker-id 0 --total-workers 4
```

## Logical Worker IDs

Worker IDs are **logical**, not physical. Any machine can run any worker ID.

If laptop #5 fails, another laptop can take over its worker_id:

```bash
# Originally on laptop #5:
python scripts/run_worker.py --worker-id 4 --total-workers 8 --task ocr

# Now on laptop #12 (same command, same result):
python scripts/run_worker.py --worker-id 4 --total-workers 8 --task ocr
```

## Sharding

For shardable tasks, data is split deterministically:

- `hash(id) % total_workers` assigns each row to exactly one worker
- No overlap between workers
- Complete coverage guaranteed
- Stable across restarts (same IDs, same shard)

## Caching & Resume

All expensive operations are cached. If a worker crashes:

1. Restart with the same command
2. Cache detects completed IDs
3. Only remaining IDs are processed

Enable/disable with `--resume` / `--no-resume`.

## Worker Output Contract

Every worker produces:

```
outputs/worker_results/worker_{id}/
├── config.json      # Worker configuration
├── metrics.json     # Task metrics
├── runtime.json     # Timing information
├── output.parquet   # Main output (stable IDs)
├── errors.csv       # Error log
└── README.txt       # Worker summary
```

## Merging Worker Outputs

After all workers finish:

```bash
python scripts/merge_workers.py --input-dir outputs/worker_results
```

This verifies:
- Schema compatibility across workers
- No duplicate IDs
- No missing IDs
- Expected row counts

## Resource Safety

Before processing, the worker prints a workload summary:

```
task:          ocr
worker_id:     0 / 8
row count:     12500 (shard)
batch_size:    32
GPU:           None
cache status:  3200 cached, 9300 remaining
```

Configure limits: `--max-rows 1000 --max-runtime 3600 --batch-size 16`

## Failure Recovery

| Scenario | Recovery |
|----------|----------|
| Kaggle session timeout | Restart — cache resumes |
| Notebook restart | Re-run cell — cache resumes |
| Internet loss | Local processing continues |
| Machine restart | Restart worker — cache resumes |
| Partial processing | Restart — only remaining IDs processed |

## Configuration Hierarchy

```
WorkerConfig defaults
    ↓ overridden by
configs/default.yaml
    ↓ overridden by
configs/tasks/{task}.yaml
    ↓ overridden by
configs/worker.yaml
    ↓ overridden by
CLI arguments (--worker-id, --task, etc.)
    ↓ overridden by
Environment variables (WORKER_ID, TOTAL_WORKERS, WORKER_TASK)
```

## Day-1 Schema Configuration

After reading the problem statement, update these fields:

```yaml
# configs/default.yaml (or via CLI)
id_column: product_id          # actual ID column name
target_column: category        # actual target column name
group_column: brand            # if group-aware split needed
metric: f1_macro               # actual competition metric
text_column: description       # actual text column name
image_column: image_path       # actual image column name
```
