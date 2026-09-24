# Day-1 Checklist — Amazon ML Challenge 2026

Execute these steps in order when the competition opens and the real dataset is released.

## Phase 1: Understand the Problem (30 min)

- [ ] **1. Read the full problem statement carefully**
- [ ] **2. Identify the target variable** → update `target_column` in config
- [ ] **3. Identify the evaluation metric** → update `metric` in config
- [ ] **4. Identify modalities**: text? images? structured? multimodal?
- [ ] **5. Identify the ID column** → update `id_column` in config
- [ ] **6. Identify any group column** (e.g., brand, user) → update `group_column`
- [ ] **7. Note the number of classes** (classification) or value range (regression)
- [ ] **8. Note the train/test size**

## Phase 2: Data Inspection (30 min)

```bash
# Copy competition data to data/raw/
# Then inspect:
python scripts/inspect_data.py --input data/raw/train.csv --id-column <ID> --target-column <TARGET>
```

- [ ] **9. Run data inspection** → review `outputs/logs/inspection_report.json`
- [ ] **10. Check for duplicates** — duplicate rows, duplicate IDs
- [ ] **11. Check for data leakage** — train/test overlap
- [ ] **12. Check missing values** — which columns, how many
- [ ] **13. Check text columns** — length distribution, languages
- [ ] **14. Check image columns** — missing images, corrupted images
- [ ] **15. Decide split method**: random, stratified, or group-aware

## Phase 3: Create Validation Set (15 min)

```bash
# Update configs/default.yaml with correct column names, then:
python scripts/run_worker.py --task etl --input-path data/raw \
  --id-column <ID> --target-column <TARGET> --split-method stratified
```

- [ ] **16. Create a fixed validation set** (same seed across all workers)
- [ ] **17. Verify no leakage** between train and validation

## Phase 4: Quick Baseline (30 min)

```bash
python scripts/run_worker.py --task baseline --input-path data/raw \
  --enable-baseline --model-type logistic --id-column <ID> --target-column <TARGET>
```

- [ ] **18. Run logistic regression baseline**
- [ ] **19. Run LightGBM baseline**
- [ ] **20. Record baseline scores** — this is the bar to beat

## Phase 5: Decide Worker Assignments (15 min)

Based on data inspection:

- [ ] **21. Is OCR relevant?** (images with text)
- [ ] **22. Is CV relevant?** (images with visual content)
- [ ] **23. Is NLP relevant?** (text columns)
- [ ] **24. Are embeddings worth extracting?**

## Phase 6: Assign Workers (15 min)

Example assignment for 8 machines:

| Worker ID | Task | Machine |
|-----------|------|---------|
| 0-3 | OCR | Laptops 1-4 |
| 0-1 | CV | Laptops 5-6 |
| 0-1 | NLP | Laptops 7-8 |

- [ ] **25. Update `configs/worker.yaml`** on each machine
- [ ] **26. Distribute to all team members**

## Phase 7: Run Workers in Parallel (hours)

```bash
# Each machine runs its assigned task:
python scripts/run_worker.py --task <TASK> --worker-id <ID> --total-workers <N>
```

- [ ] **27. Launch all workers**
- [ ] **28. Monitor progress** — check cache status, error logs
- [ ] **29. Cache expensive outputs** — OCR, embeddings save automatically

## Phase 8: Merge & Evaluate (30 min)

```bash
python scripts/merge_workers.py --input-dir outputs/worker_results
```

- [ ] **30. Merge all worker outputs**
- [ ] **31. Verify merge integrity** — no duplicates, no missing IDs
- [ ] **32. Build features from merged data**
- [ ] **33. Re-evaluate baseline with new features**

## Phase 9: Short Model Experiments (1-2 hours)

```bash
python scripts/train_lightweight.py --input-path data/processed \
  --model-type lightgbm --max-rows 5000
```

- [ ] **34. Run lightweight screening experiments**
- [ ] **35. Compare feature combinations**
- [ ] **36. Identify promising directions**

## Phase 10: Scale Up (hours-days)

- [ ] **37. Move promising models to strong GPU resources**
- [ ] **38. Train on complete relevant training data**
- [ ] **39. Evaluate on fixed validation set**

## Phase 11: Ensemble & Submit (30 min)

- [ ] **40. Ensemble only when validation improves**
- [ ] **41. Validate submission format**
- [ ] **42. Verify IDs match test set**
- [ ] **43. Submit**

## Quick Reference: Config Update Template

```yaml
# configs/default.yaml — update these on Day 1:
id_column: <ACTUAL_ID>
target_column: <ACTUAL_TARGET>
group_column: <ACTUAL_GROUP_OR_EMPTY>
metric: <ACTUAL_METRIC>
text_column: <ACTUAL_TEXT_COL>
image_column: <ACTUAL_IMAGE_COL>
split_method: stratified  # or group if groups exist
```

## Emergency Procedures

| Problem | Solution |
|---------|----------|
| Worker crashes | Restart — cache resumes |
| Wrong column names | Update config, re-run |
| Data too large | Use `--max-rows` to limit |
| GPU OOM | Reduce `--batch-size` |
| OCR too slow | Add more workers, increase `--total-workers` |
