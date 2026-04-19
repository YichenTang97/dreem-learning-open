# Sleep onset latency (SOL) experiments

SOL experiments are a separate, model-agnostic pipeline built on top of pretrained
sleep staging models.

Pretrained staging runs live under `EXPERIMENTS_DIRECTORY/<dataset>/<model>/`
(for example `simple_sleep_net`, `cnn_rnn`, or any other model folder).
SOL artifacts are written under `BASE_DIRECTORY/sol/`.

## Rationale

- Pretraining and SOL are different targets:
  - pretraining optimizes epoch-level sleep staging labels (consensus hypnograms),
  - SOL experiments evaluate and optimize sleep onset latency against expert-derived SOL references.
- The pipeline is architecture-agnostic: SOL scripts consume saved fold artifacts and do not assume a specific network.
- LOSO/LOOCV parity is required: SOL evaluation and SOL finetuning must keep the same held-out record and fold index as pretraining so model comparisons are valid.
- Human variability matters: scorer disagreement is tracked in SOL targets and gives context for expected model error.

## SOL pipeline (4 steps)

1. **Build expert SOL references** (`compute_sol_targets.py`)
   - Produces dataset-wide targets in `sol/targets/<dataset>/`.
   - Main file: `sol_targets.json` with per-record scorer SOLs and `consensus_sol_min`.
   - Also writes:
     - `consensus_sol_targets.json` (`record_id -> consensus_sol_min`),
     - `consensus_hypnograms.json` (`record_id -> consensus hypnogram epochs`).
   - When scorer files are available in `dreem-learning-evaluation/scorers/<dataset>/scorer_*`,
     per-scorer SOL is computed from those labels. `consensus_sol_min` is the mean of individual scorer SOLs.

2. **Evaluate pretrained model on SOL (per fold)** (`evaluate_sol.py`)
   - Reads each fold run's `hypnograms.json`.
   - Computes model-predicted SOL from predicted hypnograms using the configured AASM rule
     (`--require_consecutive`, default 1 = first non-wake epoch).
   - Compares predicted SOL vs expert consensus SOL reference.
   - Writes per-fold outputs to `sol/evaluations/<dataset>/<model>/fold_XX/sol_eval.json`,
     plus optional rollup `summary.json`.

3. **SOL-aware finetuning (per fold)** (`finetune_sol.py`)
   - Starts from each fold's pretrained checkpoint.
   - Optimizes a combined staging + SOL objective while preserving LOSO fold identity.
   - Writes fold outputs under `sol/finetuned/<dataset>/<base_model>_<tag>/fold_XX/`.

4. **Re-evaluate after finetune (per fold)**
   - If outputs use the default tree `sol/finetuned/<dataset>/<subfolder>/`, run
     `evaluate_sol.py --finetuned --model <subfolder>` (same name as the directory;
     `exp_dir` is inferred from `sol_config.finetuned_run_dir`).
   - If you used a custom `--out_dir` for finetuning, add `--exp_dir` pointing at that root.
   - Results go under `sol/evaluations/<dataset>/<basename(exp_dir)>/` (distinct from
     pretrained `sol/evaluations/.../<base_model>/`).
   - Enables direct before/after SOL comparison with identical fold structure.

Fold identity is mapped from pretraining metadata so held-out record alignment is preserved.

## Scorer-vs-model SOL metric

The pipeline stores and reuses a side-by-side benchmark so human scorers and models
are evaluated on compatible SOL references.

### Definitions (plain language)

- **Record `r`**: one sleep recording (one subject-night).
- **Scorer `i`**: one human scorer (for example `scorer_1`).
- **`SOL_i(r)`**: SOL from scorer `i` on record `r`.
- **`MeanAll(r)`**: average SOL across all human scorers for record `r`.
- **`MeanLOO_i(r)`** ("leave-one-out mean"): average SOL across all human scorers
  except scorer `i`, for record `r`.

### Definitions (equations)

Let `S` be the set of human scorers and `R` the set of records.

- `MeanAll(r) = (1 / |S|) * sum_{j in S} SOL_j(r)`
- `MeanLOO_i(r) = (1 / (|S| - 1)) * sum_{j in S, j != i} SOL_j(r)`

### Human scorer benchmark (computed in `compute_sol_targets.py`)

For each scorer `i`, compute error against the leave-one-out reference on each record:

- `Err_i(r) = SOL_i(r) - MeanLOO_i(r)`
- `AbsErr_i(r) = |Err_i(r)|`

Then summarize those per-record errors by averaging **across records**:

- `MAE_i`: average absolute error across records.
- `RMSE_i`: root mean squared error across records.
- `Bias_i`: average signed error across records (positive = over-estimation).
- `StdErr_i`: standard deviation of signed errors across records.
- `Pearson_i = corr(SOL_i, MeanLOO_i)` when variance permits

Equivalent equation form:

- `MAE_i = (1 / |R_i|) * sum_{r in R_i} |Err_i(r)|`
- `RMSE_i = sqrt((1 / |R_i|) * sum_{r in R_i} Err_i(r)^2)`
- `Bias_i = (1 / |R_i|) * sum_{r in R_i} Err_i(r)`
- `StdErr_i = std({Err_i(r) : r in R_i})`

where `R_i` is the set of records with valid SOL for scorer `i` and its reference.

These per-scorer results are stored under:

- `sol_targets.json` -> `_scorer_vs_mean_benchmark.per_scorer_vs_loo_mean`

For each scorer `i`, the same aggregate metrics (`mae_min`, `rmse_min`, `bias_min`,
`std_err_min`, `pearson_r`) against **`MeanAll`** (the same reference as the model)
are stored under:

- `_scorer_vs_mean_benchmark.per_scorer_vs_mean_all`

Rollups across scorers (mean and standard deviation of each metric) are stored as:

- `_scorer_vs_mean_benchmark.human_aggregate_vs_loo_mean` — LOO reference (same
  metrics as per-scorer LOO, summarized across scorers). **This is the default
  human baseline in `evaluate_sol.py` prints** (alongside the model vs `MeanAll`).
- `_scorer_vs_mean_benchmark.human_aggregate_vs_mean_all` — `MeanAll` reference
  (same metric definitions as `model_vs_human_mean`, for optional same-reference analysis).

A compact MAE-only summary (LOO) is kept for backward compatibility:

- `_scorer_vs_mean_benchmark.human_baseline_mae.mean_mae_min`
- `_scorer_vs_mean_benchmark.human_baseline_mae.std_mae_min`

And the per-record human mean reference is stored for model comparison:

- `_scorer_vs_mean_benchmark.per_record_mean_sol_min` (this is `MeanAll(r)`).

### Model comparison (computed in `evaluate_sol.py`)

Model SOL predictions are compared against the stored human mean reference:

- `ModelErr(r) = SOL_model(r) - MeanAll(r)`
- Aggregate into MAE/RMSE/Bias/StdErr/Pearson exactly as above.

Equation form:

- `MAE_model = (1 / |R_m|) * sum_{r in R_m} |ModelErr(r)|`
- `RMSE_model = sqrt((1 / |R_m|) * sum_{r in R_m} ModelErr(r)^2)`
- `Bias_model = (1 / |R_m|) * sum_{r in R_m} ModelErr(r)`
- `StdErr_model = std({ModelErr(r) : r in R_m})`

where `R_m` is the set of records with valid model SOL and human mean reference.

This is reported as:

- `summary.json` -> `scorer_vs_mean_benchmark.model_vs_human_mean`

### Why this metric is useful

- Human scorers are not compared to themselves when using the LOO reference.
- Model comparison uses a stable dataset-wide human reference (`MeanAll`).
- **`evaluate_sol.py` prints the human column using the LOO framework** (rollup
  `human_aggregate_vs_loo_mean`): same metric definitions as for the model, but
  the human baseline matches the standard inter-rater benchmark (scorer vs mean
  of the other scorers). The model column remains vs `MeanAll`.
- For same-reference-as-model human summaries (each scorer vs `MeanAll`), see
  `human_aggregate_vs_mean_all` and `per_scorer_vs_mean_all` in `sol_targets.json`.
- Human and model results are on the same error scale (minutes), enabling direct
  interpretation of whether model performance is within expert variability.

## In summary

Base sleep models are trained for staging first, then a separate SOL pipeline is run:
build expert SOL targets, evaluate held-out fold predictions against those targets,
fine-tune each fold for SOL, and re-evaluate. Keeping outputs under
`BASE_DIRECTORY/sol/` with per-fold LOSO directories ensures alignment with pretraining
and supports fair comparison across models and against expert variability.

## Examples

```bash
# 1) Build SOL targets (uses scorer repo when available)
python sol_experiments/compute_sol_targets.py --dataset dodh

# 2) Evaluate pretrained model folds on SOL
python sol_experiments/evaluate_sol.py --dataset dodh --model simple_sleep_net

# 3) Fine-tune for SOL
python sol_experiments/finetune_sol.py --dataset dodh --base_model simple_sleep_net

# 4) Re-evaluate fine-tuned outputs (default layout: --model = finetuned subfolder name)
python sol_experiments/evaluate_sol.py --dataset dodh --model simple_sleep_net_ft_c10m_a0.50 --finetuned
```
