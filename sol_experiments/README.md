# Sleep onset latency (SOL) experiments

Staging **pretraining** lives under `scripts/` and uses **consensus** hypnogram labels. This folder is a **separate** pipeline: expert-derived SOL targets, per-fold **LOSOCV** evaluation against any pretrained model under `EXPERIMENTS_DIRECTORY/<dataset>/<model>/`, optional SOL-aware fine-tuning, and re-evaluation.

Artifacts go under **`BASE_DIRECTORY/sol/`** (see `settings.py`): dataset-wide targets, per-model per-fold evaluations, and per-fold fine-tuned runs.

## Stages

1. **Expert reference** — `compute_sol_targets.py`: one JSON per dataset (all records), default `sol/targets/<dataset>/sol_targets.json`.
2. **Model SOL (pre-finetune), per fold** — `evaluate_sol.py`: reads each fold’s `hypnograms.json`, writes `sol/evaluations/<dataset>/<model>/fold_XX/sol_eval.json` plus optional `summary.json`.
3. **Fine-tune on SOL, per fold** — `finetune_sol.py`: from the pretrained checkpoint for that fold; outputs under `sol/finetuned/<dataset>/<base_model>_ft_.../fold_XX/`.
4. **Re-evaluate** — run `evaluate_sol.py` again pointing at the fine-tuned run root or the same model name if outputs are merged as expected.

Fold indices and held-out **test_record** match pretraining (`experiment_fold_index`, `scripts/experiment_utils/index_experiments.py`).

## Examples (model-agnostic)

```bash
# Targets (any dataset with h5 + optional dreem-learning-evaluation)
python sol_experiments/compute_sol_targets.py --dataset dodh

# Evaluate a SimpleSleepNet LOOCV run
python sol_experiments/evaluate_sol.py --dataset dodh --model simple_sleep_net

# Evaluate CNN-RNN (same flags; model name is just the experiments folder)
python sol_experiments/evaluate_sol.py --dataset dodh --model cnn_rnn
```

Use `--base-experiments-dir` if your memmap configs live outside the default tree. Pretraining drivers include `scripts/run_cnn_rnn.py` and other `scripts/run_*` entry points — SOL does not assume a specific architecture.
