# Memar et al. classical baseline (`memar_et_al`)

Pejman Memar and Farhad Faradji, *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 26, no. 1, pp. 84–95, 2018.

This folder holds configuration for the **same Dreem memmap pipeline** as `simple_sleep_net` (DOD-H `dodh` block in `memmaps.json`), so **LOSO fold indices match** `simple_sleep_net` / `cnn_rnn` when using `scripts/experiment_utils` and `sol_experiments/evaluate_sol.py`.

**Training protocol** (paper Sec. VIII-B, subject cross-validation): leave one subject out for testing; **Kruskal–Wallis → mRMR → RF** use **all remaining subjects** (`epochs` pooled). No held-out validation split. Defaults: **`--mrmr-k 40`**, **`--n-estimators 100`**. `description.json` sets `memar_et_al_subject_cv: true` and empty `validation_records` (allowed by `indexed_run_complete` for this flag).

## Contents

| File | Role |
|------|------|
| `memmaps.json` | Identical `dodh` (and `dodo`) memmap spec as `simple_sleep_net` — **same SHA-1 hash** for `dodh`. |
| `memar_et_al_config.json` | **Single EEG derivation** for features: key `eeg_signal` (e.g. `signals/eeg/F4_O2`). Must match an entry in `memmaps.json` → `signals` → `eeg` → `signals`. |
| `bands.json` | Table I frequency bands (Hz); delta implemented as **4 Hz lowpass** (0 Hz highpass not realizable). |
| `dataset.json` | `temporal_context` 21 (for description parity with other base experiments). |
| `feature_names.json` | 104 names: 13 features × 8 bands (Hjorth **Activity** omitted as redundant with SD²; SD + HM + HC + …). |

## Equation → code mapping

Implementation: `dreem_learning_open/memar_et_al/features.py` (Section IV A–K), `selection.py` (Kruskal–Wallis + mRMR), `io.py` (memmap epochs, channel `signals/eeg/F4_O2`).

**mRMR:** The code tries PyPI `mrmr-selection` (`mrmr.mrmr_classif`) when `pandas` / `mrmr` import successfully; otherwise it uses a **greedy mutual-information minus correlation** mRMR surrogate (same interface). Install optional: `pip install mrmr-selection pandas`.

## Commands

From the repo root (after `settings.py` and H5 data under `BASE_DIRECTORY`):

```text
python scripts/run_memar_et_al.py --memmap-only
python scripts/run_memar_et_al.py
python scripts/run_memar_et_al.py --folds 0 1 --no-force --skip-memmap-build
python scripts/run_memar_et_al.py --workers 4 --no-force --skip-memmap-build
```

Outputs are written under ``experiments/dodh/memar_et_al/``. Use ``--eeg-only`` or ``--all-eeg-channels`` to write under ``experiments/dodh/memar_et_al_eeg/`` (same convention as other EEG-focused runs).

Parallel folds (``--workers N``) use one process per fold. Use ``--skip-memmap-build`` so workers do not all run ``h5_to_memmaps``; use ``--no-force`` so they do not delete each other’s outputs. RandomForest defaults to ``n_jobs=-1`` (all cores per fold); set ``--rf-n-jobs`` to limit CPU when using many parallel folds. ``--clear-feature-cache-after`` removes the extracted-feature cache when the run finishes.

Training logs six steps per fold (memmap index → feature extraction → KW → mRMR → RF fit → predict) with timings; ``tqdm`` bars run when ``--workers 1`` and not ``--quiet``. Use ``--quiet`` for warnings only.

Change the default channel by editing `memar_et_al_config.json`, or override for one run:

```text
python scripts/run_memar_et_al.py --eeg-signal signals/eeg/C3_M2
python scripts/run_memar_et_al.py --config path/to/custom_memar_et_al_config.json
```

Loader: `dreem_learning_open.memar_et_al.config.load_memar_et_al_config` / `get_eeg_signal`.

SOL evaluation (requires `hypnograms.json` under each run UUID):

```text
python sol_experiments/evaluate_sol.py --model memar_et_al --dataset dodh
```

## Run artifacts

Each fold writes a UUID directory under `EXPERIMENTS_DIRECTORY/dodh/memar_et_al/` with `description.json`, `hypnograms.json`, and `best_model.gz` (tar containing `memar_rf.joblib` — sklearn `RandomForestClassifier`, not `ModuloNet`).
