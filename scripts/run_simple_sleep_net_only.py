"""
Train and evaluate only the Simple Sleep Net base experiment (DOD-H LOOV).

Run from the repository root, with the package installed or on PYTHONPATH.

Parallel folds (e.g. RunPod with several GPUs)
-----------------------------------------------
1. Build memmaps once (single process), without training folds::

     python scripts/run_simple_sleep_net_only.py --memmap-only

   Or build memmaps and run fold 0 only, without wiping existing outputs::

     python scripts/run_simple_sleep_net_only.py --folds 0 --no-force

2. Run one process per fold in parallel. Each must use ``--no-force`` so workers
   do not ``rmtree`` the shared experiment directory, and ``--skip-memmap-build``
   so they do not all rewrite memmaps at once::

     CUDA_VISIBLE_DEVICES=0 python scripts/run_simple_sleep_net_only.py --folds 0 --no-force --skip-memmap-build &
     CUDA_VISIBLE_DEVICES=1 python scripts/run_simple_sleep_net_only.py --folds 1 --no-force --skip-memmap-build &
     # ... one GPU per fold, or time-slice on one GPU.

On one GPU, parallel Python processes will contend for VRAM; prefer one process
per GPU or sequential folds.
"""
import argparse

from dreem_learning_open.settings import DODH_SETTINGS, EXPERIMENTS_DIRECTORY
from dreem_learning_open.utils.run_experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--folds',
        type=int,
        nargs='*',
        default=None,
        metavar='N',
        help='LOOV fold indices to run (default: all folds). Example: --folds 0 1 2',
    )
    parser.add_argument(
        '--skip-memmap-build',
        action='store_true',
        help='Assume H5→memmap already done; skip h5_to_memmaps (required for parallel workers).',
    )
    parser.add_argument(
        '--no-force',
        action='store_true',
        help='Do not delete existing outputs under the experiment save folder before running. '
             'Use for every parallel fold worker so they do not remove each other\'s runs.',
    )
    parser.add_argument(
        '--reuse-incomplete-uuids',
        action='store_true',
        help='When --no-force is used, reuse existing incomplete run UUIDs per fold instead '
             'of creating new UUID directories.',
    )
    parser.add_argument(
        '--eeg-only',
        action='store_true',
        help='Train on EEG channels only; save under EXPERIMENTS_DIRECTORY/<dataset>/simple_sleep_net_eeg/.',
    )
    parser.add_argument(
        '--memmap-only',
        action='store_true',
        help='Only ensure H5→memmap pipeline exists for this config (creates if missing); no fold training.',
    )
    args = parser.parse_args()

    datasets = {'dodh': DODH_SETTINGS}
    experiments_directory = 'scripts/base_experiments/'
    fold_to_run = args.folds if args.folds is not None else None

    run_experiments(
        ['simple_sleep_net'],
        experiments_directory,
        EXPERIMENTS_DIRECTORY,
        datasets=datasets,
        fold_to_run=fold_to_run,
        force=not args.no_force,
        skip_memmap_build=args.skip_memmap_build,
        reuse_incomplete_uuids=args.reuse_incomplete_uuids,
        eeg_only=args.eeg_only,
        memmap_only=args.memmap_only,
    )


if __name__ == '__main__':
    main()
