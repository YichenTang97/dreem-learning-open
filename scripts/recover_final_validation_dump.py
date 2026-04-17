"""
Temporary utility: scan experiment run folders; where best_model.gz exists but
hypnograms.json and/or description final fields are missing, re-run the same
final validation + dump as logger.log_experiment (test-set validate, metrics,
records_split, hypnograms.json, updated description.json).

Requires a readable description.json with dataset_parameters.split (train/val/test),
groups_description, features_description, trainers_parameters, and on-disk memmap
record paths resolvable from --repo-root.

Does not re-train. By default does not rewrite best_model.gz (use --rewrite-best-model
to mirror logger exactly).

If description.json has an empty ``features_description`` but the checkpoint uses
extra memmap features (e.g. ``epoch_index``), feature specs are taken from the
checkpoint and merged in (and written back into description on success).

Also prints every run folder that has no ``best_model.gz`` or a file that is not a
valid ModuloNet checkpoint tar (``missing`` / ``invalid_or_corrupt_tar``).

Usage:
  python scripts/recover_final_validation_dump.py --dry-run
  python scripts/recover_final_validation_dump.py --dataset dodh --algo cnn_rnn
  python scripts/recover_final_validation_dump.py --runs-root path/to/dodh/cnn_rnn
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tarfile
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dreem_learning_open.datasets.dataset import DreemDataset
from dreem_learning_open.models.modulo_net.net import ModuloNet
from dreem_learning_open.settings import EXPERIMENTS_DIRECTORY
from dreem_learning_open.trainers.trainer import Trainer


def is_valid_checkpoint_tar(path: str) -> bool:
    try:
        return os.path.isfile(path) and tarfile.is_tarfile(path)
    except OSError:
        return False


def _deep_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_json_safe(v) for v in obj]
    # NumPy scalars (Trainer metrics) without importing numpy at module load
    if hasattr(obj, "item") and callable(getattr(obj, "item")) and not isinstance(
        obj, (bytes, str, dict, list)
    ):
        try:
            obj = obj.item()
        except Exception:
            pass
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def resolve_record_path(raw: str, repo_root: str) -> str:
    raw_norm = os.path.normpath(raw.replace("/", os.sep))
    if os.path.isdir(raw_norm):
        return raw_norm
    joined = os.path.normpath(os.path.join(repo_root, raw_norm))
    if os.path.isdir(joined):
        return joined
    raise FileNotFoundError("Record path not found: {!r} (tried {!r})".format(raw, joined))


def resolve_split_paths(paths: List[str], repo_root: str) -> List[str]:
    out: List[str] = []
    for p in paths:
        if not isinstance(p, str):
            raise TypeError("split entry must be str, got {!r}".format(type(p)))
        out.append(resolve_record_path(p, repo_root))
    return out


def description_outputs_complete(description: Dict[str, Any]) -> bool:
    perf = description.get("performance_on_test_set")
    if not isinstance(perf, dict) or len(perf) == 0:
        return False
    pr = description.get("performance_per_records")
    if not isinstance(pr, dict) or len(pr) == 0:
        return False
    rs = description.get("records_split")
    if not isinstance(rs, dict):
        return False
    for key in ("train_records", "validation_records", "test_records"):
        if key not in rs or not isinstance(rs[key], list) or len(rs[key]) == 0:
            return False
    return True


def iter_run_dirs(
    experiments_root: str,
    dataset_filter: Optional[str],
    algo_filter: Optional[str],
) -> Iterator[Tuple[str, str, str, str]]:
    for dataset in sorted(os.listdir(experiments_root)):
        if dataset_filter and dataset != dataset_filter:
            continue
        dpath = os.path.join(experiments_root, dataset)
        if not os.path.isdir(dpath):
            continue
        for algo in sorted(os.listdir(dpath)):
            if algo_filter and algo != algo_filter:
                continue
            apath = os.path.join(dpath, algo)
            if not os.path.isdir(apath):
                continue
            for run_id in sorted(os.listdir(apath)):
                if run_id.startswith("."):
                    continue
                rpath = os.path.join(apath, run_id)
                if os.path.isdir(rpath):
                    yield dataset, algo, run_id, rpath


def recover_run(
    run_dir: str,
    repo_root: str,
    dry_run: bool,
    backup: bool,
    rewrite_best_model: bool,
    verbose_validate: bool,
) -> Tuple[str, Optional[str]]:
    """
    Returns (status, detail). status in:
    ok | skip | error | dry_run | incomplete_best_model
    """
    model_path = os.path.join(run_dir, "best_model.gz")
    hyp_path = os.path.join(run_dir, "hypnograms.json")
    desc_path = os.path.join(run_dir, "description.json")

    if not os.path.isfile(model_path):
        return "incomplete_best_model", "missing"
    if not is_valid_checkpoint_tar(model_path):
        return "incomplete_best_model", "invalid_or_corrupt_tar"

    hyp_ok = os.path.isfile(hyp_path)
    if not os.path.isfile(desc_path):
        return "skip", "no description.json"

    try:
        with open(desc_path, "r", encoding="utf-8-sig") as f:
            description = json.load(f)
    except Exception as exc:
        return "error", "description load: {}".format(exc)

    if not isinstance(description, dict):
        return "error", "description is not an object"

    if hyp_ok and description_outputs_complete(description):
        return "skip", None

    dp = description.get("dataset_parameters")
    if not isinstance(dp, dict):
        return "error", "missing dataset_parameters"
    split = dp.get("split")
    if not isinstance(split, dict):
        return "error", "missing dataset_parameters.split"
    for key in ("train", "val", "test"):
        if key not in split or not isinstance(split[key], list) or len(split[key]) == 0:
            return "error", "missing or empty split[{}]".format(key)

    tp = description.get("trainers_parameters")
    if not isinstance(tp, dict) or "args" not in tp:
        return "error", "missing trainers_parameters.args"

    groups_description = description.get("groups_description")
    if not isinstance(groups_description, dict):
        return "error", "missing groups_description"
    desc_features = description.get("features_description")
    if desc_features is None:
        desc_features = {}
    if not isinstance(desc_features, dict):
        return "error", "features_description must be a dict"

    try:
        train_records = resolve_split_paths(split["train"], repo_root)
        validation_records = resolve_split_paths(split["val"], repo_root)
        test_records = resolve_split_paths(split["test"], repo_root)
    except FileNotFoundError as exc:
        return "error", str(exc)

    for i, record in enumerate(test_records):
        if record in train_records or record in validation_records:
            return "error", "test record overlaps train/val: {}".format(record)

    if dry_run:
        return "dry_run", None

    if backup:
        bak = desc_path + ".bak.{}".format(int(time.time()))
        shutil.copy2(desc_path, bak)

    try:
        best_net = ModuloNet.load(model_path)
    except Exception as exc:
        return "error", "ModuloNet.load failed: {}".format(exc)

    # Checkpoint may list features (e.g. epoch_index) while description.json has {}.
    # DreemDataset must load the same memmap features ModuloNet.get_args expects.
    features_description = dict(best_net.features) if best_net.features else {}
    features_description.update(desc_features)

    dataset_test = DreemDataset(
        groups_description,
        features_description=features_description,
        transform_parameters=dp.get("transform_parameters"),
        temporal_context=dp["temporal_context"],
        temporal_context_mode=dp["temporal_context_mode"],
        records=test_records,
    )

    trainer_save_folder = os.path.join(run_dir, "training")
    os.makedirs(trainer_save_folder, exist_ok=True)
    trainer = Trainer(
        net=best_net,
        save_folder=trainer_save_folder,
        **tp["args"],
    )

    try:
        performance_on_test_set, _, performance_per_records, hypnograms = trainer.validate(
            dataset_test,
            return_metrics_per_records=True,
            verbose=verbose_validate,
        )
    except Exception as exc:
        return "error", "validate failed: {}".format(exc)

    performance_per_records = {
        os.path.split(record)[-2]: metric for record, metric in performance_per_records.items()
    }

    records_split = {
        "train_records": [os.path.basename(os.path.normpath(r)) for r in train_records],
        "validation_records": [os.path.basename(os.path.normpath(r)) for r in validation_records],
        "test_records": [os.path.basename(os.path.normpath(r)) for r in test_records],
    }

    meta = description.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        description["metadata"] = meta
    if not meta.get("end"):
        meta["end"] = int(time.time())

    if features_description:
        description["features_description"] = features_description

    description["performance_on_test_set"] = _deep_json_safe(performance_on_test_set)
    description["performance_per_records"] = _deep_json_safe(performance_per_records)
    description["records_split"] = records_split

    with open(desc_path, "w", encoding="utf-8") as f:
        json.dump(description, f, indent=4)

    padding = 0
    for group in groups_description:
        padding = groups_description[group]["padding"] // 30

    if padding > 0:
        hyp_out = {k: x["predicted"][padding:-padding] for k, x in hypnograms.items()}
    else:
        hyp_out = {k: x["predicted"] for k, x in hypnograms.items()}

    with open(hyp_path, "w", encoding="utf-8") as f:
        json.dump(hyp_out, f, indent=4)

    if rewrite_best_model:
        best_net.save(model_path)

    return "ok", None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-root",
        default=EXPERIMENTS_DIRECTORY,
        help="Root containing <dataset>/<algo>/<run_id>/",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Single runs directory (e.g. .../dodh/cnn_rnn). Overrides experiments-root walk.",
    )
    parser.add_argument(
        "--repo-root",
        default=_REPO_ROOT,
        help="Project root for resolving relative memmap paths in description",
    )
    parser.add_argument("--dataset", default=None, help="Only this dataset name")
    parser.add_argument("--algo", default=None, help="Only this algorithm folder")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print runs that would be recovered without writing",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not copy description.json to description.json.bak.<ts> before overwrite",
    )
    parser.add_argument(
        "--rewrite-best-model",
        action="store_true",
        help="Re-save best_model.gz after dump (same as logger; usually unnecessary)",
    )
    parser.add_argument(
        "--verbose-validate",
        action="store_true",
        help="Forward verbose=True to validate (more console output)",
    )
    args = parser.parse_args()

    counts = {
        "ok": 0,
        "skip": 0,
        "error": 0,
        "dry_run": 0,
        "incomplete_best_model": 0,
    }

    if args.runs_root:
        run_dirs: List[Tuple[str, str, str, str]] = []
        base = os.path.normpath(args.runs_root)
        parts = base.replace("\\", "/").rstrip("/").split("/")
        if len(parts) < 2:
            print("runs-root should end with .../<dataset>/<algo>", file=sys.stderr)
            return 2
        algo = parts[-1]
        dataset = parts[-2]
        for name in sorted(os.listdir(base)):
            if name.startswith("."):
                continue
            rpath = os.path.join(base, name)
            if os.path.isdir(rpath):
                run_dirs.append((dataset, algo, name, rpath))
    else:
        run_dirs = list(
            iter_run_dirs(args.experiments_root, args.dataset, args.algo)
        )

    for dataset, algo, run_id, run_path in run_dirs:
        label = "{}/{}/{}".format(dataset, algo, run_id)
        status, err = recover_run(
            run_path,
            repo_root=os.path.abspath(args.repo_root),
            dry_run=args.dry_run,
            backup=not args.no_backup,
            rewrite_best_model=args.rewrite_best_model,
            verbose_validate=args.verbose_validate,
        )
        counts[status] = counts.get(status, 0) + 1
        if status == "incomplete_best_model":
            print(
                "{}  no completed best_model.gz ({})".format(label, err or "unknown")
            )
            continue
        if status == "skip":
            continue
        if status == "error":
            print("{}  ERROR: {}".format(label, err))
            continue
        if status == "dry_run":
            print("{}  would recover (missing hyp or incomplete description)".format(label))
            continue
        print("{}  recovered".format(label))

    print(
        "Summary: ok={} dry_run={} skipped={} no_completed_best_model={} errors={}".format(
            counts.get("ok", 0),
            counts.get("dry_run", 0),
            counts.get("skip", 0),
            counts.get("incomplete_best_model", 0),
            counts.get("error", 0),
        )
    )
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
