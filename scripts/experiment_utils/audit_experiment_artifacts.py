"""
Detect experiment run folders where training likely finished (per description.json)
but artifacts are missing, empty, or suspicious — e.g. after a bad sync that
overwrote newer files.

``rclone copy`` without ``--update`` (-u) can replace *newer* destination files with
older source files. Recovery: pull from the other side (R2) if that copy is still
intact, or use R2 object versioning / backups if both sides were clobbered.

**Core trio** (what most workflows need): ``description.json``, ``hypnograms.json``,
``best_model.gz``. The script **cross-checks** them (experiment id vs folder name,
hypnogram keys vs ``dataset_parameters.split.test``).

**Optional local fix** (``--fix`` / ``--apply``): only ``description.json`` — strip a
UTF-8 BOM and rewrite clean UTF-8 JSON if present. It does **not** copy
``training/best_net`` onto ``best_model.gz`` (those must stay authoritative; use
remote restore if needed).

This script does not call rclone.

Examples:
  python scripts/experiment_utils/audit_experiment_artifacts.py
  python scripts/experiment_utils/audit_experiment_artifacts.py --dataset dodh --algo cnn_rnn
  python scripts/experiment_utils/audit_experiment_artifacts.py --runs-root D:/data/experiments/dodh/cnn_rnn
  python scripts/experiment_utils/audit_experiment_artifacts.py --json-out audit.json
  python scripts/experiment_utils/audit_experiment_artifacts.py --core-detail
  python scripts/experiment_utils/audit_experiment_artifacts.py --fix --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dreem_learning_open.settings import EXPERIMENTS_DIRECTORY
from dreem_learning_open.utils.indexed_run_complete import (
    check_indexed_run_complete as check_run_complete,
)

# Artifacts checked for missing-file listing (same as indexed fold completion)
REQUIRED_REL_PATHS = [
    "description.json",
    "hypnograms.json",
    "best_model.gz",
]

# Below this size, treat as likely truncated / not a real checkpoint export
MIN_BEST_MODEL_BYTES = 1024
MIN_HYPNOGRAMS_BYTES = 32

# ModuloNet.save() writes an uncompressed tar (see modulo_net/net.py), despite .gz name.


def is_valid_checkpoint_tar(path: str) -> bool:
    try:
        return os.path.isfile(path) and tarfile.is_tarfile(path)
    except OSError:
        return False


def find_training_best_net(run_dir: str) -> Optional[str]:
    """Prefer training/best_net; fall back to legacy trainingbest_net (see logger.py)."""
    candidates = [
        os.path.join(run_dir, "training", "best_net"),
        os.path.join(run_dir, "trainingbest_net"),
    ]
    for p in candidates:
        if is_valid_checkpoint_tar(p):
            return p
    return None


def load_json_file(path: str) -> Tuple[Optional[Any], Optional[str]]:
    """Return (data, error). Tries utf-8-sig to strip BOM."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f), None
    except Exception as exc:
        return None, str(exc)


def expected_hypnogram_key_basename(description: dict) -> Optional[str]:
    """
    Trainer stores hypnograms under os.path.split(record)[-2], i.e. the parent
    directory of the test record folder (memmap hash dir), not the record UUID.
    Match using dataset_parameters.split.test[0] when present.
    """
    dp = description.get("dataset_parameters")
    if not isinstance(dp, dict):
        return None
    split = dp.get("split")
    if not isinstance(split, dict):
        return None
    test = split.get("test")
    if not isinstance(test, list) or len(test) < 1:
        return None
    p0 = test[0]
    if not isinstance(p0, str):
        return None
    # Same as trainer.validate: key is parent path of record; basename is memmap hash folder.
    return os.path.basename(os.path.dirname(os.path.normpath(p0)))


def hypnograms_match_expected_key(hyp_data: Any, expected_parent_basename: Optional[str]) -> Tuple[bool, List[str]]:
    """
    Check hypnograms keys against the parent-dir basename from split.test.
    Supports:
    - logger format: { key: [predicted stages...] }  (key = path to memmap hash dir)
    - nested format: { key: {"predicted": [...], "target": [...] } }
    """
    issues: List[str] = []
    if expected_parent_basename is None:
        issues.append("cannot_check_hypnograms_no_split_test_in_description")
        return False, issues
    if not isinstance(hyp_data, dict) or len(hyp_data) == 0:
        issues.append("hypnograms_empty_or_not_object")
        return False, issues

    for k, v in hyp_data.items():
        kb = os.path.basename(os.path.normpath(str(k)))
        if kb != expected_parent_basename:
            continue
        if isinstance(v, dict):
            if "predicted" in v:
                return True, issues
            issues.append("hypnogram_value_dict_missing_predicted")
            return False, issues
        if isinstance(v, list):
            return True, issues
        issues.append("hypnogram_value_unexpected_type")
        return False, issues

    issues.append(
        "no_hypnogram_key_for_split_test_parent:{!r}".format(expected_parent_basename)
    )
    return False, issues


def cross_check_core_triplet(
    run_id: str, description: Optional[dict], hyp_data: Optional[Any]
) -> List[str]:
    issues: List[str] = []
    if not isinstance(description, dict):
        return issues

    exp_id = (description.get("metadata") or {}).get("experiment_id")
    if exp_id is not None and str(exp_id) != str(run_id):
        issues.append("metadata.experiment_id_mismatch_folder:{}_vs_{}".format(exp_id, run_id))

    exp_parent = expected_hypnogram_key_basename(description)
    if hyp_data is not None:
        ok, hyp_issues = hypnograms_match_expected_key(hyp_data, exp_parent)
        issues.extend(hyp_issues)
        if not ok and not hyp_issues:
            issues.append("hypnograms_split_test_check_failed")

    return issues


def audit_core_files(run_dir: str, run_id: str) -> Dict[str, Any]:
    """Validate description.json, hypnograms.json, best_model.gz + cross-checks."""
    out: Dict[str, Any] = {
        "description": {},
        "hypnograms": {},
        "best_model": {},
        "checkpoint_source": None,
        "cross_check_issues": [],
        "core_ok": False,
    }

    desc_path = os.path.join(run_dir, "description.json")
    hyp_path = os.path.join(run_dir, "hypnograms.json")
    model_path = os.path.join(run_dir, "best_model.gz")

    desc_dict: Optional[dict] = None
    hyp_data: Optional[Any] = None

    # description.json
    if not os.path.isfile(desc_path):
        out["description"] = {"readable": False, "error": "missing"}
    else:
        data, err = load_json_file(desc_path)
        if err:
            out["description"] = {"readable": False, "error": err}
        else:
            desc_dict = data if isinstance(data, dict) else None
            out["description"] = {
                "readable": True,
                "size_bytes": file_size(desc_path),
                "has_metadata_end": bool((desc_dict.get("metadata") or {}).get("end"))
                if desc_dict
                else False,
                "has_performance_on_test_set": bool(
                    isinstance((desc_dict or {}).get("performance_on_test_set"), dict)
                    and len((desc_dict or {}).get("performance_on_test_set") or {}) > 0
                )
                if desc_dict
                else False,
            }

    # hypnograms.json
    if not os.path.isfile(hyp_path):
        out["hypnograms"] = {"readable": False, "error": "missing"}
    else:
        hdata, herr = load_json_file(hyp_path)
        if herr:
            out["hypnograms"] = {"readable": False, "error": herr}
        else:
            hyp_data = hdata
            out["hypnograms"] = {
                "readable": True,
                "size_bytes": file_size(hyp_path),
                "n_keys": len(hdata) if isinstance(hdata, dict) else 0,
            }

    # best_model.gz (ModuloNet tar)
    ckpt = find_training_best_net(run_dir)
    if ckpt:
        out["checkpoint_source"] = os.path.relpath(ckpt, run_dir).replace("\\", "/")

    if not os.path.isfile(model_path):
        out["best_model"] = {"valid_tar": False, "error": "missing", "size_bytes": None}
    else:
        sz = file_size(model_path)
        valid = is_valid_checkpoint_tar(model_path)
        out["best_model"] = {
            "valid_tar": valid,
            "size_bytes": sz,
            "error": None if valid else "not_a_valid_tar_checkpoint",
        }
        if sz is not None and sz < MIN_BEST_MODEL_BYTES and valid:
            out["best_model"]["warning"] = "small_file"

    out["cross_check_issues"] = cross_check_core_triplet(run_id, desc_dict, hyp_data)

    core_ok = (
        out["description"].get("readable")
        and out["hypnograms"].get("readable")
        and out["best_model"].get("valid_tar")
        and len(out["cross_check_issues"]) == 0
    )
    if out["description"].get("readable") and out["hypnograms"].get("readable"):
        exp_parent = expected_hypnogram_key_basename(desc_dict) if desc_dict else None
        if exp_parent:
            hm_ok, _ = hypnograms_match_expected_key(hyp_data, exp_parent)
            if not hm_ok:
                core_ok = False
    out["core_ok"] = bool(core_ok)

    return out


def apply_safe_fixes(run_dir: str, apply: bool) -> Dict[str, Any]:
    """
    Safe local fixes: description.json UTF-8 BOM strip only.
    Does not modify best_model.gz or hypnograms.json.
    """
    actions: List[str] = []
    preview: List[str] = []

    path = os.path.join(run_dir, "description.json")
    if not os.path.isfile(path):
        return {"apply": apply, "actions": actions, "would_do": preview}

    try:
        raw = open(path, "rb").read()
    except OSError:
        return {"apply": apply, "actions": actions, "would_do": preview}

    if raw.startswith(b"\xef\xbb\xbf"):
        msg = "strip_utf8_bom_and_rewrite description.json"
        preview.append(msg)
        if apply:
            text = raw.decode("utf-8-sig")
            data = json.loads(text)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, indent=4)
                f.write("\n")
            actions.append(msg)

    return {"apply": apply, "actions": actions, "would_do": preview}


def description_suggests_finished(description: dict) -> bool:
    """True if logs look like a finished run (end time + test metrics)."""
    if not isinstance(description, dict):
        return False
    meta = description.get("metadata") or {}
    if not meta.get("end"):
        return False
    perf = description.get("performance_on_test_set")
    if not isinstance(perf, dict) or len(perf) == 0:
        return False
    return True


def file_size(path: str) -> Optional[int]:
    try:
        return os.path.getsize(path) if os.path.isfile(path) else None
    except OSError:
        return None


def audit_run(run_dir: str, run_id: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "run_id": run_id,
        "run_path": run_dir.replace("\\", "/"),
        "category": "unknown",
        "index_complete": False,
        "index_reason": None,
        "description_suggests_finished": False,
        "missing_files": [],
        "suspicious_small_files": [],
        "notes": [],
    }
    out["core"] = audit_core_files(run_dir, run_id)

    desc_path = os.path.join(run_dir, "description.json")
    if not os.path.isfile(desc_path):
        out["category"] = "no_description"
        out["notes"].append("No description.json; cannot infer training status.")
        return out

    try:
        with open(desc_path, "r", encoding="utf-8-sig") as f:
            description = json.load(f)
    except Exception as exc:
        out["category"] = "description_unreadable"
        out["notes"].append("description.json parse error: {}".format(exc))
        return out

    out["description_suggests_finished"] = description_suggests_finished(description)

    complete, reason = check_run_complete(run_dir, description)
    out["index_complete"] = complete
    out["index_reason"] = reason

    for rel in REQUIRED_REL_PATHS:
        fp = os.path.join(run_dir, rel)
        if not os.path.isfile(fp):
            out["missing_files"].append(rel)

    for rel, min_b in (
        ("best_model.gz", MIN_BEST_MODEL_BYTES),
        ("hypnograms.json", MIN_HYPNOGRAMS_BYTES),
    ):
        fp = os.path.join(run_dir, rel)
        sz = file_size(fp)
        if sz is not None and sz < min_b:
            out["suspicious_small_files"].append({"path": rel, "size_bytes": sz, "min_expected": min_b})

    if complete:
        out["category"] = "ok"
        return out

    # Strong signal: description says done + has metrics, but artifacts bad/missing
    if out["description_suggests_finished"] and (out["missing_files"] or out["suspicious_small_files"]):
        out["category"] = "suspected_sync_damage"
        out["notes"].append(
            "metadata.end and performance_on_test_set present but artifacts incomplete; "
            "consistent with overwrite or partial copy."
        )
        return out

    if out["suspicious_small_files"] and not out["description_suggests_finished"]:
        out["category"] = "suspicious_small_files"
        return out

    if out["missing_files"] and out["description_suggests_finished"]:
        out["category"] = "suspected_sync_damage"
        return out

    out["category"] = "incomplete_or_failed"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dodh", help="Dataset name")
    parser.add_argument("--algo", default="cnn_rnn", help="Algorithm folder name")
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Override runs root (default: EXPERIMENTS_DIRECTORY/<dataset>/<algo>)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Write full report as JSON to this path",
    )
    parser.add_argument(
        "--print-rclone-recovery",
        action="store_true",
        help="Print example rclone copy lines (fill in remote:path yourself)",
    )
    parser.add_argument(
        "--remote-prefix",
        default="r2:your-bucket/path/to/experiments",
        help="Prefix for --print-rclone-recovery (no trailing slash)",
    )
    parser.add_argument(
        "--core-detail",
        action="store_true",
        help="Print core trio cross-check details for runs where core_ok is false",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Try safe fixes on description.json only (strip UTF-8 BOM if present)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="With --fix, actually write files (default: dry-run preview only)",
    )
    args = parser.parse_args()

    if args.apply and not args.fix:
        parser.error("--apply requires --fix")

    runs_root = args.runs_root or os.path.join(EXPERIMENTS_DIRECTORY, args.dataset, args.algo)
    if not os.path.isdir(runs_root):
        raise FileNotFoundError("Runs root does not exist: {!r}".format(runs_root))

    run_ids = sorted(
        name
        for name in os.listdir(runs_root)
        if os.path.isdir(os.path.join(runs_root, name))
        and not name.startswith(".")
    )

    rows: List[Dict[str, Any]] = []
    for run_id in run_ids:
        run_dir = os.path.join(runs_root, run_id)
        rows.append(audit_run(run_dir, run_id))

    fix_results_by_id: Dict[str, Dict[str, Any]] = {}
    if args.fix:
        for run_id in run_ids:
            run_dir = os.path.join(runs_root, run_id)
            fr = apply_safe_fixes(run_dir, args.apply)
            fix_results_by_id[run_id] = fr
        if args.apply:
            rows = []
            for run_id in run_ids:
                run_dir = os.path.join(runs_root, run_id)
                row = audit_run(run_dir, run_id)
                row["fix_result"] = fix_results_by_id.get(run_id, {})
                rows.append(row)
        else:
            for r in rows:
                r["fix_result"] = fix_results_by_id.get(r["run_id"], {})

    damaged = [r for r in rows if r["category"] == "suspected_sync_damage"]
    ok = [r for r in rows if r["category"] == "ok"]
    suspicious = [r for r in rows if r["category"] == "suspicious_small_files"]
    core_ok_runs = [r for r in rows if r.get("core", {}).get("core_ok")]
    core_bad = [r for r in rows if not r.get("core", {}).get("core_ok")]
    index_ok_core_bad = [r for r in rows if r["category"] == "ok" and not r.get("core", {}).get("core_ok")]
    not_index_complete = [r for r in rows if not r.get("index_complete")]
    core_ok_not_index = [
        r
        for r in rows
        if r.get("core", {}).get("core_ok") and not r.get("index_complete")
    ]

    print("Runs root:", runs_root)
    print("Run folders scanned:", len(rows))
    print("Index-complete (artifact + description checks):", len(ok))
    print("Core trio OK (description + hypnograms + best_model.tar + cross-checks):", len(core_ok_runs))
    print("Suspected sync damage (finished in description, bad/missing artifacts):", len(damaged))
    print("Suspicious small files (other):", len(suspicious))
    if index_ok_core_bad:
        print(
            "WARNING: index-complete but core check failed ({}): {}".format(
                len(index_ok_core_bad),
                ", ".join(r["run_id"] for r in index_ok_core_bad),
            )
        )
    if core_ok_not_index:
        print(
            "Note: core OK but not index-complete ({}): {}".format(
                len(core_ok_not_index),
                ", ".join(r["run_id"] for r in core_ok_not_index),
            )
        )
        print(
            "  (Usually training did not finish: no metadata.end / no test metrics; "
            "files may still be present.)"
        )

    if not_index_complete:
        print("\n--- Not index-complete ({}) ---".format(len(not_index_complete)))
        print("(Missing files, unfinished training in description, etc. --fix does not repair these.)")
        for r in not_index_complete:
            print(
                "  {}  [{}]  {}".format(
                    r["run_id"],
                    r.get("category"),
                    r.get("index_reason") or "",
                )
            )

    if core_bad:
        print("\n--- Core trio not OK ({}) ---".format(len(core_bad)))
        print("(Use --core-detail for full dicts. --fix only strips UTF-8 BOM on description.json.)")
        for r in core_bad:
            c = r.get("core") or {}
            parts: List[str] = []
            d = c.get("description") or {}
            h = c.get("hypnograms") or {}
            m = c.get("best_model") or {}
            if not d.get("readable"):
                parts.append("description:{}".format(d.get("error", "?")))
            if not h.get("readable"):
                parts.append("hypnograms:{}".format(h.get("error", "?")))
            if not m.get("valid_tar"):
                parts.append("best_model:{}".format(m.get("error", "?")))
            xci = c.get("cross_check_issues") or []
            if xci:
                parts.append("; ".join(xci))
            print("  {}  {}".format(r["run_id"], " | ".join(parts) if parts else "see core"))

    if args.fix:
        print("\n--- Fix (--fix) ---")
        print(
            "Only action: strip UTF-8 BOM from description.json if present. "
            "Does not restore missing/corrupt hypnograms or best_model.gz."
        )
        if not args.apply:
            print("Dry-run (use --apply to write)")
        any_fix = False
        for r in rows:
            fr = r.get("fix_result")
            if not fr:
                continue
            if fr.get("actions") or fr.get("would_do"):
                any_fix = True
            if fr.get("actions"):
                print("{} applied: {}".format(r["run_id"], fr["actions"]))
            elif fr.get("would_do"):
                print("{} would: {}".format(r["run_id"], fr["would_do"]))
        if not any_fix:
            print("No UTF-8 BOM on any description.json to remove.")
            if core_bad or not_index_complete:
                print("Runs still need attention: see sections above (remote restore, re-train, or manual fix).")

    if args.core_detail and core_bad:
        print("\n--- Core trio / cross-check issues ---")
        for r in core_bad:
            c = r.get("core") or {}
            print("\n{}".format(r["run_id"]))
            print("  description:", c.get("description"))
            print("  hypnograms:", c.get("hypnograms"))
            print("  best_model:", c.get("best_model"))
            if c.get("checkpoint_source"):
                print("  training checkpoint available:", c["checkpoint_source"])
            xci = c.get("cross_check_issues") or []
            if xci:
                print("  cross_check:", "; ".join(xci))

    if damaged:
        print("\n--- Suspected sync damage (review these first) ---")
        for r in damaged:
            print("\n{}".format(r["run_id"]))
            print("  path: {}".format(r["run_path"]))
            if r["missing_files"]:
                print("  missing:", ", ".join(r["missing_files"]))
            if r["suspicious_small_files"]:
                for s in r["suspicious_small_files"]:
                    print(
                        "  small file: {} ({} bytes; expected >= {})".format(
                            s["path"], s["size_bytes"], s["min_expected"]
                        )
                    )
            for n in r["notes"]:
                print("  note:", n)

    if suspicious:
        print("\n--- Suspicious small files (description does not claim finished) ---")
        for r in suspicious:
            print("{} -> {}".format(r["run_id"], r["suspicious_small_files"]))

    if args.print_rclone_recovery and damaged:
        remote_base = args.remote_prefix.rstrip("/")
        print("\n--- Example recovery (remote -> local; overwrites local run folder) ---")
        print("# Set --remote-prefix to the rclone path that matches this runs_root:")
        print("#   runs_root = {}".format(runs_root))
        print("# Verify remote has full runs: rclone ls {}/<run_id>/".format(remote_base))
        print("# Then:")
        for r in damaged:
            local_run = os.path.join(runs_root, r["run_id"])
            print('rclone copy "{}/{}/" "{}/" -vP'.format(remote_base, r["run_id"], local_run))
        print(
            "\n# Add --dry-run first to preview. If remote is newer/larger, this restores it."
        )

    if args.json_out:
        report = {
            "runs_root": runs_root,
            "summary": {
                "total": len(rows),
                "ok": len(ok),
                "core_ok": len(core_ok_runs),
                "suspected_sync_damage": len(damaged),
                "suspicious_small_files": len(suspicious),
                "index_ok_but_core_failed": len(index_ok_core_bad),
            },
            "runs": rows,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("\nWrote:", args.json_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
