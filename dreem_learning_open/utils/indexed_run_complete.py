"""
Single definition of "complete enough to index / count as a finished fold" for
``completed_folds.jsonl`` and parallel runners.

Does *not* require ``metadata.end`` or ``training/best_net``. Requires on-disk
artifacts and populated description fields so a fold is not skipped as done
when only training partially finished or metadata was never updated.

If several artifacts are absent, the failure reason lists all of them
(``missing_files:hypnograms.json,best_model.gz``), not only the first.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple


def check_indexed_run_complete(run_dir: str, description: dict) -> Tuple[bool, Optional[str]]:
    if not isinstance(description, dict):
        return False, "invalid_description_type"

    required_files = [
        os.path.join(run_dir, "description.json"),
        os.path.join(run_dir, "hypnograms.json"),
        os.path.join(run_dir, "best_model.gz"),
    ]
    missing_rels = []
    for file_path in required_files:
        if not os.path.isfile(file_path):
            missing_rels.append(
                os.path.relpath(file_path, run_dir).replace("\\", "/")
            )
    if missing_rels:
        return False, "missing_files:{}".format(",".join(missing_rels))

    perf = description.get("performance_on_test_set")
    if perf is None:
        return False, "performance_on_test_set_null"
    if not isinstance(perf, dict) or len(perf) == 0:
        return False, "performance_on_test_set_empty"

    pr = description.get("performance_per_records")
    if pr is None:
        return False, "performance_per_records_null"
    if not isinstance(pr, dict) or len(pr) == 0:
        return False, "performance_per_records_empty"

    rs = description.get("records_split")
    if rs is None:
        return False, "records_split_null"
    if not isinstance(rs, dict):
        return False, "records_split_invalid"

    memar_no_val = bool(description.get("memar_et_al_subject_cv"))

    for key in ("train_records", "validation_records", "test_records"):
        v = rs.get(key)
        if v is None:
            return False, "records_split_null:{}".format(key)
        if not isinstance(v, list):
            return False, "records_split_invalid_list:{}".format(key)
        if len(v) == 0:
            if key == "validation_records" and memar_no_val:
                continue
            return False, "records_split_empty:{}".format(key)

    test_records = rs.get("test_records", [])
    if not isinstance(test_records, list) or len(test_records) != 1:
        return False, "invalid_test_records"

    return True, None
