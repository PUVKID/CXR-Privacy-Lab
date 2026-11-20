"""Auditable blur/mosaic sweep bookkeeping without torch or image side effects."""

from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import uuid


LEVELS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
DATASETS = ("NIH", "COVID19")
TASKS = ("verification", "retrieval")


def condition_plan() -> list[dict]:
    """Preserve the user's 10 levels; 0 means identity for either operator."""
    output = [{"id": "original_000", "mode": "original", "strength": 0,
               "operator_mode": "original", "operator_strength": 0,
               "identity_alias": False}]
    for mode in ("blur", "mosaic"):
        for strength in LEVELS:
            identity = strength == 0 or (mode == "mosaic" and strength == 1)
            output.append({"id": f"{mode}_{strength:03d}", "mode": mode,
                           "strength": strength,
                           "operator_mode": "original" if identity else mode,
                           "operator_strength": 0 if identity else strength,
                           "identity_alias": identity})
    return output


def _validate_metrics(task: str, value: dict) -> None:
    keys = ("auc", "accuracy") if task == "verification" else (
        "precision_at_1", "map_at_r", "r_precision")
    if task not in TASKS or not isinstance(value, dict):
        raise ValueError("Unsupported task or metric object")
    for key in keys:
        number = value.get(key)
        if not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"Missing or nonfinite measured {task} metric: {key}")
    # Reject even non-core NaN fields instead of exporting non-standard JSON.
    json.dumps(value, allow_nan=False)


def validate_baseline(artifacts: dict, baseline: dict, hashes: dict, *,
                      evaluator_code_sha256: dict | None = None) -> None:
    """Never mix old/new weights or start the sweep on an incomplete baseline."""
    if baseline.get("completed") is not True:
        raise ValueError("Baseline evaluation is not complete")
    if artifacts.get("published_cxr_initialization") is not False:
        raise ValueError("The sweep requires the newly trained baseline, not published CXR weights")
    if evaluator_code_sha256 is not None and baseline.get("evaluator_code_sha256") != evaluator_code_sha256:
        raise ValueError("Baseline evaluator fingerprint differs; do not mix changed evaluation code")
    for task in TASKS:
        checkpoint = hashes.get(task)
        if not checkpoint or artifacts.get(task + "_sha256") != checkpoint:
            raise ValueError(f"Artifact checkpoint does not match {task}")
        for dataset in DATASETS:
            result = baseline.get(dataset, {})
            if result.get(task + "_sha256") != checkpoint:
                raise ValueError(f"Baseline checkpoint does not match {dataset} {task}")
            _validate_metrics(task, result.get(task))


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def open_report(path: Path, binding_payload: dict) -> dict:
    """Bind the whole report to its checkpoints, code, manifests and settings."""
    encoded = json.dumps(binding_payload, sort_keys=True, ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    binding = hashlib.sha256(encoded).hexdigest()
    path = Path(path)
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != 1 or value.get("binding") != binding:
            raise ValueError("Privacy resume binding mismatch; existing results were not overwritten")
        return value
    return {
        "schema": 1, "binding": binding, "binding_payload": binding_payload,
        "created_at": _stamp(), "updated_at": _stamp(), "completed": False,
        "conditions": condition_plan(), "datasets_used": list(DATASETS),
        "protocol": "fixed_baseline_transformed_query_original_gallery",
        "same_image_excluded_for_patient_retrieval": True,
        "clinical_utility_evaluated": False,
        "limitations": [
            "Blur/mosaic are an extension, not experiments reported by the original paper.",
            "No test-set retraining or tuning; the baseline checkpoint stays fixed.",
            "Blur radius and mosaic tile size are pixels in each source image before model resize.",
            "PrivacyScore@1 is 100*(1-P@1), an empirical attack-failure complement, not a formal privacy guarantee.",
            "Patient matching is not image reconstruction; low performance does not prove irreversibility.",
            "AUC below .5 may be exploitable by score inversion; raw AUC is not a privacy certificate.",
        ],
        "results": {dataset: {} for dataset in DATASETS},
    }


def task_is_complete(report: dict, dataset: str, condition_id: str, task: str) -> bool:
    row = report.get("results", {}).get(dataset, {}).get(condition_id, {})
    if row.get(task + "_status") != "completed":
        return False
    try:
        _validate_metrics(task, row.get(task))
    except (ValueError, TypeError):
        return False
    return True


def record_result(report: dict, dataset: str, condition: dict, task: str,
                  metrics: dict) -> None:
    if dataset not in DATASETS or condition not in condition_plan():
        raise ValueError("Unexpected dataset or condition")
    _validate_metrics(task, metrics)
    row = report["results"][dataset].setdefault(condition["id"], copy.deepcopy(condition))
    row[task] = copy.deepcopy(metrics)
    row[task + "_status"] = "completed"
    row[task + "_updated_at"] = _stamp()
    row[task + "_measurement_source"] = (
        "baseline_reused_identity_transform"
        if condition["operator_mode"] == "original" else "measured_transformed_query")
    report["completed"] = report_is_complete(report)
    report["updated_at"] = _stamp()


def report_is_complete(report: dict) -> bool:
    """Do not trust a stale summary boolean after an interrupted/manual edit."""
    return all(task_is_complete(report, name, item["id"], kind)
               for name in DATASETS for item in condition_plan() for kind in TASKS)


def attack_reduction(original_rank1: float, transformed_rank1: float) -> dict:
    for value in (original_rank1, transformed_rank1):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Rank-1 rates must be finite numbers in [0,1]")
    return {
        "privacy_score_at_1": 100.0 * (1.0 - transformed_rank1),
        "privacy_score_definition": "100 * (1 - measured patient retrieval P@1)",
        "rank1_reduction_percentage_points": 100.0 * (original_rank1 - transformed_rank1),
        "relative_rank1_reduction": ((original_rank1 - transformed_rank1) / original_rank1
                                     if original_rank1 > 0 else None),
        "formal_privacy_guarantee": False,
    }


def save_report(path: Path, report: dict) -> None:
    _atomic_text(path, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


def write_csv(path: Path, report: dict) -> None:
    columns = ["dataset", "condition", "mode", "strength", "identity_alias",
               "verification_status", "verification_auc", "verification_accuracy",
               "verification_n_pairs", "retrieval_status", "retrieval_precision_at_1",
               "retrieval_map_at_r", "retrieval_r_precision", "retrieval_eligible_queries",
               "privacy_score_at_1", "rank1_reduction_percentage_points",
               "relative_rank1_reduction", "formal_privacy_guarantee"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for dataset in DATASETS:
        for item in condition_plan():
            row = report["results"][dataset].get(item["id"])
            if not row:
                continue
            verification, retrieval = row.get("verification", {}), row.get("retrieval", {})
            output = {"dataset": dataset, "condition": item["id"], "mode": item["mode"],
                      "strength": item["strength"], "identity_alias": item["identity_alias"],
                      "verification_status": row.get("verification_status", "pending"),
                      "retrieval_status": row.get("retrieval_status", "pending")}
            for key in ("auc", "accuracy", "n_pairs"):
                output["verification_" + key] = verification.get(key)
            for key in ("precision_at_1", "map_at_r", "r_precision", "eligible_queries"):
                output["retrieval_" + key] = retrieval.get(key)
            for key in ("privacy_score_at_1", "rank1_reduction_percentage_points",
                        "relative_rank1_reduction", "formal_privacy_guarantee"):
                output[key] = retrieval.get(key)
            writer.writerow(output)
    _atomic_text(path, buffer.getvalue(), "utf-8-sig")
