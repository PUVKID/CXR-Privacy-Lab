"""Literal primary-paper reference values; do not treat them as guaranteed targets."""

import hashlib
import math
from pathlib import Path

PAPER_NIH_RESULTS = {
    "verification_auc": 0.9940,
    "verification_accuracy": 0.9555,
    "retrieval_precision_at_1": 0.9963,
    "retrieval_map_at_r": 0.9748,
    "retrieval_r_precision": 0.9763,
}


def paper_comparison(measured: dict[str, float]) -> dict[str, dict]:
    """Only compare numbers actually measured; no fabricated success indicators."""
    result = {}
    for key, paper in PAPER_NIH_RESULTS.items():
        if key in measured and measured[key] is not None:
            actual = float(measured[key])
            result[key] = {"paper": paper, "measured": actual,
                           "difference": actual - paper,
                           "identical_reported_value": round(actual, 4) == paper}
    return result


def verification_is_complete(existing: dict, checkpoint_sha256: str, bootstrap: int) -> bool:
    """Only skip a completed metric/CI for the *same* checkpoint hash."""
    if (existing.get("verification_sha256") != checkpoint_sha256 or
            "verification" not in existing):
        return False
    if bootstrap == 0:
        return True
    interval = existing["verification"].get("confidence_interval", {})
    return interval.get("bootstrap_repeats", 0) >= bootstrap


def retrieval_is_complete(existing: dict, checkpoint_sha256: str) -> bool:
    """A stored hash alone is not evidence that retrieval evaluation finished."""
    if existing.get("retrieval_sha256") != checkpoint_sha256:
        return False
    values = existing.get("retrieval")
    if not isinstance(values, dict):
        return False
    for key in ("precision_at_1", "map_at_r", "r_precision"):
        value = values.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0.0 <= value <= 1.0):
            return False
    return True


def evaluator_fingerprint(source_root: Path) -> dict[str, str]:
    """Bind baseline and perturbation comparisons to identical prediction/metric code."""
    files = ("cxr7_evaluate.py", "cxr7_reports.py", "cxr_reid/models.py",
             "cxr_reid/evaluation.py", "cxr_reid/inference.py",
             "cxr_reid/image_ops.py", "cxr_reid/data.py")
    return {name: hashlib.sha256((Path(source_root) / name).read_bytes()).hexdigest()
            for name in files}


def validate_evaluation_resume(saved: dict, code_hashes: dict, model_hashes: dict) -> None:
    """Fail closed instead of mixing evaluations from a different implementation."""
    if saved.get("evaluator_code_sha256") != code_hashes:
        raise ValueError("Evaluation code fingerprint changed; existing results preserved")
    for dataset in ("NIH", "COVID19"):
        values = saved.get(dataset, {})
        for task in ("verification", "retrieval"):
            if task in values and values.get(task + "_sha256") != model_hashes.get(task):
                raise ValueError(f"Evaluation checkpoint changed: {dataset} {task}")
