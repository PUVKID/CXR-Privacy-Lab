"""Fixed-model original/blur/mosaic evaluation, using only the two existing datasets.

No training data or test labels are modified. This is the user's extension to the
paper, not a claim that the paper reported these privacy-transform experiments.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import traceback
import uuid

import numpy as np

from cxr7_privacy_contract import (DATASETS, TASKS, attack_reduction, condition_plan,
                                   open_report, record_result, save_report,
                                   report_is_complete, task_is_complete,
                                   validate_baseline, write_csv)


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def cached_original(path: Path, metadata: dict, loader) -> np.ndarray:
    """Only four original embedding arrays are cached; transformed images never are."""
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if path.exists() and sidecar.exists():
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        if saved.get("binding") != metadata:
            raise ValueError(f"Original embedding cache binding mismatch: {path}")
        if saved.get("sha256") != _sha(path):
            raise ValueError(f"Original embedding cache checksum mismatch: {path}")
        result = np.load(path, allow_pickle=False)
    else:
        result = np.asarray(loader(), dtype=np.float32)
        if (result.ndim != 2 or len(result) != metadata["n_images"] or
                not np.isfinite(result).all()):
            raise ValueError("Original embedding array is malformed or nonfinite")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temp.open("wb") as file:
                np.save(file, result, allow_pickle=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()
        save_report(sidecar, {"binding": metadata, "sha256": _sha(path)})
    if (result.ndim != 2 or len(result) != metadata["n_images"] or
            result.dtype != np.float32 or not np.isfinite(result).all()):
        raise ValueError("Cached original embedding array is malformed or nonfinite")
    return result


def evaluate_group(report: dict, dataset: str, task: str, baseline_metrics: dict,
                   extract_original, extract_query, measure, on_save,
                   on_condition=None, *, conditions=None) -> None:
    """Resume per task and condition while holding one immutable gallery in memory.

    Callbacks encapsulate device work; all resume decisions and result recording
    are performed here and tested with actual metrics on synthetic embeddings.
    """
    original = None
    for condition in condition_plan() if conditions is None else conditions:
        if task_is_complete(report, dataset, condition["id"], task):
            continue
        if on_condition is not None:
            on_condition(condition)
        if condition["operator_mode"] == "original":
            outcome = copy.deepcopy(baseline_metrics)
        else:
            if original is None:
                original = extract_original()
            query = extract_query(condition["operator_mode"], condition["operator_strength"])
            outcome = measure(query, original)
            del query
        if task == "retrieval":
            outcome.update(attack_reduction(baseline_metrics["precision_at_1"],
                                            outcome["precision_at_1"]))
        record_result(report, dataset, condition, task, outcome)
        on_save()


def ordered_groups():
    """Finish the entire original/blur stage before starting any mosaic work."""
    for stage in ("original", "blur", "mosaic"):
        conditions = [condition for condition in condition_plan() if condition["mode"] == stage]
        for task in TASKS:
            for dataset in DATASETS:
                yield stage, task, dataset, conditions


def refresh_examples(report: dict, export, on_save) -> None:
    """Revalidate examples on successful resume; the exporter preserves originals."""
    examples = report.setdefault("examples", {})
    for condition in condition_plan():
        examples[condition["id"]] = export(condition["mode"], condition["strength"])
    on_save()


def arguments():
    parser = argparse.ArgumentParser(description="CXR7 original / 10 blur / 10 mosaic evaluation")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retrieval-batch", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 0 or args.retrieval_batch < 1:
        parser.error("workers >= 0 and retrieval-batch >= 1 are required")
    return args


def run() -> None:
    args = arguments()
    seven = Path(__file__).resolve().parent.parent
    args.project = args.project.resolve()
    args.output = args.output.resolve()
    from cxr7_contract import (assert_training_scope, audit_official_manifests,
                               audit_image_presence, EXPECTED_SHA256)
    assert_training_scope(str(seven), str(args.output), str(args.project))
    artifacts_file = args.output / "artifacts.json"
    baseline_file = args.output / "evaluation_results.json"
    if not artifacts_file.is_file() or not baseline_file.is_file():
        raise FileNotFoundError("The newly trained baseline must finish its original evaluation first")
    artifacts = json.loads(artifacts_file.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    checkpoints = {
        "verification": args.output / "verification/selected_weights.pth",
        "retrieval": args.output / "retrieval_phase2/selected_weights.pth",
    }
    hashes = {task: _sha(path) for task, path in checkpoints.items()}
    from cxr7_reports import evaluator_fingerprint
    validate_baseline(artifacts, baseline, hashes,
                      evaluator_code_sha256=evaluator_fingerprint(Path(__file__).resolve().parent))
    official = seven / "official_source"
    audit_official_manifests(official)
    audit_image_presence(args.project, official)

    source_names = ["cxr7_privacy.py", "cxr7_privacy_contract.py", "cxr7_examples.py",
                    "cxr_reid/evaluation.py", "cxr_reid/inference.py",
                    "cxr_reid/image_ops.py", "cxr_reid/models.py", "cxr_reid/data.py"]
    binding_payload = {
        "checkpoints_sha256": hashes,
        "baseline_results_sha256": _sha(baseline_file),
        "official_manifests": EXPECTED_SHA256,
        "sources": {name: _sha(seven / "src" / name) for name in source_names},
        "environment": {name: version(name) for name in
                        ("torch", "torchvision", "numpy", "Pillow", "scikit-learn")},
        "plan": condition_plan(),
        "execution_order": "original_all_then_blur_all_then_mosaic_all; within_stage_task_dataset_strength",
        "verification_batch_size": 32,
        "retrieval_batch_size": args.retrieval_batch,
        "query_transform": "source_image_before_model_resize",
        "gallery_transform": "original",
        "self_match_policy": "exclude_diagonal_same_image_for_patient_retrieval",
    }
    report_path = args.output / "privacy_results.json"
    csv_path = args.output / "privacy_results.csv"
    report = open_report(report_path, binding_payload)

    from cxr7_examples import save_condition_example

    def saved():
        save_report(report_path, report)
        write_csv(csv_path, report)

    def export(mode, strength):
        return save_condition_example(args.project, official, args.output, mode, strength)

    if report_is_complete(report):
        print("[PRIVACY] All conditions already completed with the same bindings", flush=True)
        report["completed"] = True
        refresh_examples(report, export, saved)
        return

    import torch
    from cxr7_evaluate import ensure_gpu_idle, selected_model
    from cxr_reid.data import read_pair_manifest, read_retrieval_manifest
    from cxr_reid.evaluation import calculate_verification_metrics, calculate_retrieval_metrics
    from cxr_reid.inference import extract_embeddings, predict_pair_probabilities
    from cxr_reid.models import VerificationNetwork, create_retrieval_network
    from training.state import progress
    device = ensure_gpu_idle()

    def stop_check():
        if (args.output / "STOP_REQUESTED").exists():
            raise KeyboardInterrupt("7차 privacy evaluation stopped by request; completed conditions are preserved")

    def emit(task, dataset, condition, status, done=0, total=0):
        stop_check()
        completed = sum(task_is_complete(report, name, item["id"], kind)
                        for name in DATASETS for item in condition_plan() for kind in TASKS)
        progress(args.output, phase="privacy", status=status, epoch=0, epochs=0,
                 step=completed, loss=None, task=task, dataset=dataset,
                 condition=condition, completed_task_conditions=completed,
                 total_task_conditions=len(DATASETS) * len(TASKS) * len(condition_plan()),
                 batch=done, batches=total,
                 progress_unit="queries" if status == "ranking_queries" else "batches")

    image_roots = {"NIH": args.project / "data/NIH_ChestXray14/images",
                   "COVID19": args.project / "data/COVID19/images"}
    pair_manifests = {
        "NIH": (official / "PatientVerification/image_pairs/pairs_testing.txt", 100000),
        "COVID19": (official / "PatientVerification/image_pairs/pairs_testing_covid19.txt", 1548),
    }
    retrieval_manifests = {
        "NIH": [official / "PatientRetrieval/csv_files/test.csv",
                official / "PatientRetrieval/csv_files/test_single_files.csv"],
        "COVID19": [official / "PatientRetrieval/csv_files/covid_data.csv"],
    }
    saved()
    models = {"verification": ("verification", VerificationNetwork),
              "retrieval": ("retrieval_phase2", create_retrieval_network)}
    model, loaded_task = None, None
    for stage, task, dataset, conditions in ordered_groups():
        stop_check()
        if all(task_is_complete(report, dataset, condition["id"], task) for condition in conditions):
            continue
        if loaded_task != task:
            if model is not None:
                del model
                torch.cuda.empty_cache()
            phase, factory = models[task]
            model, observed_sha = selected_model(args.output, phase, factory, device)
            if observed_sha != hashes[task]:
                raise ValueError("Baseline checkpoint changed while preparing the privacy sweep")
            loaded_task = task
        current_condition = stage
        if task == "verification":
            manifest, expected = pair_manifests[dataset]
            pairs = read_pair_manifest(manifest, expected_count=expected)
            names = sorted({name for left, right, _ in pairs for name in (left, right)})
            index = {name: i for i, name in enumerate(names)}

            def measure(query, gallery):
                truth, scores = predict_pair_probabilities(model, pairs, index, query, gallery)
                return calculate_verification_metrics(truth, scores)
        else:
            rows = read_retrieval_manifest(retrieval_manifests[dataset])
            names = [row[0] for row in rows]
            ids = [row[1] for row in rows]

            def measure(query, gallery):
                return calculate_retrieval_metrics(
                    query, gallery, ids, ids, same_source=True, gallery_keys=names,
                    progress_callback=lambda done, total: emit(
                        task, dataset, current_condition, "ranking_queries", done, total))

        def extract(mode, strength):
            emit(task, dataset, f"{mode}_{strength:03d}", "extracting")
            return extract_embeddings(
                model, image_roots[dataset], names, task=task, mode=mode,
                strength=strength, device=device,
                batch_size=32 if task == "verification" else args.retrieval_batch,
                num_workers=args.workers,
                progress_callback=lambda done, total: emit(
                    task, dataset, f"{mode}_{strength:03d}", "extracting", done, total))

        def original():
            metadata = {"binding": report["binding"], "checkpoint": hashes[task],
                        "names": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
                        "n_images": len(names), "task": task, "dataset": dataset}
            cache_file = args.output / "privacy_cache" / f"{dataset}_{task}_original.npy"
            return cached_original(cache_file, metadata, lambda: extract("original", 0))

        def on_condition(condition):
            nonlocal current_condition
            current_condition = condition["id"]
            emit(task, dataset, current_condition, "evaluating_condition")
            examples = report.setdefault("examples", {})
            if condition["id"] not in examples:
                examples[condition["id"]] = save_condition_example(
                    args.project, official, args.output, condition["mode"], condition["strength"])
                save_report(report_path, report)

        evaluate_group(report, dataset, task, baseline[dataset][task], original,
                       extract, measure, saved, on_condition, conditions=conditions)
    if model is not None:
        del model
        torch.cuda.empty_cache()
    if not report["completed"]:
        raise RuntimeError("The privacy sweep ended with pending conditions; no completion was claimed")
    refresh_examples(report, export, saved)
    emit("both", "both", "all", "completed")
    print(f"[PRIVACY RESULTS] {report_path}", flush=True)
    print(f"[PRIVACY CSV] {csv_path}", flush=True)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt as error:
        print(f"[STOPPED] {error}", flush=True)
        raise SystemExit(130)
    except Exception as error:
        print(f"PRIVACY EVALUATION FAILED: {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
