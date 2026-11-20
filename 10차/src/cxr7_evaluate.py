"""Evaluate a new 7th-run model on the author's NIH/COVID test lists only.

This is a documented PyTorch 2.x port; measurements are not published weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from cxr7_contract import assert_training_scope, audit_official_manifests, audit_image_presence
from cxr7_reports import (paper_comparison, verification_is_complete, retrieval_is_complete,
                          evaluator_fingerprint, validate_evaluation_resume)
from cxr_reid.data import read_retrieval_manifest, read_pair_manifest
from cxr_reid.evaluation import (calculate_verification_metrics, calculate_retrieval_metrics,
                                 bootstrap_auc_ci)
from cxr_reid.inference import extract_embeddings, predict_pair_probabilities
from cxr_reid.models import VerificationNetwork, create_retrieval_network
from training.state import atomic_json, digest, progress


def arguments():
    parser = argparse.ArgumentParser(description="7차 원논문 NIH + COVID 외부 테스트")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", choices=["both", "verification", "retrieval"], default="both")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retrieval-batch", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.workers < 0 or args.retrieval_batch < 1 or args.bootstrap < 0:
        parser.error("workers>=0, retrieval-batch>=1, bootstrap>=0 required")
    return args


def ensure_gpu_idle():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA / RTX GPU is required to evaluate without CPU fallback")
    free, total = torch.cuda.mem_get_info()
    if free < 9 * 1024**3:
        raise RuntimeError(f"GPU is busy. Only {free / 2**30:.1f} GiB free; "
                           "existing training is left untouched.")
    return "cuda"


def selected_model(output: Path, phase: str, factory, device: str):
    file = output / phase / "selected_weights.pth"
    if not file.is_file():
        raise FileNotFoundError(f"Completed model weights missing: {file}")
    model = factory()
    model.load_state_dict(torch.load(file, weights_only=True, map_location="cpu"), strict=True)
    return model.to(device).eval(), digest(file)


def print_result(name: str, value: dict):
    numbers = ", ".join(f"{key}={number:.5f}" for key, number in value.items()
                        if key in ("auc", "accuracy", "precision_at_1", "map_at_r", "r_precision"))
    print(f"[EVALUATION] {name}: {numbers}", flush=True)


def run():
    args = arguments()
    args.project = args.project.resolve()
    args.output = args.output.resolve()
    seven = Path(__file__).resolve().parent.parent
    assert_training_scope(str(seven), str(args.output), str(args.project))
    if not (args.output / "artifacts.json").is_file():
        raise FileNotFoundError("7차 학습이 아직 모두 끝나지 않았습니다: artifacts.json 없음")
    artifacts = json.loads((args.output / "artifacts.json").read_text(encoding="utf-8"))
    if artifacts.get("published_cxr_initialization") is not False:
        raise ValueError("Evaluation must use the newly trained CXR7 models")
    model_hashes = {}
    for task, phase in (("verification", "verification"), ("retrieval", "retrieval_phase2")):
        if args.tasks not in ("both", task):
            continue
        model_hashes[task] = digest(args.output / phase / "selected_weights.pth")
        if model_hashes[task] != artifacts.get(task + "_sha256"):
            raise ValueError(f"Completed artifact checkpoint mismatch: {task}")
    official = seven / "official_source"
    audit_official_manifests(official)
    audit_image_presence(args.project, official)
    device = ensure_gpu_idle()

    nih_root = args.project / "data/NIH_ChestXray14/images"
    covid_root = args.project / "data/COVID19/images"
    vr = official / "PatientVerification/image_pairs"
    rr = official / "PatientRetrieval/csv_files"
    results_path = args.output / "evaluation_results.json"
    code_hashes = evaluator_fingerprint(Path(__file__).resolve().parent)
    results = {"paper_source": "Packhäuser et al., Scientific Reports 2022, DOI:10.1038/s41598-022-19045-3",
               "author_commit": "880168f564afa07936a8d8fb3b868b140ab937d9",
               "datasets_used": ["NIH_ChestXray14", "COVID19 Image Data Collection"],
               "reference_model_is_not_used": True,
               "method": "documented_windows_pytorch2_port",
               "evaluator_code_sha256": code_hashes,
               "NIH": {}, "COVID19": {}, "completed": False}
    if results_path.exists():
        old = json.loads(results_path.read_text(encoding="utf-8"))
        validate_evaluation_resume(old, code_hashes, {
            "verification": artifacts.get("verification_sha256"),
            "retrieval": artifacts.get("retrieval_sha256")})
        for key in ("NIH", "COVID19"):
            results[key].update(old.get(key, {}))

    def heartbeat(status):
        def update(done, total):
            if (args.output / "STOP_REQUESTED").exists():
                raise KeyboardInterrupt("STOP_REQUESTED")
            progress(args.output, phase="evaluation", status=status,
                     epoch=0, epochs=0, step=done, steps=total, loss=None)
        return update

    if args.tasks in ("both", "verification"):
        model, sha = selected_model(args.output, "verification", VerificationNetwork, device)
        for dataset, root, manifest, expected in (
            ("NIH", nih_root, vr / "pairs_testing.txt", 100000),
            ("COVID19", covid_root, vr / "pairs_testing_covid19.txt", 1548),
        ):
            if verification_is_complete(results[dataset], sha, args.bootstrap):
                print(f"[SKIP] {dataset} verification evaluation already saved with same checkpoint", flush=True)
                continue
            progress(args.output, phase="evaluation", status="verification_" + dataset,
                     epoch=0, epochs=0, step=0, loss=None)
            pairs = read_pair_manifest(manifest, expected_count=expected)
            filenames = sorted({name for left, right, _ in pairs for name in (left, right)})
            embeddings = extract_embeddings(model, root, filenames, task="verification",
                                            mode="original", strength=0, device=device,
                                            batch_size=32, num_workers=args.workers,
                                            progress_callback=heartbeat("verification_" + dataset))
            truth, probabilities = predict_pair_probabilities(
                model, pairs, {name: i for i, name in enumerate(filenames)}, embeddings, embeddings)
            outcome = calculate_verification_metrics(truth, probabilities)
            results[dataset]["verification"] = outcome
            results[dataset]["verification_sha256"] = sha
            print_result(dataset + " verification", outcome)
            atomic_json(results_path, results)
            if args.bootstrap:
                progress(args.output, phase="evaluation", status="bootstrap_" + dataset,
                         epoch=0, epochs=0, step=0, loss=None)
                outcome["confidence_interval"] = bootstrap_auc_ci(
                    truth, probabilities, repeats=args.bootstrap, seed=3,
                    progress_callback=heartbeat("bootstrap_" + dataset))
                atomic_json(results_path, results)
        del model
        torch.cuda.empty_cache()

    if args.tasks in ("both", "retrieval"):
        model, sha = selected_model(args.output, "retrieval_phase2", create_retrieval_network, device)
        for dataset, root, csvs in (
            ("COVID19", covid_root, [rr / "covid_data.csv"]),
            ("NIH", nih_root, [rr / "test.csv", rr / "test_single_files.csv"]),
        ):
            if retrieval_is_complete(results[dataset], sha):
                print(f"[SKIP] {dataset} retrieval evaluation already saved with same checkpoint", flush=True)
                continue
            progress(args.output, phase="evaluation", status="retrieval_" + dataset,
                     epoch=0, epochs=0, step=0, loss=None)
            rows = read_retrieval_manifest(csvs)
            filenames = [name for name, _ in rows]
            ids = [label for _, label in rows]
            vectors = extract_embeddings(model, root, filenames, task="retrieval",
                                         mode="original", strength=0, device=device,
                                         batch_size=args.retrieval_batch,
                                         num_workers=args.workers,
                                         progress_callback=heartbeat("retrieval_" + dataset))
            outcome = calculate_retrieval_metrics(
                vectors, vectors, ids, ids, same_source=True, gallery_keys=filenames,
                progress_callback=heartbeat("retrieval_metrics_" + dataset))
            results[dataset]["retrieval"] = outcome
            results[dataset]["retrieval_sha256"] = sha
            print_result(dataset + " retrieval", outcome)
            atomic_json(results_path, results)
        del model
        torch.cuda.empty_cache()

    observed = {}
    if "verification" in results["NIH"]:
        observed["verification_auc"] = results["NIH"]["verification"]["auc"]
        observed["verification_accuracy"] = results["NIH"]["verification"]["accuracy"]
    if "retrieval" in results["NIH"]:
        metrics = results["NIH"]["retrieval"]
        observed.update(retrieval_precision_at_1=metrics["precision_at_1"],
                        retrieval_map_at_r=metrics["map_at_r"],
                        retrieval_r_precision=metrics["r_precision"])
    results["paper_comparison"] = paper_comparison(observed)
    results["completed"] = all("verification" in results[key] and "retrieval" in results[key]
                               for key in ("NIH", "COVID19"))
    atomic_json(results_path, results)
    progress(args.output, phase="evaluation", status="completed", epoch=0,
             epochs=0, step=0, loss=None, completed=results["completed"])
    print(f"[RESULTS] {results_path}", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print(f"EVALUATION FAILED: {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
