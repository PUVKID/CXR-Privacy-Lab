"""CXR7 original-protocol contract; intentionally requires no GPU packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import ntpath
import os
from pathlib import Path
import re
from datetime import datetime, timezone


SOURCE_COMMIT = "880168f564afa07936a8d8fb3b868b140ab937d9"
EXPECTED_SHA256 = {
    "PatientVerification/train_val_list.txt": "61fbe896321c1c1c8b75f3e4f3a08e4fef6486d95ef8a667c31d4d60dca6cb81",
    "PatientVerification/test_list.txt": "38ca5ef7f756092946f57c1a59faca882ed589a1ab1f72590b45dc06c6d5e1cc",
    "PatientVerification/image_pairs/pairs_training.txt": "58d2083346f545d2101e3b0cab6fc28c6671fa89f9de148a50fed241da4fa68d",
    "PatientVerification/image_pairs/pairs_validation.txt": "e22535919654ed1b17d600e91bfca2642b732d70779f64204d2ab0c3134b1049",
    "PatientVerification/image_pairs/pairs_testing.txt": "74da57c146cd8f4b183be8057201059bc990cdaeb15b9db712d359b4b4aa6310",
    "PatientVerification/image_pairs/pairs_testing_covid19.txt": "0daf9bd09e27a4c1ded08c89be6afd613364294dc3a289b691e0601b2dc81f55",
    "PatientRetrieval/csv_files/train.csv": "75d5e9d337187cd8dbc962303d42469f741ace47a46e9b0bc63ba748ea5708b7",
    "PatientRetrieval/csv_files/validation.csv": "16dfad3c6c3128b3c8f8388ffab4ea02121ba1a7b72d26bca72be361449a0541",
    "PatientRetrieval/csv_files/validation_single_files.csv": "5de86add746a61d580fe1dcaf1a8b0922bff319d204ca9c529852042cdf2ea3b",
    "PatientRetrieval/csv_files/test.csv": "a24e78afe49b71466999ad844a5d4b059c3ff20bcc0a43d46c33fe788d21718d",
    "PatientRetrieval/csv_files/test_single_files.csv": "606f539d7f1f1543cbe982f363fcc6743a468d795b4f4937cc5490969f84f452",
    "PatientRetrieval/csv_files/covid_data.csv": "7b4eeb1f8bbd8aa98946138069d886f634d0fef2f5deb2b2392fa5d7c25b1a7f",
}
EXPECTED_PAIR_COUNTS = {"pairs_training.txt": 792294,
                        "pairs_validation.txt": 50000, "pairs_testing.txt": 100000,
                        "pairs_testing_covid19.txt": 1548}
EXPECTED_CSV_COUNTS = {"train.csv": 61755, "validation.csv": 8637,
                       "validation_single_files.csv": 2178, "test.csv": 24223,
                       "test_single_files.csv": 1372, "covid_data.csv": 781}


def assert_seventh_workspace(path: str) -> None:
    """이10차 사본에서는 이전 차수·원본 밖으로 쓰지 못하게 제한합니다."""
    normalized = ntpath.normpath(str(path).replace("/", "\\"))
    if ntpath.splitdrive(normalized)[0].upper() != "E:":
        raise ValueError("10차 사본은 E: 프로젝트 안에서만 실행합니다.")
    tail = [part for part in normalized.split("\\") if part]
    if len(tail) < 2 or tail[-1] != "10차" or tail[-2] != "흉부 엑스레이 환자 재식별 실험":
        raise ValueError("독립학습 작업 위치는 프로젝트10차여야 합니다.")
    if ".." in str(path).replace("/", "\\").split("\\"):
        raise ValueError("Path traversal is not allowed")


def assert_training_scope(seven: str, output: str, project: str) -> None:
    """고정된10차 조건별 UUID 출력과 공통 원본 읽기만 허용합니다."""
    from cxr10_contract import assert_run_scope
    assert_run_scope(seven, output, project)


def prepare_new_run(base: Path, run_id: str) -> Path:
    """Exclusive creation: an old run is never reused without explicit resume."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", run_id):
        raise ValueError("Unsafe run identifier")
    output = Path(base) / run_id
    output.mkdir(parents=True, exist_ok=False)
    return output


def count_pair_records(path: Path) -> int:
    total = 0
    with open(path, encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            columns = raw.split()
            if not columns:
                continue
            if len(columns) != 3 or columns[2] not in ("0", "0.0", "1", "1.0"):
                raise ValueError(f"Malformed pair in {path.name} at line {lineno}")
            total += 1
    return total


def count_csv_records(path: Path) -> int:
    total = 0
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"Image Index", "Patient ID"}.issubset(reader.fieldnames):
            raise ValueError(f"CSV columns do not match the author: {path.name}")
        for lineno, row in enumerate(reader, 2):
            if not (row["Image Index"] or "").strip() or not (row["Patient ID"] or "").strip():
                raise ValueError(f"CSV has missing filename/patient at {path.name}:{lineno}")
            total += 1
    return total


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_official_manifests(official: Path, *, check_sha256: bool = True) -> dict:
    """Never project/drop images: author-supplied manifests are the experiment."""
    official = Path(official)
    for relative, expected in EXPECTED_SHA256.items():
        file = official / relative
        if not file.is_file():
            raise FileNotFoundError(f"Missing official manifest: {file}")
        if check_sha256 and sha256_file(file) != expected:
            raise ValueError(f"Official split hash mismatch: {relative}")
    pairs = {name: count_pair_records(official / "PatientVerification/image_pairs" / name)
             for name in EXPECTED_PAIR_COUNTS}
    csv_counts = {name: count_csv_records(official / "PatientRetrieval/csv_files" / name)
                  for name in EXPECTED_CSV_COUNTS}
    if pairs != EXPECTED_PAIR_COUNTS or csv_counts != EXPECTED_CSV_COUNTS:
        raise ValueError(f"Original manifest record counts differ: pairs={pairs} csv={csv_counts}")
    verification = official / "PatientVerification"
    names = (verification / "train_val_list.txt").read_text(encoding="utf-8").split()
    test_names = (verification / "test_list.txt").read_text(encoding="utf-8").split()
    if len(names) != 86524 or len(test_names) != 25596:
        raise ValueError("NIH original verification split size mismatch")
    if len(set(names + test_names)) != 112120:
        raise ValueError("NIH filenames overlap or are missing")
    return {"author_commit": SOURCE_COMMIT, "verification_pairs": pairs,
            "retrieval_csv": csv_counts, "nih_original_count": 112120,
            "verification_train_images": 75708,
            "verification_validation_images": 10816,
            "verification_test_images": 25596,
            "split_policy": "UNMODIFIED_AUTHOR_SPLITS"}


def audit_image_presence(project: Path, official: Path) -> dict:
    root = Path(project) / "data"
    nih = root / "NIH_ChestXray14/images"
    covid = root / "COVID19/images"
    if not nih.is_dir() or not covid.is_dir():
        raise FileNotFoundError(f"Two datasets must exist: {nih} and {covid}")
    vr = Path(official) / "PatientVerification"
    names = set((vr / "train_val_list.txt").read_text(encoding="utf-8").split())
    names.update((vr / "test_list.txt").read_text(encoding="utf-8").split())
    with open(official / "PatientRetrieval/csv_files/covid_data.csv", newline="", encoding="utf-8-sig") as f:
        covid_names = {row["Image Index"] for row in csv.DictReader(f)}
    missing_nih = [name for name in names if not (nih / name).is_file()]
    missing_covid = [name for name in covid_names if not (covid / name).is_file()]
    if missing_nih or missing_covid:
        raise FileNotFoundError(f"Missing NIH ({len(missing_nih)}) or COVID ({len(missing_covid)}) files: "
                                f"{missing_nih[:3] + missing_covid[:3]}")
    return {"nih_present": len(names), "covid_present": len(covid_names),
            "nih_dir": str(nih), "covid_dir": str(covid)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--seven", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifests-only", action="store_true")
    args = parser.parse_args()
    assert_seventh_workspace(str(args.seven))
    report = audit_official_manifests(args.official)
    if not args.manifests_only:
        report.update(audit_image_presence(args.project, args.official))
    report["checked_at"] = datetime.now(timezone.utc).isoformat()
    if args.output:
        target = args.output.resolve()
        if not target.is_relative_to(args.seven.resolve()):
            raise ValueError("Preflight output may only be written to 7차")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
