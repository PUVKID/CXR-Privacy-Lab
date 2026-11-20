"""원 저자가 공개한 환자별 시험 목록을 훼손하지 않고 읽는다."""

from __future__ import annotations

import csv
from pathlib import Path


def read_retrieval_manifest(csv_paths: list[Path]) -> list[tuple[str, str]]:
    """원본 CSV에서 이미지 이름과 환자 ID를 문자열로 읽는다."""
    output = []
    for csv_path in csv_paths:
        if not Path(csv_path).is_file():
            raise FileNotFoundError(f"원 논문의 분할 CSV가 없습니다: {csv_path}")
        with open(csv_path, newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                filename = (row.get("Image Index") or "").strip()
                patient_id = (row.get("Patient ID") or "").strip()
                if filename and patient_id:
                    output.append((filename, patient_id))
    if not output:
        raise ValueError("CSV에 유효한 검사 이미지와 환자 ID가 없습니다.")
    unique_names = {n for n, _ in output}
    if len(unique_names) != len(output):
        raise ValueError("평가 목록에 중복 이미지가 발견되었습니다.")
    return output


def read_pair_manifest(path: Path, expected_count: int | None = None) -> list[tuple[str, str, int]]:
    """저자 제공 검증용 원본 이미지 쌍 및 라벨(1=같은 환자)을 보존한다."""
    pairs = []
    with open(path, encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            cols = line.split()
            if not cols:
                continue
            if len(cols) != 3 or cols[2] not in ("0", "0.0", "1", "1.0"):
                raise ValueError(f"{path.name} {number}행의 환자 쌍 형식이 잘못되었습니다.")
            pairs.append((cols[0], cols[1], int(float(cols[2]))))
    if expected_count and len(pairs) != expected_count:
        raise ValueError(f"검증 쌍은 {expected_count:,}개여야 하나 현재 {len(pairs):,}개입니다.")
    return pairs


def validate_image_list(root: Path, names: list[str]) -> None:
    """입력 이미지의 범위를 고정하고 누락 파일을 초기에 찾는다."""
    missing = []
    for name in set(names):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"허용되지 않는 상대 경로입니다: {name}")
        if not (root / path).is_file():
            missing.append(name)
            if len(missing) >= 10:
                break
    if missing:
        raise FileNotFoundError(f"NIH 이미지가 없습니다(최대 10개): {missing}")


def validate_patient_separation(*manifests: list[tuple[str, str]]) -> None:
    """학습·검증·평가 데이터 간 환자 중복을 검사한다."""
    ids = [{patient for _, patient in subset} for subset in manifests]
    for left in range(len(ids)):
        for right in range(left + 1, len(ids)):
            overlap = ids[left] & ids[right]
            if overlap:
                raise ValueError(f"환자 데이터 누수: 분할 {left}/{right}에 {len(overlap)}명 중복")
