"""고정된 전수 디코딩 근거에서 동일 픽셀 중복 제거 목록을 결정적으로 파생합니다."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from .provenance import digest, signature


POLICY = "exact_pixel_cross_patient_drop_all_same_patient_first_filename_v1"
SCHEMA = "nih_exact_pixel_cleaning_v1"
TRUSTED_RAW = {
    "sha256": "2ded62030dc4d54214b2bf045741b90b114df129501886701d28a919798228ff",
    "validation_engine": "bff38251c0e5d2040512245bebcbefbff443ce71e74c5ef57258e02c6a9b059f",
    "content_fingerprint": "0d76a4f98f90ed6641d1346eabb01af41391f7aa021804554014151fef94f58b",
    "images": 112120,
}
COVID_SCHEMA = "covid_exact_pixel_cleaning_v1"
TRUSTED_COVID_RAW = {
    "sha256": "b5978c0baf9457b7fc20cd706d8b77d4df3e6160cdffac93d6c04b9e0fd88724",
    "validation_engine": "95fac8ac7657876ace6c8215366897784bb257a16dbc932d5ef5f06667fcdf6d",
    "content_fingerprint": "bd8709157ffa84c8660965b53667009f57e3bef3430b3a89b24e006b493299d3",
    "images": 781,
    "bytes": 187044,
}


def dataset_contract(dataset):
    if dataset == "NIH":
        return TRUSTED_RAW, SCHEMA
    if dataset == "COVID19":
        return TRUSTED_COVID_RAW, COVID_SCHEMA
    raise ValueError("정제 근거는 NIH 또는 COVID19의 고정 계약만 지원합니다.")


def canonical_selection(records):
    groups = defaultdict(list)
    for name, row in records.items():
        if (Path(name).name != name or any(char in name for char in (":", "/", "\\"))
                or set(row) != {"sha256", "pixel_sha256", "patient_id", "split"}
                or any(not isinstance(row[k], str) for k in row)
                or any(not re.fullmatch(r"[0-9a-f]{64}", row[k]) for k in ("sha256", "pixel_sha256"))):
            raise ValueError("픽셀 근거의 영상 경로·레코드 형식이 유효하지 않습니다.")
        groups[row["pixel_sha256"]].append(name)
    excluded, duplicated = [], []
    cross_groups = cross_images = same_groups = 0
    for pixel, names in sorted(groups.items()):
        if len(names) < 2:
            continue
        names = sorted(names)
        patients = {records[n]["patient_id"] for n in names}
        if len(patients) > 1:
            reason, dropped, canonical = "patient_identity_conflict", names, None
            cross_groups += 1
            cross_images += len(names)
        else:
            if len({records[n]["split"] for n in names}) != 1:
                raise ValueError("같은 환자가 서로 다른 분할에 있습니다. 정제로 분할 누수를 우회하지 않습니다.")
            reason, dropped, canonical = "same_patient_exact_pixel_duplicate", names[1:], names[0]
            same_groups += 1
        duplicated.append({"pixel_sha256": pixel, "images": names, "canonical": canonical})
        excluded.extend({"name": n, "reason": reason, "canonical": canonical, "pixel_sha256": pixel} for n in dropped)
    excluded.sort(key=lambda row: row["name"])
    excluded_names = {row["name"] for row in excluded}
    allowed = sorted(set(records) - excluded_names)
    if len({records[n]["pixel_sha256"] for n in allowed}) != len(allowed):
        raise ValueError("정제 후에도 동일 픽셀이 남아 있습니다.")
    summary = {"raw_images": len(records), "allowed_images": len(allowed),
               "excluded_images": len(excluded), "cross_patient_groups": cross_groups,
               "cross_patient_excluded_images": cross_images, "same_patient_groups": same_groups,
               "same_patient_excluded_images": len(excluded) - cross_images,
               "excluded_by_split": dict(sorted(Counter(records[n]["split"] for n in excluded_names).items())),
               "remaining_duplicate_groups": 0}
    return {"policy": POLICY, "allowed_names": allowed, "excluded": excluded,
            "duplicate_groups": duplicated, "summary": summary,
            "allowed_names_sha256": signature(allowed)}


def read_trusted_raw(path, *, dataset="NIH"):
    trusted, _ = dataset_contract(dataset)
    raw = Path(path).read_bytes()
    import hashlib
    if "bytes" in trusted and len(raw) != trusted["bytes"]:
        raise ValueError("기존 전수 디코딩 근거의 고정 크기가 다릅니다.")
    if hashlib.sha256(raw).hexdigest() != trusted["sha256"]:
        raise ValueError("기존 전수 디코딩 근거의 고정 SHA256가 다릅니다.")
    data = json.loads(raw)
    if (data.get("validation_engine") != trusted["validation_engine"]
            or data.get("content_fingerprint") != trusted["content_fingerprint"]
            or signature(data["records"]) != trusted["content_fingerprint"]
            or data.get("images") != trusted["images"]
            or data.get("decoded_attempts") != trusted["images"]
            or len(data["records"]) != trusted["images"] or data.get("decode_errors")):
        raise ValueError("기존 전수 디코딩 근거의 엔진·내용·수량 계약이 다릅니다.")
    return data


def source_path(base, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("검사 근거의 상대 경로가 유효하지 않습니다.")
    resolved = (Path(base) / path).resolve()
    if not resolved.is_relative_to((Path(base) / "validation_runs").resolve()):
        raise ValueError("검사 근거는 validation_runs 안에 있어야 합니다.")
    return resolved


def derive_manifest(base, source, *, engine, dataset="NIH"):
    trusted, schema = dataset_contract(dataset)
    base, source = Path(base).resolve(), Path(source).resolve()
    if not source.is_relative_to(base / "validation_runs"):
        raise ValueError("원 전수 검사 근거는 validation_runs 안에 있어야 합니다.")
    raw = read_trusted_raw(source, dataset=dataset)
    # 원래 images/blocked/records는 바꾸지 않고 별도 정책 필드만 더합니다.
    return {**raw, "derived_schema": schema, "derivation_engine": engine,
            "source_decode_audit": {"relative_path": source.relative_to(base).as_posix(), **trusted},
            "cleaning": canonical_selection(raw["records"])}


def validate_derived_manifest(base, data, *, engine, dataset="NIH"):
    trusted, schema = dataset_contract(dataset)
    if data.get("derived_schema") != schema or data.get("derivation_engine") != engine:
        raise ValueError("파생 정제 근거의 스키마·엔진이 다릅니다.")
    audit = data.get("source_decode_audit", {})
    if {k: audit.get(k) for k in trusted} != trusted:
        raise ValueError("파생 정제 근거의 원 검사 출처가 다릅니다.")
    raw = read_trusted_raw(source_path(base, audit.get("relative_path", "")), dataset=dataset)
    if any(data.get(k) != value for k, value in raw.items()):
        raise ValueError("파생 근거가 보존해야 할 원 전수 검사 내용을 바꿨습니다.")
    expected = canonical_selection(raw["records"])
    if signature(data.get("cleaning")) != signature(expected):
        raise ValueError("제외·허용 목록이 고정 정제 정책을 다시 계산한 결과와 다릅니다.")
    return set(expected["allowed_names"])


def selection_from_report(base, report, *, engine, dataset="NIH"):
    dataset_contract(dataset)
    key = "NIH_ChestXray14" if dataset == "NIH" else "COVID19"
    dataset_report = report.get(dataset) or report.get("datasets", {}).get(key, {})
    selection = dataset_report.get("selection")
    if not selection:
        raise ValueError("누수 제거 후 재학습에는 검증된 정제 목록이 필요합니다.")
    manifest = source_path(base, selection["manifest_relative"])
    if digest(manifest) != selection["manifest_sha256"]:
        raise ValueError("보고서의 정제 근거 파일 SHA256가 다릅니다.")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    allowed = validate_derived_manifest(base, data, engine=engine, dataset=dataset)
    if (selection.get("policy") != POLICY
            or selection.get("allowed_names_sha256") != signature(sorted(allowed))
            or signature(selection.get("summary")) != signature(data["cleaning"]["summary"])):
        raise ValueError("보고서와 정제 목록의 정책·내용이 다릅니다.")
    return allowed


def project_pairs(pairs, allowed):
    allowed = set(allowed)
    return [(a, b, label) for a, b, label in pairs if a in allowed and b in allowed]


def project_rows(rows, allowed, *, multi_image_only=False):
    selected = [(name, patient) for name, patient in rows if name in allowed]
    if multi_image_only:
        counts = Counter(patient for _, patient in selected)
        selected = [(name, patient) for name, patient in selected if counts[patient] > 1]
    return selected


def project_external_evaluation(rows, pairs, allowed):
    """외부 검증 쌍과 검색 목록을 한 허용 집합에 투영하며 균형을 보충하지 않습니다."""
    selected_rows = project_rows(rows, allowed)
    selected_pairs = project_pairs(pairs, allowed)
    if not selected_rows or not selected_pairs:
        raise ValueError("외부 정제 후 평가 가능한 영상 또는 쌍이 없습니다.")
    patient_counts = Counter(patient for _, patient in selected_rows)
    summary = {"experiment_images": len(selected_rows), "experiment_patients": len(patient_counts),
               "experiment_test_pairs": len(selected_pairs),
               "experiment_positive_pairs": sum(label == 1 for _, _, label in selected_pairs),
               "experiment_negative_pairs": sum(label == 0 for _, _, label in selected_pairs),
               "experiment_eligible_queries": sum(count for count in patient_counts.values() if count > 1)}
    return selected_rows, selected_pairs, summary


def project_protocol(names, pairs, retrieval, allowed):
    """세 학습·선택 분할을 평가와 같은 전역 허용 집합에 투영합니다."""
    allowed = set(allowed)
    return ({phase: [name for name in items if name in allowed] for phase, items in names.items()},
            {phase: project_pairs(items, allowed) for phase, items in pairs.items()},
            {phase: project_rows(items, allowed, multi_image_only=(phase == "train"))
             for phase, items in retrieval.items()})
