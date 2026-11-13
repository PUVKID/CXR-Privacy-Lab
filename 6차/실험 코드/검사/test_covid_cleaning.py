"""COVID 외부 평가의 독립 정제 근거와 쌍·검색의 동일 허용 집합을 검사합니다."""

from contextlib import contextmanager
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cxr_reid import cleaning
from cxr_reid.dataset_validation import verify_current_file_hashes, inspect_external_dataset, validation_engine_signature
from cxr_reid.provenance import signature


@contextmanager
def covid_fixture():
    # 합성 파일만 운영체제 임시 폴더에 두고 실제 의료 원본의 위치는 바꾸지 않습니다.
    with tempfile.TemporaryDirectory(prefix="cxr-covid-synthetic-") as directory:
        base = Path(directory)
        source = base / "validation_runs/origin/covid_pixels.private.json"
        source.parent.mkdir(parents=True)
        rows = [("case_a.png", "patient_a", "same"), ("case_b.png", "patient_b", "same"),
                ("case_c.png", "patient_c", "c"), ("case_d.png", "patient_c", "d"),
                ("case_e.png", "patient_e", "e")]
        records = {name: {"sha256": hashlib.sha256(name.encode()).hexdigest(),
                          "pixel_sha256": hashlib.sha256(pixel.encode()).hexdigest(),
                          "patient_id": patient, "split": "external"} for name, patient, pixel in rows}
        raw = {"images": len(records), "decoded_attempts": len(records), "records": records,
               "validation_engine": "synthetic-covid-origin", "content_fingerprint": signature(records),
               "blocking_duplicate_groups": 1, "decode_errors": []}
        blob = json.dumps(raw).encode();source.write_bytes(blob)
        proof = {"sha256": hashlib.sha256(blob).hexdigest(), "validation_engine": raw["validation_engine"],
                 "content_fingerprint": raw["content_fingerprint"], "images": len(records), "bytes": len(blob)}
        # 합성 COVID 근거를 주입하며 실제 고정 raw 앵커는 변경하지 않습니다.
        with patch.object(cleaning, "TRUSTED_COVID_RAW", proof, create=True):
            yield base, source, raw


class CovidCleaningContract(unittest.TestCase):
    def test_covid_schema_is_independent_and_raw_blocking_is_preserved(self):
        with covid_fixture() as (base, source, raw):
            before = source.read_bytes()
            data = cleaning.derive_manifest(base, source, engine="new-engine", dataset="COVID19")
            self.assertNotEqual(data["derived_schema"], cleaning.SCHEMA)
            self.assertEqual(data["blocking_duplicate_groups"], 1)
            self.assertEqual(data["records"], raw["records"])
            self.assertEqual(source.read_bytes(), before)
            allowed = cleaning.validate_derived_manifest(base, data, engine="new-engine", dataset="COVID19")
            self.assertEqual(allowed, {"case_c.png", "case_d.png", "case_e.png"})

    def test_nih_or_other_raw_cannot_be_accepted_as_covid(self):
        with covid_fixture() as (_, source, _):
            with self.assertRaisesRegex(ValueError, "고정 SHA256"):
                cleaning.read_trusted_raw(source)
            source.write_bytes(source.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "고정"):
                cleaning.read_trusted_raw(source, dataset="COVID19")

    def test_rehashed_covid_exclusion_tamper_is_rejected(self):
        with covid_fixture() as (base, source, _):
            data = cleaning.derive_manifest(base, source, engine="new-engine", dataset="COVID19")
            data["cleaning"]["allowed_names"].append("case_a.png")
            data["cleaning"]["allowed_names_sha256"] = signature(data["cleaning"]["allowed_names"])
            with self.assertRaisesRegex(ValueError, "고정 정제 정책"):
                cleaning.validate_derived_manifest(base, data, engine="new-engine", dataset="COVID19")

    def test_covid_report_binds_manifest_and_cannot_substitute_nih_selection(self):
        with covid_fixture() as (base, source, _):
            data = cleaning.derive_manifest(base, source, engine="new-engine", dataset="COVID19")
            manifest = base / "validation_runs/new/covid_pixels.private.json"
            manifest.parent.mkdir(parents=True);manifest.write_text(json.dumps(data), encoding="utf-8")
            selection = {"manifest_relative": manifest.relative_to(base).as_posix(),
                         "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                         "policy": cleaning.POLICY, "allowed_names_sha256": data["cleaning"]["allowed_names_sha256"],
                         "summary": data["cleaning"]["summary"]}
            report = {"COVID19": {"selection": selection}}
            allowed = cleaning.selection_from_report(base, report, engine="new-engine", dataset="COVID19")
            self.assertEqual(len(allowed), 3)
            with self.assertRaises(ValueError):
                cleaning.selection_from_report(base, report, engine="new-engine")
            changed = copy.deepcopy(report);changed["COVID19"]["selection"]["summary"]["excluded_images"] = 0
            with self.assertRaises(ValueError):
                cleaning.selection_from_report(base, changed, engine="new-engine", dataset="COVID19")

    def test_external_pairs_are_projected_without_balancing_and_rows_share_allowed_images(self):
        with covid_fixture() as (base, source, _):
            data = cleaning.derive_manifest(base, source, engine="new-engine", dataset="COVID19")
            allowed = set(data["cleaning"]["allowed_names"])
            pairs = [("case_c.png", "case_d.png", 1), ("case_c.png", "case_e.png", 0),
                     ("case_d.png", "case_e.png", 0), ("case_a.png", "case_e.png", 0)]
            projected = cleaning.project_pairs(pairs, allowed)
            self.assertEqual(projected, pairs[:3])
            self.assertEqual(sum(label == 1 for _, _, label in projected), 1)
            self.assertEqual(sum(label == 0 for _, _, label in projected), 2)
            rows = [(name, name) for name in ["case_a.png", "case_c.png", "case_d.png", "case_e.png"]]
            self.assertEqual({name for name, _ in cleaning.project_rows(rows, allowed)}, allowed)
            selected_rows, selected_pairs, summary = cleaning.project_external_evaluation(rows, pairs, allowed)
            self.assertEqual({name for name, _ in selected_rows}, allowed)
            self.assertEqual(selected_pairs, pairs[:3])
            self.assertEqual(summary["experiment_test_pairs"], 3)
            self.assertEqual(summary["experiment_positive_pairs"], 1)
            self.assertEqual(summary["experiment_negative_pairs"], 2)

    def test_empty_external_selection_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "평가 가능한"):
            cleaning.project_external_evaluation([("synthetic.png", "synthetic")], [], set())

    def test_current_byte_gate_checks_covid_images_excluded_by_policy(self):
        with covid_fixture() as (base, source, raw):
            cleaning.derive_manifest(base, source, engine="new-engine", dataset="COVID19")
            images = base / "images";images.mkdir()
            for name in raw["records"]:(images / name).write_bytes(name.encode())
            self.assertEqual(verify_current_file_hashes(images, raw["records"], workers=1), 5)
            (images / "case_a.png").write_bytes(b"changed excluded bytes")
            with self.assertRaisesRegex(ValueError, "원영상 바이트"):
                verify_current_file_hashes(images, raw["records"], workers=1)

    def test_prepare_reuses_fixed_covid_audit_and_rechecks_all_781_bytes_before_publishing(self):
        # 781개 합성 파일의 외장 HDD 생성 비용과 실제 원영상 검증을 분리합니다.
        with tempfile.TemporaryDirectory(prefix="cxr-covid-synthetic-781-") as directory:
            base = Path(directory)
            images = base / "data/COVID19/images";images.mkdir(parents=True)
            rows, records = [], {}
            for index in range(781):
                name = f"synthetic_image_{index:04d}.png"
                patient = f"synthetic_patient_{index // 2}" if index > 1 else f"synthetic_conflict_{index}"
                payload = name.encode();(images / name).write_bytes(payload)
                rows.append((name, patient))
                pixel = "synthetic_same_pixel" if index < 2 else f"synthetic_pixel_{index}"
                records[name] = {"sha256": hashlib.sha256(payload).hexdigest(),
                                 "pixel_sha256": hashlib.sha256(pixel.encode()).hexdigest(),
                                 "patient_id": patient, "split": "external"}
            csv_file = base / "official/PatientRetrieval/csv_files/covid_data.csv"
            csv_file.parent.mkdir(parents=True)
            with csv_file.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file);writer.writerow(["Image Index", "Patient ID"]);writer.writerows(rows)
            pair_file = base / "official/PatientVerification/image_pairs/pairs_testing_covid19.txt"
            pair_file.parent.mkdir(parents=True)
            pairs = [(2, 3, 1), (4, 5, 1), (2, 4, 0), (0, 4, 0)]
            pair_file.write_text("".join(f"{rows[a][0]} {rows[b][0]} {label}\n" for a, b, label in pairs), encoding="utf-8")
            raw = {"images": 781, "decoded_attempts": 781, "records": records,
                   "validation_engine": "synthetic-covid-origin", "content_fingerprint": signature(records),
                   "blocking_duplicate_groups": 1, "decode_errors": []}
            source = base / "validation_runs/원본_전수검사/covid_pixels.private.json"
            source.parent.mkdir(parents=True);blob = json.dumps(raw).encode();source.write_bytes(blob)
            proof = {"sha256": hashlib.sha256(blob).hexdigest(), "validation_engine": raw["validation_engine"],
                     "content_fingerprint": raw["content_fingerprint"], "images": 781, "bytes": len(blob)}
            fixed = {"source_files_sha256": {
                "PatientRetrieval/csv_files/covid_data.csv": hashlib.sha256(csv_file.read_bytes()).hexdigest(),
                "PatientVerification/image_pairs/pairs_testing_covid19.txt": hashlib.sha256(pair_file.read_bytes()).hexdigest()}}
            destination = base / "validation_runs/new/covid_pixels.private.json"
            with patch.object(cleaning, "TRUSTED_COVID_RAW", proof), \
                    patch("cxr_reid.dataset_validation.sources", return_value=fixed), \
                    patch("cxr_reid.dataset_validation.inspect_image_pixels", side_effect=AssertionError("재디코딩 호출 금지")):
                report = inspect_external_dataset(images, csv_file, pair_file, expected_images=781,
                    expected_pairs=4, dataset_name="합성 외부 집합", pixel_manifest_path=destination)
                self.assertEqual(report["paper_retrieval_images"], 781)
                self.assertEqual(report["pixel_validation"]["blocking_duplicate_groups"], 1)
                self.assertEqual(report["experiment_images"], 779)
                self.assertEqual(report["experiment_test_pairs"], 3)
                self.assertEqual(report["experiment_positive_pairs"], 2)
                self.assertEqual(report["experiment_negative_pairs"], 1)
                allowed = cleaning.selection_from_report(base, {"COVID19": report},
                    engine=validation_engine_signature(), dataset="COVID19")
                self.assertEqual(len(allowed), 779)
                (images / rows[0][0]).write_bytes(b"changed excluded image")
                with self.assertRaisesRegex(ValueError, "원영상 바이트"):
                    inspect_external_dataset(images, csv_file, pair_file, expected_images=781,
                        expected_pairs=4, dataset_name="합성 외부 집합",
                        pixel_manifest_path=base / "validation_runs/new2/covid_pixels.private.json")
                self.assertFalse((base / "validation_runs/new2/covid_pixels.private.json").exists())


if __name__ == "__main__":
    unittest.main()
