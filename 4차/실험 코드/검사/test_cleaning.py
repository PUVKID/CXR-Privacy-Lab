"""중복 정제의 동일 허용 집합·불변 근거·오염 차단을 검사합니다."""

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cxr_reid import cleaning
from cxr_reid.dataset_validation import verify_current_file_hashes
from cxr_reid.provenance import signature


def records_fixture():
    rows = [("1_a.png", "1", "train", "a"), ("1_b.png", "1", "train", "a"),
            ("1_c.png", "1", "train", "c"), ("2_a.png", "2", "train", "x"),
            ("3_a.png", "3", "validation", "x"), ("4_a.png", "4", "test", "t"),
            ("4_b.png", "4", "test", "t"), ("5_a.png", "5", "train", "g")]
    return {name: {"sha256": hashlib.sha256(name.encode()).hexdigest(),
                   "pixel_sha256": hashlib.sha256(pixel.encode()).hexdigest(),
                   "patient_id": patient, "split": split} for name, patient, split, pixel in rows}


@contextmanager
def trusted_fixture():
    tmp = Path(__file__).resolve().parents[2] / "work/tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp) as directory:
        base = Path(directory)
        path = base / "validation_runs/origin/nih_pixels.private.json"
        path.parent.mkdir(parents=True)
        records = records_fixture()
        raw = {"records": records, "images": len(records), "decoded_attempts": len(records),
               "validation_engine": "synthetic-origin-engine", "content_fingerprint": signature(records),
               "blocking_duplicate_groups": 2, "decode_errors": []}
        blob = json.dumps(raw).encode()
        path.write_bytes(blob)
        proof = {"sha256": hashlib.sha256(blob).hexdigest(), "validation_engine": raw["validation_engine"],
                 "content_fingerprint": raw["content_fingerprint"], "images": len(records)}
        # 합성 입력의 신뢰 근거를 주입하는 단위 검사이며 실제 고정 baseline을 변경하지 않습니다.
        with patch.object(cleaning, "TRUSTED_RAW", proof):
            yield base, path, raw


class CleaningContract(unittest.TestCase):
    def test_conflicting_identity_drops_every_image_and_same_patient_keeps_first_filename(self):
        result = cleaning.canonical_selection(records_fixture())
        self.assertEqual(result["allowed_names"], ["1_a.png", "1_c.png", "4_a.png", "5_a.png"])
        self.assertEqual(result["summary"]["excluded_images"], 4)
        self.assertEqual(result["summary"]["cross_patient_excluded_images"], 2)
        self.assertEqual(result["summary"]["remaining_duplicate_groups"], 0)

    def test_record_order_never_changes_canonical_or_exclusion_policy(self):
        rows = records_fixture()
        self.assertEqual(cleaning.canonical_selection(rows), cleaning.canonical_selection(dict(reversed(list(rows.items())))))

    def test_same_patient_cross_split_is_blocked_instead_of_canonicalized(self):
        rows = records_fixture()
        rows["1_b.png"]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "다른 분할"):
            cleaning.canonical_selection(rows)

    def test_pairs_rows_and_singleton_training_eligibility_share_the_allowed_set(self):
        rows = records_fixture()
        allowed = cleaning.canonical_selection(rows)["allowed_names"]
        pairs = [("1_a.png", "1_b.png", 1), ("1_a.png", "1_c.png", 1),
                 ("1_a.png", "2_a.png", 0), ("1_a.png", "5_a.png", 0)]
        self.assertEqual(cleaning.project_pairs(pairs, allowed), [pairs[1], pairs[3]])
        manifest = [(name, row["patient_id"]) for name, row in rows.items()]
        self.assertEqual(len(cleaning.project_rows(manifest, allowed)), 4)
        self.assertEqual(cleaning.project_rows(manifest, allowed, multi_image_only=True),
                         [("1_a.png", "1"), ("1_c.png", "1")])
        names, projected, retrieval = cleaning.project_protocol(
            {"train": list(rows)}, {"train": pairs}, {"train": manifest}, allowed)
        self.assertEqual(names["train"], allowed)
        self.assertEqual(projected["train"], [pairs[1], pairs[3]])
        self.assertEqual(retrieval["train"], [("1_a.png", "1"), ("1_c.png", "1")])

    def test_untrusted_or_malicious_bytes_are_rejected_before_json_parsing(self):
        with trusted_fixture() as (_, source, _):
            source.write_bytes(b"arbitrary non-json or pickle bytes")
            with patch.object(cleaning.json, "loads") as parser:
                with self.assertRaisesRegex(ValueError, "고정 SHA256"):
                    cleaning.read_trusted_raw(source)
                parser.assert_not_called()

    def test_origin_blocking_evidence_is_unchanged_and_valid_derived_policy_is_accepted(self):
        with trusted_fixture() as (base, source, raw):
            before = source.read_bytes()
            result = cleaning.derive_manifest(base, source, engine="current-engine")
            self.assertEqual(result["blocking_duplicate_groups"], 2)
            self.assertEqual(result["records"], raw["records"])
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(cleaning.validate_derived_manifest(base, result, engine="current-engine")), 4)

    def test_stale_derivation_engine_is_rejected(self):
        with trusted_fixture() as (base, source, _):
            result = cleaning.derive_manifest(base, source, engine="old-engine")
            with self.assertRaisesRegex(ValueError, "스키마·엔진"):
                cleaning.validate_derived_manifest(base, result, engine="new-engine")

    def test_rehashed_exclusion_list_contamination_is_rejected(self):
        with trusted_fixture() as (base, source, _):
            result = cleaning.derive_manifest(base, source, engine="current-engine")
            result["cleaning"]["allowed_names"].append("2_a.png")
            result["cleaning"]["allowed_names_sha256"] = signature(result["cleaning"]["allowed_names"])
            with self.assertRaisesRegex(ValueError, "고정 정제 정책"):
                cleaning.validate_derived_manifest(base, result, engine="current-engine")

    def test_changed_source_split_or_pixel_record_is_rejected(self):
        with trusted_fixture() as (base, source, _):
            result = cleaning.derive_manifest(base, source, engine="current-engine")
            result["records"]["1_a.png"]["split"] = "test"
            with self.assertRaisesRegex(ValueError, "원 전수 검사 내용"):
                cleaning.validate_derived_manifest(base, result, engine="current-engine")

    def test_original_bytes_are_checked_even_for_excluded_images(self):
        with trusted_fixture() as (base, _, raw):
            images = base / "images"
            images.mkdir()
            for name in raw["records"]:
                (images / name).write_bytes(name.encode())
            self.assertEqual(verify_current_file_hashes(images, raw["records"], workers=2), 8)
            (images / "2_a.png").write_bytes(b"changed excluded source")
            with self.assertRaisesRegex(ValueError, "원영상 바이트"):
                verify_current_file_hashes(images, raw["records"], workers=2)

    def test_origin_path_escape_is_rejected(self):
        with trusted_fixture() as (base, _, _):
            with self.assertRaises(ValueError):
                cleaning.source_path(base, "../outside/manifest.json")

    def test_report_payload_hash_and_allowed_summary_are_both_bound(self):
        with trusted_fixture() as (base, source, _):
            data = cleaning.derive_manifest(base, source, engine="current-engine")
            target = base / "validation_runs/new/nih_pixels.private.json"
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(data), encoding="utf-8")
            selection = {"manifest_relative": target.relative_to(base).as_posix(),
                         "manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                         "policy": cleaning.POLICY,
                         "allowed_names_sha256": data["cleaning"]["allowed_names_sha256"],
                         "summary": data["cleaning"]["summary"]}
            report = {"NIH": {"selection": selection}}
            self.assertEqual(len(cleaning.selection_from_report(base, report, engine="current-engine")), 4)
            changed = copy.deepcopy(report)
            changed["NIH"]["selection"]["summary"]["excluded_images"] = 0
            with self.assertRaisesRegex(ValueError, "정책·내용"):
                cleaning.selection_from_report(base, changed, engine="current-engine")
            target.write_bytes(target.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "파일 SHA256"):
                cleaning.selection_from_report(base, report, engine="current-engine")

    def test_matching_file_sha_cannot_override_an_unrelated_origin_engine(self):
        with trusted_fixture() as (_, source, _):
            proof = dict(cleaning.TRUSTED_RAW, validation_engine="unrelated-engine")
            with patch.object(cleaning, "TRUSTED_RAW", proof):
                with self.assertRaisesRegex(ValueError, "엔진·내용·수량"):
                    cleaning.read_trusted_raw(source)


if __name__ == "__main__":
    unittest.main()
