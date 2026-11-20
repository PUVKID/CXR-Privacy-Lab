import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr7_reports import paper_comparison, verification_is_complete  # noqa: E402


class PaperComparisonTests(unittest.TestCase):
    def test_reports_differences_without_calling_close_result_identical(self):
        measured = {"verification_auc": 0.9900, "verification_accuracy": 0.9500,
                    "retrieval_precision_at_1": 0.9950, "retrieval_map_at_r": 0.9700}
        output = paper_comparison(measured)
        self.assertEqual(output["verification_auc"]["paper"], 0.9940)
        self.assertAlmostEqual(output["verification_auc"]["difference"], -0.0040)
        self.assertFalse(output["verification_auc"]["identical_reported_value"])
        self.assertFalse(output["verification_accuracy"]["identical_reported_value"])

    def test_missing_metrics_are_not_invented(self):
        self.assertEqual(paper_comparison({}), {})

    def test_resume_does_not_skip_unfinished_bootstrap(self):
        existing = {"verification_sha256": "abcd", "verification": {"auc": 0.9}}
        self.assertFalse(verification_is_complete(existing, "abcd", 10000))
        self.assertTrue(verification_is_complete(existing, "abcd", 0))
        existing["verification"]["confidence_interval"] = {"bootstrap_repeats": 10000}
        self.assertTrue(verification_is_complete(existing, "abcd", 10000))
        self.assertFalse(verification_is_complete(existing, "changed", 10000))

    def test_evaluator_fingerprint_detects_changed_preprocessing(self):
        import cxr7_reports as reports
        self.assertTrue(hasattr(reports, "evaluator_fingerprint"), "Evaluator code binding is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cxr_reid").mkdir()
            for name in ("cxr7_evaluate.py", "cxr7_reports.py", "cxr_reid/models.py",
                         "cxr_reid/evaluation.py", "cxr_reid/inference.py",
                         "cxr_reid/image_ops.py", "cxr_reid/data.py"):
                (root / name).write_text("# original\n", encoding="utf-8")
            first = reports.evaluator_fingerprint(root)
            (root / "cxr_reid/inference.py").write_text("# changed resize\n", encoding="utf-8")
            second = reports.evaluator_fingerprint(root)
            self.assertNotEqual(first, second)
            self.assertEqual(first["cxr_reid/models.py"], second["cxr_reid/models.py"])

    def test_resume_rejects_other_code_or_checkpoint(self):
        import cxr7_reports as reports
        self.assertTrue(hasattr(reports, "validate_evaluation_resume"), "Mixed-evaluation guard is missing")
        saved = {"evaluator_code_sha256": {"inference": "abc"},
                 "NIH": {"verification": {"auc": .9}, "verification_sha256": "v1"},
                 "COVID19": {}}
        reports.validate_evaluation_resume(saved, {"inference": "abc"}, {"verification": "v1", "retrieval": "r1"})
        with self.assertRaisesRegex(ValueError, "code"):
            reports.validate_evaluation_resume(saved, {"inference": "new"}, {"verification": "v1", "retrieval": "r1"})
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            reports.validate_evaluation_resume(saved, {"inference": "abc"}, {"verification": "v2", "retrieval": "r1"})

    def test_retrieval_resume_requires_all_finite_measured_metrics(self):
        import cxr7_reports as reports
        self.assertTrue(hasattr(reports, "retrieval_is_complete"), "Retrieval result completeness guard missing")
        saved = {"retrieval_sha256": "r1"}
        self.assertFalse(reports.retrieval_is_complete(saved, "r1"))
        saved["retrieval"] = {"precision_at_1": .9, "map_at_r": .8, "r_precision": .85}
        self.assertTrue(reports.retrieval_is_complete(saved, "r1"))
        self.assertFalse(reports.retrieval_is_complete(saved, "other"))
        saved["retrieval"]["map_at_r"] = float("nan")
        self.assertFalse(reports.retrieval_is_complete(saved, "r1"))
