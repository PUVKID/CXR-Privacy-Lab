"""The sweep must not fabricate metrics, mix checkpoints, or lose 0-strength baselines."""

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class PrivacyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.find_spec("cxr7_privacy_contract")
        cls.api = None
        if spec is not None:
            import cxr7_privacy_contract
            cls.api = cxr7_privacy_contract

    def require_api(self):
        self.assertIsNotNone(self.api, "The audited privacy sweep contract has not been implemented")
        return self.api

    def test_zero_strength_aliases_but_nonzero_blur_is_measured(self):
        api = self.require_api()
        plan = api.condition_plan()
        self.assertEqual(len(plan), 21)
        conditions = {row["id"]: row for row in plan}
        self.assertEqual(conditions["blur_000"]["operator_mode"], "original")
        self.assertEqual(conditions["mosaic_000"]["operator_mode"], "original")
        self.assertEqual(conditions["mosaic_001"]["operator_mode"], "original")
        self.assertEqual(conditions["blur_001"]["operator_mode"], "blur")
        self.assertEqual(conditions["blur_256"]["operator_strength"], 256)
        self.assertEqual(conditions["mosaic_256"]["operator_strength"], 256)

    def test_refuses_unfinished_or_mismatched_baseline_models(self):
        api = self.require_api()
        hashes = {"verification": "a" * 64, "retrieval": "b" * 64}
        artifacts = {"published_cxr_initialization": False,
                     "verification_sha256": hashes["verification"],
                     "retrieval_sha256": hashes["retrieval"]}
        baseline = {"completed": True}
        for name in ("NIH", "COVID19"):
            baseline[name] = {"verification_sha256": hashes["verification"],
                              "retrieval_sha256": hashes["retrieval"],
                              "verification": {"auc": .9, "accuracy": .8},
                              "retrieval": {"precision_at_1": .7, "map_at_r": .6,
                                            "r_precision": .6}}
        api.validate_baseline(artifacts, baseline, hashes)
        current_code = {"evaluation.py": "c" * 64}
        baseline["evaluator_code_sha256"] = current_code
        self.assertTrue("evaluator_code_sha256" in __import__("inspect").signature(api.validate_baseline).parameters,
                        "Baseline validation must bind the code used for comparison")
        api.validate_baseline(artifacts, baseline, hashes, evaluator_code_sha256=current_code)
        with self.assertRaisesRegex(ValueError, "evaluator"):
            api.validate_baseline(artifacts, baseline, hashes,
                                  evaluator_code_sha256={"evaluation.py": "d" * 64})
        baseline["completed"] = False
        with self.assertRaisesRegex(ValueError, "complete"):
            api.validate_baseline(artifacts, baseline, hashes)
        baseline["completed"] = True
        baseline["COVID19"]["retrieval_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            api.validate_baseline(artifacts, baseline, hashes)

    def test_resume_rejects_configuration_changes_without_overwriting(self):
        api = self.require_api()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "privacy_results.json"
            report = api.open_report(path, {"checkpoint": "first"})
            api.save_report(path, report)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "binding"):
                api.open_report(path, {"checkpoint": "changed"})
            self.assertEqual(before, path.read_bytes())

    def test_partial_condition_is_not_skipped_and_csv_has_blank_missing_metrics(self):
        api = self.require_api()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "privacy_results.json"
            report = api.open_report(path, {"checkpoint": "first"})
            condition = api.condition_plan()[2]  # blur strength 1
            api.record_result(report, "NIH", condition, "verification",
                              {"auc": .81, "accuracy": .72})
            self.assertTrue(api.task_is_complete(report, "NIH", "blur_001", "verification"))
            self.assertFalse(api.task_is_complete(report, "NIH", "blur_001", "retrieval"))
            csv_path = Path(folder) / "privacy_results.csv"
            api.write_csv(csv_path, report)
            with csv_path.open(encoding="utf-8-sig", newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["verification_auc"], "0.81")
            self.assertEqual(row["retrieval_precision_at_1"], "")

    def test_privacy_complement_is_empirical_and_not_published_guarantee(self):
        api = self.require_api()
        output = api.attack_reduction(.80, .60)
        self.assertAlmostEqual(output["privacy_score_at_1"], 40.0)
        self.assertAlmostEqual(output["rank1_reduction_percentage_points"], 20.0)
        self.assertFalse(output["formal_privacy_guarantee"])
        self.assertIsNone(api.attack_reduction(0.0, .1)["relative_rank1_reduction"])
        with self.assertRaises(ValueError):
            api.attack_reduction(.8, float("nan"))

    def test_nonfinite_or_missing_metrics_never_become_completed(self):
        api = self.require_api()
        with tempfile.TemporaryDirectory() as folder:
            report = api.open_report(Path(folder) / "missing.json", {"checkpoint": "first"})
            condition = api.condition_plan()[0]
            for metrics in ({"auc": .9}, {"auc": float("nan"), "accuracy": .9}):
                with self.assertRaises(ValueError):
                    api.record_result(report, "NIH", condition, "verification", metrics)
            self.assertFalse(api.task_is_complete(report, "NIH", "original_000", "verification"))

    def test_stale_completed_flag_cannot_hide_missing_condition(self):
        api = self.require_api()
        self.assertTrue(hasattr(api, "report_is_complete"), "Completion must inspect actual conditions")
        with tempfile.TemporaryDirectory() as folder:
            report = api.open_report(Path(folder) / "missing.json", {"checkpoint": "first"})
            report["completed"] = True
            self.assertFalse(api.report_is_complete(report))


if __name__ == "__main__":
    unittest.main()
