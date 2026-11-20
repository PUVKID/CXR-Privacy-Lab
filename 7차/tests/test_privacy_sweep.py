"""Exercise orchestration with real metrics and synthetic (non-patient) embeddings."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr7_privacy_contract import open_report, task_is_complete
from cxr_reid.evaluation import calculate_retrieval_metrics, calculate_verification_metrics


class PrivacySweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = None
        if importlib.util.find_spec("cxr7_privacy") is not None:
            import cxr7_privacy
            cls.api = cxr7_privacy

    def require_api(self):
        self.assertIsNotNone(self.api, "Privacy sweep orchestration has not been implemented")
        return self.api

    def test_original_gallery_is_reused_and_resuming_completed_group_does_no_work(self):
        api = self.require_api()
        vectors = np.array([[1., 0.], [.9, .1], [0., 1.], [.1, .9]], dtype=np.float32)
        ids = ["a", "a", "b", "b"]
        names = ["a1", "a2", "b1", "b2"]
        baseline = calculate_retrieval_metrics(vectors, vectors, ids, ids,
                                              same_source=True, gallery_keys=names)
        calls = {"original": 0, "query": 0, "saved": 0}

        def original():
            calls["original"] += 1
            return vectors

        def transformed(mode, strength):
            calls["query"] += 1
            return vectors[:, ::-1].copy()

        def measure(query, gallery):
            return calculate_retrieval_metrics(query, gallery, ids, ids,
                                               same_source=True, gallery_keys=names)

        def saved():
            calls["saved"] += 1

        with tempfile.TemporaryDirectory() as folder:
            report = open_report(Path(folder) / "privacy_results.json", {"test": "fixture"})
            api.evaluate_group(report, "NIH", "retrieval", baseline, original,
                               transformed, measure, saved)
            self.assertEqual(calls, {"original": 1, "query": 17, "saved": 21})
            results = report["results"]["NIH"]
            self.assertEqual(results["blur_001"]["retrieval"]["precision_at_1"], 0.0)
            self.assertEqual(results["blur_000"]["retrieval"]["precision_at_1"], 1.0)
            self.assertEqual(results["mosaic_001"]["retrieval"]["privacy_score_at_1"], 0.0)
            self.assertFalse(report["completed"])
            api.evaluate_group(report, "NIH", "retrieval", baseline, original,
                               transformed, measure, saved)
            self.assertEqual(calls, {"original": 1, "query": 17, "saved": 21})

    def test_global_stage_order_finishes_all_blur_before_any_mosaic(self):
        api = self.require_api()
        self.assertTrue(hasattr(api, "ordered_groups"), "Global stage scheduler is missing")
        vectors = np.array([[1., 0.], [.9, .1], [0., 1.], [.1, .9]], dtype=np.float32)
        ids = ["a", "a", "b", "b"]
        metric_by_task = {
            "verification": calculate_verification_metrics([0, 0, 1, 1], [.1, .2, .8, .9]),
            "retrieval": calculate_retrieval_metrics(vectors, vectors, ids, ids, same_source=True),
        }
        visited = []
        with tempfile.TemporaryDirectory() as folder:
            report = open_report(Path(folder) / "privacy_results.json", {"test": "stage_sequence"})
            for stage, task, dataset, conditions in api.ordered_groups():
                api.evaluate_group(
                    report, dataset, task, metric_by_task[task], lambda: vectors,
                    lambda mode, strength: vectors,
                    lambda query, gallery: dict(metric_by_task[task]), lambda: None,
                    on_condition=lambda condition: visited.append((condition["mode"], task, dataset)),
                    conditions=conditions)
            self.assertEqual([item[0] for item in visited], ["original"] * 4 + ["blur"] * 40 + ["mosaic"] * 40)
            self.assertTrue(report["completed"])

    def test_failed_condition_remains_pending_but_prior_results_survive(self):
        api = self.require_api()
        with tempfile.TemporaryDirectory() as folder:
            report = open_report(Path(folder) / "privacy_results.json", {"test": "interruption"})

            def failure(mode, strength):
                raise RuntimeError("decode error")

            with self.assertRaisesRegex(RuntimeError, "decode"):
                api.evaluate_group(report, "NIH", "verification", {"auc": .8, "accuracy": .7},
                                   lambda: np.zeros((2, 2)), failure, lambda q, g: {}, lambda: None)
            self.assertTrue(task_is_complete(report, "NIH", "original_000", "verification"))
            self.assertTrue(task_is_complete(report, "NIH", "blur_000", "verification"))
            self.assertFalse(task_is_complete(report, "NIH", "blur_001", "verification"))

    def test_embedding_cache_reuses_only_matching_finite_untampered_arrays(self):
        api = self.require_api()
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "original.npy"
            metadata = {"checkpoint": "a", "names": "b", "n_images": 2}
            vectors = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
            result = api.cached_original(cache, metadata, lambda: vectors)
            np.testing.assert_array_equal(result, vectors)

            def unnecessary_extract():
                raise AssertionError("Cache was not reused")

            reused = api.cached_original(cache, metadata, unnecessary_extract)
            np.testing.assert_array_equal(reused, vectors)
            with self.assertRaisesRegex(ValueError, "binding"):
                api.cached_original(cache, {**metadata, "checkpoint": "different"}, unnecessary_extract)
            cache.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                api.cached_original(cache, metadata, unnecessary_extract)

    def test_resume_exports_all_examples_and_restores_only_a_missing_output(self):
        api = self.require_api()
        self.assertTrue(hasattr(api, "refresh_examples"), "Completed sweeps must verify their example files")
        from cxr7_examples import save_condition_example
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            project, official, output = base / "project", base / "official", base / "out"
            for dataset in ("NIH_ChestXray14", "COVID19"):
                images = project / "data" / dataset / "images"
                images.mkdir(parents=True)
                Image.new("RGB", (16, 16), "red").save(images / "a.png")
            verification = official / "PatientVerification"
            retrieval = official / "PatientRetrieval/csv_files"
            verification.mkdir(parents=True)
            retrieval.mkdir(parents=True)
            (verification / "test_list.txt").write_text("a.png\n", encoding="utf-8")
            for name in ("test.csv", "covid_data.csv"):
                (retrieval / name).write_text("Image Index,Patient ID\na.png,p1\n", encoding="utf-8")
            report = open_report(output / "privacy_results.json", {"test": "examples"})
            export = lambda mode, strength: save_condition_example(project, official, output, mode, strength)
            api.refresh_examples(report, export, lambda: None)
            self.assertEqual(len(list((output / "대표사진").rglob("*.png"))), 42)
            deleted = output / report["examples"]["blur_002"]["NIH"]["image_path"]
            deleted.unlink()
            api.refresh_examples(report, export, lambda: None)
            self.assertTrue(deleted.is_file())
            self.assertEqual(len(list((output / "대표사진").rglob("*.png"))), 42)


if __name__ == "__main__":
    unittest.main()
