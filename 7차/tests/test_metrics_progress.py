from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_reid.evaluation import bootstrap_auc_ci, calculate_retrieval_metrics


class MetricProgressTests(unittest.TestCase):
    def test_bootstrap_has_live_progress_and_valid_interval(self):
        calls = []
        result = bootstrap_auc_ci([0, 0, 1, 1], [.1, .2, .8, .9], 30, seed=3,
                                   progress_callback=lambda done, total: calls.append((done, total)))
        self.assertEqual(calls[0], (0, 30))
        self.assertEqual(calls[-1], (30, 30))
        self.assertEqual(result["auc_ci_lower"], 1.0)
        self.assertEqual(result["auc_ci_upper"], 1.0)

    def test_bootstrap_stop_is_honored(self):
        def stop(done, total):
            raise KeyboardInterrupt("STOP_REQUESTED")
        with self.assertRaisesRegex(KeyboardInterrupt, "STOP_REQUESTED"):
            bootstrap_auc_ci([0, 0, 1, 1], [.1, .2, .8, .9], 10,
                              progress_callback=stop)

    def test_distinct_patient_retrieval_removes_exact_original_self(self):
        # A1 is nearest to transformed A1, but must not count for other-exam patient matching.
        original = np.array([[1., 0.], [0., 1.], [.9, .1], [.8, .2]], np.float32)
        labels = ["A", "A", "B", "B"]
        score = calculate_retrieval_metrics(original, original, labels, labels,
                                            same_source=True, gallery_keys=["a1", "a2", "b1", "b2"])
        # A1->B1 incorrect, A2->B2 incorrect, B1->A1 incorrect, B2->B1 correct.
        self.assertEqual(score["precision_at_1"], .25)

    def test_retrieval_metric_progress_reaches_all_queries(self):
        calls = []
        vectors = np.array([[1., 0.], [1., .1], [0., 1.], [.1, 1.]], np.float32)
        result = calculate_retrieval_metrics(vectors, vectors, ["A", "A", "B", "B"],
                                             ["A", "A", "B", "B"], same_source=True,
                                             block_size=2,
                                             progress_callback=lambda n, total: calls.append((n, total)))
        self.assertEqual(calls, [(0, 4), (2, 4), (4, 4)])
        self.assertEqual(result["precision_at_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
