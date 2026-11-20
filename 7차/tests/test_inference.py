"""Actual CPU inference must report progress and preserve query/gallery orientation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_reid.inference import extract_embeddings, predict_pair_probabilities


class MeanEncoder(torch.nn.Module):
    def encode(self, x):
        return x.mean(dim=(2, 3))


class InferenceTests(unittest.TestCase):
    def test_reports_first_final_and_preserves_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(3):
                Image.new("RGB", (8, 8), (i * 60, 40, 80)).save(root / f"{i}.png")
            calls = []
            actual = extract_embeddings(MeanEncoder(), root, ["0.png", "1.png", "2.png"],
                                        task="verification", mode="original", strength=0,
                                        device="cpu", batch_size=2, num_workers=0,
                                        progress_callback=lambda done, total: calls.append((done, total)))
            self.assertEqual(calls[0], (0, 2))
            self.assertEqual(calls[-1], (2, 2))
            self.assertEqual(actual.shape, (3, 3))
            np.testing.assert_allclose(actual[:, 0],
                                       (np.array([0, 60, 120]) / 255 - .485) / .229, rtol=1e-5)

    def test_stop_callback_prevents_forward_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (4, 4)).save(root / "0.png")
            def stop(done, total):
                raise KeyboardInterrupt("STOP_REQUESTED")
            with self.assertRaisesRegex(KeyboardInterrupt, "STOP_REQUESTED"):
                extract_embeddings(MeanEncoder(), root, ["0.png"], task="verification",
                                   mode="original", strength=0, device="cpu", batch_size=1,
                                   num_workers=0, progress_callback=stop)

    def test_empty_input_raises_clear_value_error(self):
        with self.assertRaisesRegex(ValueError, "empty|영상|names"):
            extract_embeddings(MeanEncoder(), Path("."), [], task="verification",
                               mode="original", strength=0, device="cpu", batch_size=1,
                               num_workers=0)

    def test_pair_predictions_transform_left_side_only(self):
        model = torch.nn.Module()
        model.fc_end = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.fc_end.weight.copy_(torch.tensor([[2., -1.]]))
            model.fc_end.bias.copy_(torch.tensor([.5]))
        original = np.array([[1., 1.], [2., 4.]], dtype=np.float32)
        changed = np.array([[5., 2.], [1., 3.]], dtype=np.float32)
        truth, scores = predict_pair_probabilities(model, [("a", "b", 1), ("b", "a", 0)],
                                                   {"a": 0, "b": 1}, changed, original)
        np.testing.assert_array_equal(truth, [1, 0])
        # abs([5,2]-[2,4])=[3,2] -> 4.5; abs([1,3]-[1,1])=[0,2] -> -1.5
        np.testing.assert_allclose(scores, 1 / (1 + np.exp(-np.array([4.5, -1.5]))), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
