"""Small real CPU forward passes through all sweep conditions, not a GPU trial."""

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "CPU torch test dependency unavailable")
class PrivacyCPUIntegrationTests(unittest.TestCase):
    def test_all_transforms_run_without_changing_model_or_source_images(self):
        import torch
        from cxr7_privacy import evaluate_group
        from cxr7_privacy_contract import open_report
        from cxr_reid.evaluation import calculate_verification_metrics
        from cxr_reid.inference import extract_embeddings, predict_pair_probabilities

        class TinyMatcher(torch.nn.Module):
            # A real small torch network is sufficient to test the inference /
            # transform plumbing without pretending this is the ResNet experiment.
            def __init__(self):
                super().__init__()
                self.fc_end = torch.nn.Linear(4, 1)
                with torch.no_grad():
                    self.fc_end.weight.fill_(-.25)
                    self.fc_end.bias.fill_(.5)

            def encode(self, images):
                mean = images.mean(dim=(2, 3))
                variation = images.var(dim=(1, 2, 3), unbiased=False).unsqueeze(1)
                return torch.cat((mean, variation), dim=1).sigmoid()

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                images = base / "images"
                images.mkdir()
                names = ["a0.png", "a1.png", "b0.png", "b1.png"]
                for i, name in enumerate(names):
                    grid = (np.indices((32, 32)).sum(axis=0) % (i + 2) * 40).astype(np.uint8)
                    Image.fromarray(grid).convert("RGB").save(images / name)
                before = {name: hashlib.sha256((images / name).read_bytes()).hexdigest() for name in names}
                model = TinyMatcher().eval()
                model_before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
                events = []

                def extract(mode, strength):
                    return extract_embeddings(model, images, names, task="verification", mode=mode,
                                              strength=strength, device="cpu", batch_size=2, num_workers=0,
                                              progress_callback=lambda done, total: events.append((done, total)))

                original = extract("original", 0)
                pairs = [("a0.png", "a1.png", 1), ("b0.png", "b1.png", 1),
                         ("a0.png", "b1.png", 0), ("b0.png", "a1.png", 0)]
                index = {name: i for i, name in enumerate(names)}

                def measure(query, gallery):
                    labels, scores = predict_pair_probabilities(model, pairs, index, query, gallery)
                    return calculate_verification_metrics(labels, scores)

                baseline = measure(original, original)
                report = open_report(base / "privacy_results.json", {"synthetic_test_only": True})
                evaluate_group(report, "NIH", "verification", baseline,
                               lambda: original, extract, measure, lambda: None)
                rows = report["results"]["NIH"]
                self.assertEqual(len(rows), 21)
                self.assertEqual(rows["blur_000"]["verification"], rows["original_000"]["verification"])
                self.assertTrue(all(np.isfinite(row["verification"]["auc"]) for row in rows.values()))
                self.assertIn((0, 2), events)
                self.assertIn((2, 2), events)
                after = {name: hashlib.sha256((images / name).read_bytes()).hexdigest() for name in names}
                self.assertEqual(before, after)
                for key, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, model_before[key], rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
