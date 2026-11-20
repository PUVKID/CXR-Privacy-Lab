"""Checkpoint tests with real tensors, optimizer state and CBM buffers."""

from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.losses import CrossBatchContrastive
from training.state import save_checkpoint, load_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_weights_optimizer_queue_and_rng_resume_together(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        memory = CrossBatchContrastive(embedding_size=2, memory_size=8)
        inputs = torch.tensor([[1., 2., 3.], [3., 2., 1.], [1., 3., 2.], [3., 1., 2.]])
        labels = torch.tensor([0, 0, 1, 1])
        memory(model(inputs), labels).backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, model, optimizer, metadata={"binding": "fixture", "epoch": 1}, memory=memory)
            expected_draws = (random.random(), np.random.random(), torch.rand(1))
            copied = torch.nn.Linear(3, 2)
            copied_optimizer = torch.optim.Adam(copied.parameters(), lr=.1)
            copied_memory = CrossBatchContrastive(embedding_size=2, memory_size=8)
            metadata = load_checkpoint(path, copied, copied_optimizer, binding="fixture", memory=copied_memory)
            actual_draws = (random.random(), np.random.random(), torch.rand(1))
            self.assertEqual(metadata["epoch"], 1)
            self.assertEqual(actual_draws[:2], expected_draws[:2])
            torch.testing.assert_close(actual_draws[2], expected_draws[2], rtol=0, atol=0)
            for original, loaded in zip(model.parameters(), copied.parameters()):
                torch.testing.assert_close(original, loaded, rtol=0, atol=0)
                torch.testing.assert_close(optimizer.state[original]["exp_avg"],
                                           copied_optimizer.state[loaded]["exp_avg"], rtol=0, atol=0)
            for key, value in memory.state_dict().items():
                torch.testing.assert_close(value, copied_memory.state_dict()[key], rtol=0, atol=0)

    def test_mismatched_binding_refuses_before_loading_model(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, model, optimizer, metadata={"binding": "old"})
            with torch.no_grad():
                model.weight.fill_(100.)
            expected = model.weight.detach().clone()
            with self.assertRaises(ValueError):
                load_checkpoint(path, model, optimizer, binding="new")
            torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
