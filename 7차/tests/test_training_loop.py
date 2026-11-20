"""Behavior tests for original training-loop fidelity and safe checkpoints."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.engine import optimizer_batch, validation
from training.losses import CrossBatchContrastive


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("visits", torch.tensor(0))

    def forward(self):
        self.visits += 1
        return self.weight.sqrt()


class PixelMeanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, images):
        self.batch_sizes.append(len(images))
        return images.mean(dim=(2, 3))


class HalfPixelMeanModel(PixelMeanModel):
    """CPU fixture reproducing a CUDA-autocast network's float16 output."""
    def forward(self, images):
        return super().forward(images).half()


class LoopTests(unittest.TestCase):
    def test_half_precision_validation_queue_can_be_used_by_next_training_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            images = project / "data/NIH_ChestXray14/images"
            images.mkdir(parents=True)
            rows = []
            for i in range(4):
                name = f"{i//2:08d}_{i%2:03d}.png"
                Image.new("RGB", (1024, 1024), (50+i*30,) * 3).save(images / name)
                rows.append((name, str(i//2)))
            args = SimpleNamespace(project=project, device="cpu", retrieval_workers=0,
                                   verification_workers=0, seed=3)
            memory = CrossBatchContrastive(embedding_size=3, memory_size=8)
            validation(HalfPixelMeanModel(), "retrieval_phase1", ({}, {}, {"validation": rows}),
                       args, 0, memory=memory, sanity_batches=2)
            self.assertFalse(torch.is_inference(memory.memory),
                             "Validation must not turn persistent CBM state into inference tensors")
            vectors = torch.tensor([[1., .2, .1], [.8, .4, .2], [.1, 1., .3], [.2, .8, .5]],
                                   dtype=torch.float16, requires_grad=True)
            memory.train()
            loss = memory(vectors, torch.tensor([0, 0, 1, 1]))
            loss.backward()
            self.assertTrue(torch.isfinite(vectors.grad).all())

    def test_amp_overflow_skips_once_like_native_scaler(self):
        model = ScalarModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=8.)
        try:
            _, result = optimizer_batch(model, optimizer, scaler, model)
        except FloatingPointError as error:
            self.fail("Native AMP should skip an overflowing step, not retry forever: " + str(error))
        self.assertEqual(int(model.visits), 1,
                         "Original native AMP does not rerun the same training batch")
        self.assertEqual(float(model.weight), 0.)
        self.assertFalse(result["optimizer_applied"])
        self.assertEqual(scaler.get_scale(), 4.)

    def test_retrieval_validation_enqueues_all_samples_in_batches32(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            images = project / "data/NIH_ChestXray14/images"
            images.mkdir(parents=True)
            rows = []
            for i in range(34):
                name = f"{i//2:08d}_{i%2:03d}.png"
                Image.new("RGB", (1024, 1024), (40 + i*4,) * 3).save(images / name)
                rows.append((name, str(i//2)))
            args = SimpleNamespace(project=project, device="cpu", retrieval_workers=0,
                                   verification_workers=0, seed=3)
            model = PixelMeanModel()
            memory = CrossBatchContrastive(embedding_size=3, memory_size=128)
            try:
                outcome = validation(model, "retrieval_phase1", ({}, {}, {"validation": rows}),
                                     args, 0, memory=memory)
            except TypeError as error:
                self.fail("validation must execute original CBM state updates: " + str(error))
            self.assertEqual(int(memory.count), 34)
            self.assertEqual(model.batch_sizes[:2], [32, 2])
            self.assertIn("validation_loss", outcome)
            self.assertIn("validation_precision_at_1", outcome)

    def test_sanity_validation_enqueues_only_requested_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            images = project / "data/NIH_ChestXray14/images"
            images.mkdir(parents=True)
            rows = []
            for i in range(34):
                name = f"{i//2:08d}_{i%2:03d}.png"
                Image.new("RGB", (1024, 1024), (40 + i*4,) * 3).save(images / name)
                rows.append((name, str(i//2)))
            args = SimpleNamespace(project=project, device="cpu", retrieval_workers=0,
                                   verification_workers=0, seed=3)
            model = PixelMeanModel()
            memory = CrossBatchContrastive(embedding_size=3, memory_size=128)
            try:
                validation(model, "retrieval_phase1", ({}, {}, {"validation": rows}),
                           args, 0, memory=memory, sanity_batches=1)
            except TypeError as error:
                self.fail("Original Lightning sanity validation must be represented: " + str(error))
            self.assertEqual(int(memory.count), 32)
            self.assertEqual(model.batch_sizes[0], 32)


if __name__ == "__main__":
    unittest.main()
