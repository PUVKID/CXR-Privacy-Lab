"""Small CPU arithmetic tests; these are NOT RTX4090 memory/performance tests."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.engine import create_model, optimizer_batch
from training import engine
from training.losses import CrossBatchContrastive


class TrainingArithmeticTests(unittest.TestCase):
    def test_evaluation_smoke_checks_original_size_fp32_without_grad_and_restores_train(self):
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(3, 128)
                self.observed = None

            def forward(self, images):
                self.observed = {"shape": tuple(images.shape), "grad": torch.is_grad_enabled(),
                                 "train": self.training, "autocast": torch.is_autocast_enabled("cpu")}
                return self.layer(images[:, :, 0, 0])

        model = Probe().train()
        # Expanded view has the real requested shape without a 384-MiB CPU allocation.
        images = torch.ones(1, 3, 1, 1).expand(32, 3, 1024, 1024)
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                result = engine.smoke_evaluation_forward(model, images, task="retrieval")
        except AttributeError as error:
            self.fail("GPU smoke needs a separately configured original FP32 evaluation pass: " + str(error))
        self.assertEqual(model.observed, {"shape": (32, 3, 1024, 1024), "grad": False,
                                         "train": False, "autocast": False})
        self.assertTrue(model.training)
        self.assertEqual(result["evaluation_precision"], "float32")
        self.assertEqual(result["evaluation_output_shape"], [32, 128])

    def test_evaluation_smoke_does_not_silently_accept_reduced_batch(self):
        try:
            function = engine.smoke_evaluation_forward
        except AttributeError as error:
            self.fail("Missing evaluation smoke contract: " + str(error))
        with self.assertRaises(ValueError):
            function(torch.nn.Identity(), torch.ones(1, 3, 1, 1).expand(16, 3, 1024, 1024),
                     task="retrieval")

    def test_actual_networks_have_finite_applied_cpu_steps(self):
        for phase in ("verification", "retrieval_phase1", "retrieval_phase2"):
            with self.subTest(phase=phase):
                torch.manual_seed(3)
                task = "verification" if phase == "verification" else "retrieval"
                model = create_model(task).train()
                if phase == "retrieval_phase1":
                    for parameter in model[0].parameters():
                        parameter.requires_grad_(False)
                optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001)
                scaler = torch.amp.GradScaler("cpu", enabled=False)
                images = torch.randn(2, 3, 64, 64)
                memory = CrossBatchContrastive()
                before = (model.fc_end.weight if task == "verification"
                          else model[1].second_fc[-1].weight).detach().clone()

                def forward():
                    if task == "verification":
                        return torch.nn.functional.binary_cross_entropy_with_logits(
                            model(images, images.roll(1, dims=0)).flatten(), torch.tensor([0., 1.]))
                    return memory(model(images), torch.tensor([0, 0]))

                loss, result = optimizer_batch(model, optimizer, scaler, forward)
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(result["optimizer_applied"])
                after = model.fc_end.weight if task == "verification" else model[1].second_fc[-1].weight
                self.assertFalse(torch.equal(before, after))
                with tempfile.TemporaryDirectory() as directory:
                    weights = Path(directory) / "selected_weights.pth"
                    torch.save(model.state_dict(), weights)
                    restored = create_model(task)
                    restored.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
                    restored.eval()
                    model.eval()
                    with torch.no_grad():
                        expected = model(images, images.roll(1, dims=0)) if task == "verification" else model(images)
                        output = restored(images, images.roll(1, dims=0)) if task == "verification" else restored(images)
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    self.assertEqual(tuple(output.shape), (2, 1 if task == "verification" else 128))

    def test_checkpointing_preserves_bn_state_and_cpu_gradients(self):
        torch.manual_seed(9)
        reference = create_model("retrieval", activation_checkpointing=False).train()
        checkpointed = create_model("retrieval", activation_checkpointing=True).train()
        checkpointed.load_state_dict(reference.state_dict())
        images = torch.randn(2, 3, 64, 64)
        torch.manual_seed(15)
        result = reference(images)
        result.square().mean().backward()
        torch.manual_seed(15)
        actual = checkpointed(images)
        actual.square().mean().backward()
        torch.testing.assert_close(actual, result, rtol=0, atol=0)
        for (name, value), (_, expected) in zip(checkpointed.named_buffers(), reference.named_buffers()):
            torch.testing.assert_close(value, expected, rtol=0, atol=0,
                                       msg=lambda msg, name=name: name + ": " + msg)
        for (name, value), (_, expected) in zip(checkpointed.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(value.grad, expected.grad, rtol=1e-5, atol=1e-7,
                                       msg=lambda msg, name=name: name + ": " + msg)


if __name__ == "__main__":
    unittest.main()
