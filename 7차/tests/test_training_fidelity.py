"""Regression tests for the Windows port's training mathematics and state.

Reference algorithms are executed, not checked by grepping source text.
"""

import ast
import contextlib
import io
from pathlib import Path
import random
import sys
import tempfile
import unittest
import warnings

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training._vendor.pytorch_metric_learning.losses import CrossBatchMemory, ContrastiveLoss

from cxr_reid.models import VerificationNetwork, ExtractorRes50, RetrievalHead
from training.data import RetrievalDataset, mining_order
from training.engine import create_model, seed_all
from training.losses import CrossBatchContrastive


class OriginalLossTests(unittest.TestCase):
    def test_eval_loss_enqueues_validation_like_author(self):
        loss = CrossBatchContrastive(embedding_size=2, memory_size=8).eval()
        loss(torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.1, .9]]),
             torch.tensor([0, 0, 1, 1]))
        self.assertEqual(int(loss.count), 4,
                         "Original CBM also enqueues validation embeddings in eval mode")

    def test_loss_and_gradient_match_pinned_reference(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                current = CrossBatchContrastive(embedding_size=3, memory_size=8)
                reference = CrossBatchMemory(ContrastiveLoss(), 3, memory_size=8)
                for step in range(4):
                    raw = torch.tensor([[1., .2, .1], [.8, .4, .2], [.1, 1., .3], [.2, .8, .5]], dtype=dtype)
                    raw = raw + .03 * step
                    a, b = raw.clone().requires_grad_(), raw.clone().requires_grad_()
                    labels = torch.tensor([0, 0, 1, 1])
                    actual, expected = current(a, labels), reference(b, labels)
                    actual.backward()
                    expected.backward()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_checkpoint_roundtrip_preserves_wrapped_queue(self):
        first = CrossBatchContrastive(embedding_size=2, memory_size=8)
        labels = torch.tensor([0, 0, 1, 1])
        for k in range(3):
            first(torch.tensor([[1., k*.1], [.8, .2], [0., 1.], [.2, .8]]), labels)
        second = CrossBatchContrastive(embedding_size=2, memory_size=8)
        second.load_state_dict(first.state_dict())
        a = torch.tensor([[.3, .7], [.2, .8], [.8, .2], [.9, .1]], requires_grad=True)
        b = a.detach().clone().requires_grad_()
        left, right = first(a, labels), second(b, labels)
        left.backward()
        right.backward()
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


class OriginalInitializationTests(unittest.TestCase):
    def test_phase_backend_policy_matches_author_and_disables_new_tf32(self):
        try:
            seed_all(3, phase="verification")
        except TypeError as error:
            self.fail("Verification and retrieval have different original cuDNN policies: " + str(error))
        self.assertTrue(torch.backends.cudnn.benchmark)
        self.assertFalse(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        seed_all(3, phase="retrieval_phase1")
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)

    def test_verification_head_uses_single_backbone_rng_sequence(self):
        torch.manual_seed(3)
        reference = VerificationNetwork()
        expected = {key: value.clone() for key, value in reference.state_dict().items()}
        del reference
        torch.manual_seed(3)
        actual = create_model("verification")
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0,
                                       msg=lambda msg, key=key: key + ": " + msg)

    def test_retrieval_head_uses_single_backbone_rng_sequence(self):
        torch.manual_seed(3)
        reference = torch.nn.Sequential(ExtractorRes50(), RetrievalHead())
        expected = {key: value.clone() for key, value in reference.state_dict().items()}
        del reference
        torch.manual_seed(3)
        actual = create_model("retrieval")
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0,
                                       msg=lambda msg, key=key: key + ": " + msg)


class OriginalDataTests(unittest.TestCase):
    def test_augmentation_draws_new_randomness_per_visit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (1024, 1024), (100, 100, 100)).save(root / "a.png")
            dataset = RetrievalDataset(root, [("a.png", "1")], augment=True, seed=3)
            torch.manual_seed(3)
            first, _ = dataset[0]
            second, _ = dataset[0]
            self.assertFalse(torch.equal(first, second),
                             "Author ColorJitter consumes worker RNG rather than fixing every image")

    def test_sampler_matches_executed_author_functions(self):
        source = ROOT / "official_source/PatientRetrieval/paper_retrieval_training_phase1.py"
        parsed = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"sample_func", "get_block", "build_mining_list_32"}]
        namespace = {"np": np, "random": random}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        counts = [2, 3, 5, 7, 8, 9, 14, 15, 16, 17, 20, 23, 24]
        rows = [(f"{pid:08d}_{i:03d}.png", pid) for pid, count in enumerate(counts) for i in range(count)]
        for seed in (0, 3, 17, 123):
            with self.subTest(seed=seed):
                random.seed(seed)
                np.random.seed(seed)
                names = np.arange(len(rows))
                labels = np.asarray([label for _, label in rows])
                with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                    warnings.simplefilter("ignore", DeprecationWarning)
                    expected = namespace["build_mining_list_32"](names, labels, 8)
                expected_order = [int(name) for name, _ in expected]
                self.assertEqual(mining_order(rows, seed=seed), expected_order)


if __name__ == "__main__":
    unittest.main()
