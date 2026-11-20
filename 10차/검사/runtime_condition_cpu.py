"""실제 PyTorch/Pillow 실행환경에서 합성 영상만 사용하는 CPU 경로 검사입니다."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image, ImageFilter
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cxr10_contract import condition, condition_plan
from cxr_reid.inference import XrayDataset
from training.data import PairDataset, RetrievalDataset, transform_source_image
from training import engine


def independent_transform(image, mode, strength):
    if strength == 0 or (mode == 'mosaic' and strength == 1):
        return image.copy()
    if mode == 'blur':
        return image.filter(ImageFilter.GaussianBlur(strength))
    width, height = image.size
    return image.resize((max(1, (width + strength - 1) // strength),
        max(1, (height + strength - 1) // strength)), Image.Resampling.BOX).resize(image.size, Image.Resampling.NEAREST)


def normalized(image):
    value = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    return (value - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


class RuntimeConditionTests(unittest.TestCase):
    def test_all_twenty_operators_match_pillow_and_preserve_source(self):
        pixels = np.arange(19 * 17 * 3, dtype=np.uint8).reshape(17, 19, 3)
        image = Image.fromarray(pixels)
        original = image.tobytes()
        for row in condition_plan():
            with self.subTest(condition=row['id']):
                actual = transform_source_image(image, row['id'])
                expected = independent_transform(image, row['mode'], row['strength'])
                self.assertEqual(actual.tobytes(), expected.tobytes())
                self.assertEqual(actual.size, (19, 17))
                self.assertIsNot(actual, image)
                self.assertEqual(image.tobytes(), original)

    def test_both_pair_sides_and_inference_apply_same_transform_before_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.indices((1024, 1024)).sum(axis=0).astype(np.uint8)
            rgb = np.stack((pixels, pixels ^ 127, 255 - pixels), axis=2)
            first, second = Image.fromarray(rgb), Image.fromarray(rgb[::-1].copy())
            first.save(root / 'first.png'); second.save(root / 'second.png')
            for condition_id in ('blur_000', 'blur_004', 'mosaic_000', 'mosaic_001', 'mosaic_008'):
                row = condition(condition_id)
                pair = PairDataset(root, [('first.png', 'second.png', 1)], condition_id=condition_id)
                left, right, target = pair[0]
                for actual, image in ((left, first), (right, second)):
                    expected = independent_transform(image, row['mode'], row['strength']).resize((256, 256), Image.Resampling.BILINEAR)
                    torch.testing.assert_close(actual, normalized(expected), rtol=0, atol=1e-6)
                inference = XrayDataset(root, ['first.png'], row['operator_mode'], row['operator_strength'], 256)
                torch.testing.assert_close(left, inference[0], rtol=0, atol=1e-6)
                self.assertEqual(float(target), 1.)

    def test_retrieval_transform_matches_inference_at_original1024(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.indices((1024, 1024)).sum(axis=0).astype(np.uint8)
            image = Image.fromarray(np.stack((pixels, 255 - pixels, pixels ^ 31), axis=2))
            image.save(root / 'sample.png')
            for condition_id in ('blur_002', 'mosaic_000', 'mosaic_016'):
                row = condition(condition_id)
                dataset = RetrievalDataset(root, [('sample.png', 1)], condition_id=condition_id)
                actual, label = dataset[0]
                expected = normalized(independent_transform(image, row['mode'], row['strength']))
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
                inference = XrayDataset(root, ['sample.png'], row['operator_mode'], row['operator_strength'], 1024)
                torch.testing.assert_close(actual, inference[0], rtol=0, atol=1e-6)
                self.assertEqual(label, 1)
            seen = []
            dataset = RetrievalDataset(root, [('sample.png', 1)], condition_id='blur_004', augment=True)
            dataset.transform = lambda value: seen.append(value.tobytes()) or torch.zeros(1)
            dataset[0]
            self.assertEqual(seen, [image.filter(ImageFilter.GaussianBlur(4)).tobytes()])

    def test_real_engine_binding_contains_transform_sources_and_changes_with_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / 'official'
            for relative in ('PatientVerification/train_val_list.txt', 'PatientVerification/image_pairs/pairs_training.txt',
                             'PatientRetrieval/csv_files/train.csv'):
                path = official / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic\n', encoding='utf-8')
            args = SimpleNamespace(condition_id='blur_004', output=root / 'work/runs/11111111-1111-4111-8111-111111111111/training',
                seed=3, tasks='both', device='cpu', verification_workers=0, retrieval_workers=0, activation_checkpointing=True)
            original = copy.deepcopy(engine.PROTOCOL)
            baseline, details = engine.binding(args, official)
            for name in ('cxr10_contract.py', 'cxr10_evaluate.py', 'training/data.py', 'cxr_reid/image_ops.py'):
                self.assertEqual(details['sources'][name], hashlib.sha256((ROOT / 'src' / name).read_bytes()).hexdigest())
            try:
                for key, value in [('verification_lr', .01), ('verification_epochs', 99), ('verification_resolution', 128)]:
                    engine.PROTOCOL[key] = value
                    self.assertNotEqual(baseline, engine.binding(args, official)[0])
                    engine.PROTOCOL.clear(); engine.PROTOCOL.update(copy.deepcopy(original))
                (official / 'PatientRetrieval/csv_files/train.csv').write_text('changed\n', encoding='utf-8')
                self.assertNotEqual(baseline, engine.binding(args, official)[0])
            finally:
                engine.PROTOCOL.clear(); engine.PROTOCOL.update(original)


if __name__ == '__main__':
    print(json.dumps({'torch': torch.__version__, 'device': 'cpu', 'medical_images_used': False,
        'cuda_tensors_created': False}, ensure_ascii=False), flush=True)
    unittest.main()
