"""Representative examples must use one source image per dataset, never select by score."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


class ExampleTests(unittest.TestCase):
    def module(self):
        self.assertTrue((SRC / "cxr7_examples.py").is_file(), "Representative exporter is missing")
        import cxr7_examples
        return cxr7_examples

    def fixture(self, base):
        project, official, out = base / "data_project", base / "official", base / "new_run"
        for ds in ("NIH_ChestXray14", "COVID19"):
            root = project / "data" / ds / "images"
            root.mkdir(parents=True)
            arr = np.indices((16, 16)).sum(axis=0) % 2 * 255
            Image.fromarray(arr.astype(np.uint8)).convert("RGB").save(root / "a.png")
            Image.new("RGB", (16, 16), "blue").save(root / "b.png")
        vr = official / "PatientVerification"
        rr = official / "PatientRetrieval" / "csv_files"
        vr.mkdir(parents=True)
        rr.mkdir(parents=True)
        (vr / "test_list.txt").write_text("b.png\na.png\n", encoding="utf-8")
        (rr / "test.csv").write_text("Image Index,Patient ID\nb.png,1\na.png,1\n", encoding="utf-8")
        (rr / "covid_data.csv").write_text("Image Index,Patient ID\nb.png,9\na.png,9\n", encoding="utf-8")
        out.mkdir()
        return project, official, out

    def test_all_strengths_keep_one_source_and_leave_inputs_unchanged(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            project, official, out = self.fixture(Path(tmp))
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in project.rglob("*.png")}
            for mode, level in [("original", 0), ("blur", 2), ("mosaic", 8)]:
                result = m.save_condition_example(project, official, out, mode, level)
                self.assertEqual(set(result), {"NIH", "COVID19"})
                for item in result.values():
                    self.assertEqual(item["source_filename"], "a.png")
                    self.assertTrue((out / item["image_path"]).is_file())
            after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in project.rglob("*.png")}
            self.assertEqual(before, after)
            self.assertEqual(len(list((out / "대표사진").rglob("*.png"))), 6)

    def test_zero_modes_are_pixel_identical_and_repeated_export_idempotent(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            project, official, out = self.fixture(Path(tmp))
            first = m.save_condition_example(project, official, out, "original", 0)
            for mode in ("blur", "mosaic"):
                zero = m.save_condition_example(project, official, out, mode, 0)
                with Image.open(out / first["NIH"]["image_path"]) as a, Image.open(out / zero["NIH"]["image_path"]) as b:
                    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            again = m.save_condition_example(project, official, out, "original", 0)
            self.assertEqual(first, again)

    def test_modified_source_is_not_silently_replaced(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            project, official, out = self.fixture(Path(tmp))
            m.save_condition_example(project, official, out, "original", 0)
            Image.new("RGB", (16, 16), "red").save(project / "data/NIH_ChestXray14/images/a.png")
            with self.assertRaisesRegex(ValueError, "source|원본"):
                m.save_condition_example(project, official, out, "blur", 1)

    def test_rejects_strength_outside_declared_conditions_before_writing(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            project, official, out = self.fixture(Path(tmp))
            for mode, level in [("blur", -1), ("mosaic", 3), ("blur", 1.5), ("../../bad", 2)]:
                with self.assertRaises(ValueError):
                    m.save_condition_example(project, official, out, mode, level)
            self.assertEqual(list(out.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
