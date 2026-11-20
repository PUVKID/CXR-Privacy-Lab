"""The 7th-run guard and original-manifest checks must work without torch/GPU."""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cxr7_contract import (  # noqa: E402
    assert_seventh_workspace,
    assert_training_scope,
    count_csv_records,
    count_pair_records,
    prepare_new_run,
)


class SeventhRunSafetyTests(unittest.TestCase):
    def test_accepts_only_E_drive_seventh_directory(self):
        assert_seventh_workspace(r"E:\2026\흉부 엑스레이 환자 재식별 실험\7차")
        for bad in (r"C:\runs\7차", r"E:\2026\흉부 엑스레이 환자 재식별 실험\5차",
                    r"E:\2026\흉부 엑스레이 환자 재식별 실험\7차\..\5차"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                assert_seventh_workspace(bad)

    def test_no_existing_run_can_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "runs"
            run = prepare_new_run(base, "uuid-1")
            self.assertTrue(run.is_dir())
            with self.assertRaises(FileExistsError):
                prepare_new_run(base, "uuid-1")

    def test_training_only_writes_under_7_runs_and_uses_two_datasets(self):
        seven = r"E:\2026\흉부 엑스레이 환자 재식별 실험\7차"
        data = r"E:\2026\흉부 엑스레이 환자 재식별 실험\연구 자료"
        assert_training_scope(seven, seven + r"\runs\uuid-1", data)
        for output, project in ((r"E:\2026\흉부 엑스레이 환자 재식별 실험\6차\runs\uuid-1", data),
                                (seven + r"\runs\..\..\5차\runs\uuid-1", data),
                                (seven + r"\runs\uuid-1", r"C:\data")):
            with self.subTest(output=output, project=project), self.assertRaises(ValueError):
                assert_training_scope(seven, output, project)

    def test_never_allows_unsafe_run_name(self):
        with tempfile.TemporaryDirectory() as folder:
            for bad in ("../6차", "", ".", "../../5차", "x/y", "x\\y"):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    prepare_new_run(Path(folder), bad)


class ManifestTests(unittest.TestCase):
    def test_counts_only_nonempty_triples_and_rejects_malformed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pairs.txt"
            path.write_text("a.png b.png 1.0\n\nc.png d.png 0.0\n", encoding="utf-8")
            self.assertEqual(count_pair_records(path), 2)
            path.write_text("a.png b.png unknown\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                count_pair_records(path)

    def test_reads_author_csv_with_unnamed_index_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            path.write_text(",Image Index,Patient ID\n0,image1.png,1\n1,image2.png,2\n", encoding="utf-8")
            self.assertEqual(count_csv_records(path), 2)
            path.write_text(",Image Index,Patient ID\n0,image1.png,\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                count_csv_records(path)


if __name__ == "__main__":
    unittest.main()
