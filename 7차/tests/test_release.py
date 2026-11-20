"""The distributable CMD must carry a verified complete payload, not download code."""
import base64
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def builder(self):
        path = ROOT / "packaging/build_release.py"
        self.assertTrue(path.exists(), "Self-contained CMD builder is missing")
        spec = importlib.util.spec_from_file_location("cxr7_release_builder", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self, base):
        root = base / "package"
        (root / "packaging").mkdir(parents=True)
        (root / "src").mkdir()
        (root / "src/code.py").write_text("print('fixed code')\n", encoding="utf-8")
        (root / "Start-CXR7.cmd").write_text("@echo off\n", encoding="ascii")
        (root / "All-In-One.ps1").write_text("Write-Output ready\n", encoding="utf-8")
        (root / "packaging/Install-Payload.ps1").write_bytes((ROOT / "packaging/Install-Payload.ps1").read_bytes())
        (root / "__pycache__").mkdir()
        (root / "__pycache__/wrong.pyc").write_bytes(b"not source")
        return root

    def assert_complete_payload(self, source_bytes=None):
        module = self.builder()
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp))
            if source_bytes is not None:
                (root / "src/code.py").write_bytes(source_bytes)
            expected_source_bytes = (root / "src/code.py").read_bytes()
            out = Path(tmp) / "out"
            files = module.build(root, out)
            payload = module.inspect_self_extracting_cmd(files["cmd"])
            self.assertEqual(payload, files["zip"].read_bytes())
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                self.assertEqual(archive.read("src/code.py"), expected_source_bytes)
                self.assertIn("package_manifest.json", archive.namelist())
                self.assertFalse(any("__pycache__" in name for name in archive.namelist()))
            self.assertLess(max(map(len, files["cmd"].read_text(encoding="utf-8").splitlines()[:12])), 8191)

    def test_cmd_contains_complete_checksum_verified_payload(self):
        self.assert_complete_payload()

    def test_cmd_preserves_explicit_lf_source_bytes(self):
        self.assert_complete_payload(b"print('fixed code')\n")

    def test_cmd_preserves_explicit_windows_crlf_source_bytes(self):
        self.assert_complete_payload(b"print('fixed code')\r\n")

    def test_corrupt_embedded_payload_is_rejected(self):
        module = self.builder()
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp))
            files = module.build(root, Path(tmp) / "out")
            text = files["cmd"].read_text(encoding="utf-8")
            marker = "\n__CXR7_ZIP_BEGIN__\n"
            prefix, tail = text.split(marker, 1)
            char = "A" if tail[0] != "A" else "B"
            files["cmd"].write_text(prefix + marker + char + tail[1:], encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA|hash"):
                module.inspect_self_extracting_cmd(files["cmd"])


if __name__ == "__main__":
    unittest.main()
