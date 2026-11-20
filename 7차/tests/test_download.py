"""Model initialization downloads must preserve existing files and verify bytes."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_reid.download import download_with_resume  # noqa: E402


class DownloadTests(unittest.TestCase):
    def test_verifies_local_file_hash_and_keeps_original_if_correct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "remote.bin"
            source.write_bytes(b"correct model bytes")
            destination = root / "cache" / "model.pth"
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()
            download_with_resume(source.as_uri(), destination, digest=checksum)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            download_with_resume(source.as_uri(), destination, digest=checksum)

    def test_wrong_hash_does_not_replace_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "remote.bin"
            source.write_bytes(b"bad bytes")
            destination = root / "cache" / "model.pth"
            with self.assertRaises(ValueError):
                download_with_resume(source.as_uri(), destination, digest="0" * 64)
            self.assertFalse(destination.exists())

    def test_existing_corrupt_cache_raises_without_deleting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "remote.bin"
            source.write_bytes(b"expected")
            destination = root / "cache" / "model.pth"
            destination.parent.mkdir()
            destination.write_bytes(b"do not overwrite")
            with self.assertRaises(ValueError):
                download_with_resume(source.as_uri(), destination,
                                     digest=hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(destination.read_bytes(), b"do not overwrite")
