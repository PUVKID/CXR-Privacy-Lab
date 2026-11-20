"""Check the exact SHA-256 of a downloaded pretrained checkpoint before use."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download_with_resume(url: str, destination: Path, *, digest: str) -> Path:
    """Fetch to E: cache and commit only if digest matches; never overwrite a bad cache.

    This implementation restarts interrupted transfers instead of reusing unchecked
    partial content. The public API name is retained for the supplied ImageNet loader.
    """
    target = Path(destination)
    if len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest.lower()):
        raise ValueError("A valid expected SHA-256 is mandatory")
    if target.exists():
        if _sha256(target) != digest.lower():
            raise ValueError(f"An existing model cache has wrong SHA-256; never overwrite it: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=target.name + ".", suffix=".tmp",
                                         dir=target.parent, delete=False) as file:
            temporary = Path(file.name)
            with urlopen(url, timeout=120) as response:
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    file.write(block)
        if _sha256(temporary) != digest.lower():
            raise ValueError(f"Downloaded ImageNet file hash mismatch: {url}")
        if target.exists():
            raise FileExistsError(f"Concurrent model cache change detected: {target}")
        os.replace(temporary, target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
