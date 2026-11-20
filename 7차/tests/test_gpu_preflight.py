"""Regression guard for the observed PowerShell native-command exit-code bug.

On Windows PowerShell 5.1, piping `nvidia-smi` into `Select-Object -First 1`
caused a valid GPU CSV line to be returned with native exit code -1.
The user's 2026-11-18 reproduction is the behavioral failing case; this
portable source-level check prevents that unsafe native pipeline returning.
"""

from pathlib import Path
import re
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "All-In-One.ps1"


class GPUProbeRegressionTests(unittest.TestCase):
    def test_native_gpu_query_is_not_short_circuited_by_select_first(self):
        source = RUNNER.read_text(encoding="utf-8-sig")
        match = re.search(
            r"(?ms)^function Assert-GpuAvailable\s*\{(?P<body>.*?)^\}", source
        )
        self.assertIsNotNone(match, "The runner has no GPU availability check")
        body = match.group("body")
        self.assertIn("& nvidia-smi --query-gpu=", body)
        self.assertNotRegex(
            body,
            r"(?m)&\s*nvidia-smi[^\n]*\|\s*Select-Object\s+-First\s+1",
            "The native GPU process must finish before filtering its CSV lines",
        )


if __name__ == "__main__":
    unittest.main()
