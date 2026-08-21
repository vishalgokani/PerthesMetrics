from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inference.run_inference import executable, resolve_device


class InferenceRuntimeTests(unittest.TestCase):
    def test_current_python_environment_wins_over_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            expected = prefix / "Scripts" / "nnUNetv2_predict.exe"
            expected.parent.mkdir()
            expected.touch()
            with (
                patch("inference.run_inference.sys.prefix", str(prefix)),
                patch("inference.run_inference.shutil.which", return_value=r"C:\cpu_env\nnUNetv2_predict.exe"),
            ):
                self.assertEqual(executable("nnUNetv2_predict"), str(expected))

    def test_auto_device_falls_back_to_cpu_without_cuda(self) -> None:
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict("sys.modules", {"torch": torch}):
            self.assertEqual(resolve_device("auto", None), "cpu")

    def test_explicit_cuda_does_not_silently_fall_back(self) -> None:
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict("sys.modules", {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "--device cuda was requested"):
                resolve_device("cuda", None)


if __name__ == "__main__":
    unittest.main()
