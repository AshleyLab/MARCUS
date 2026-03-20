"""Smoke: 12 x N .npy -> PNG (requires [preprocessing])."""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestEcgNpyToPng(unittest.TestCase):
    def test_npy_to_png_writes_file(self) -> None:
        try:
            from video_chat_ui.preprocessing.ecg import npy_to_png
        except ImportError as e:
            self.skipTest(f"preprocessing extra not installed: {e}")

        arr = np.random.randn(12, 100).astype(np.float32) * 0.5
        buf = io.BytesIO()
        np.save(buf, arr)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "ecg.png"
            npy_to_png(buf.getvalue(), out)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
