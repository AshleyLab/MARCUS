"""
Smoke tests for ECG XML parsing and PNG conversion.

All tests are self-contained and require only the ``[preprocessing]`` extra.
No GPU, server, or external data is needed.
"""
from __future__ import annotations

import base64
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# XML fixture builders
# ---------------------------------------------------------------------------

STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _encode_int16_b64(arr: np.ndarray) -> str:
    """Encode a float32 array as base64-packed little-endian int16 bytes."""
    int16 = arr.astype(np.int16)
    raw = struct.pack(f"<{len(int16)}h", *int16.tolist())
    return base64.b64encode(raw).decode("ascii")


def _make_ge_muse_xml(n_leads: int = 12, n_samples: int = 5000, sample_rate: int = 500) -> str:
    """Generate a minimal GE Muse-like XML string for testing.

    The XML contains ``n_leads`` leads (taken from the standard 12-lead set).
    Each lead's waveform is a synthetic sine wave encoded as base64 int16 data.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0, n_samples / sample_rate, n_samples)

    lead_blocks = []
    for i in range(n_leads):
        lead_name = STANDARD_LEADS[i]
        # Scale-factor: 0.005 mV/count → multiply signal by 1/0.005 = 200 to get counts
        amplitude_counts = (np.sin(2 * np.pi * 1.5 * t + i * 0.3) * 200).astype(np.float32)
        b64 = _encode_int16_b64(amplitude_counts)
        lead_blocks.append(
            f"""    <LeadData>
      <LeadID>{lead_name}</LeadID>
      <LeadAmplitudeUnitsPerBit>0.005</LeadAmplitudeUnitsPerBit>
      <WaveFormData>{b64}</WaveFormData>
    </LeadData>"""
        )

    leads_xml = "\n".join(lead_blocks)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<RestingECG>
  <Waveform>
    <SampleBase>{sample_rate}</SampleBase>
{leads_xml}
  </Waveform>
</RestingECG>"""


def _make_philips_xml(n_leads: int = 12, n_samples: int = 5000, sample_rate: int = 500) -> str:
    """Generate a minimal Philips PageWriter-like XML string for testing."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, n_samples / sample_rate, n_samples)

    signal_blocks = []
    for i in range(n_leads):
        lead_name = STANDARD_LEADS[i]
        values = (np.sin(2 * np.pi * 1.5 * t + i * 0.3) * 0.5).astype(np.float32)
        data_str = " ".join(f"{v:.4f}" for v in values)
        signal_blocks.append(
            f"""    <Signal>
      <Name>{lead_name}</Name>
      <SampleRate>{sample_rate}</SampleRate>
      <Data>{data_str}</Data>
    </Signal>"""
        )

    signals_xml = "\n".join(signal_blocks)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Ecg>
  <Signals>
{signals_xml}
  </Signals>
</Ecg>"""


def _write_tmp_xml(xml_str: str, suffix: str = ".xml") -> Path:
    """Write XML string to a named temporary file and return its path."""
    import tempfile

    tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8")
    tf.write(xml_str)
    tf.close()
    return Path(tf.name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEcgXmlParser(unittest.TestCase):
    """Unit tests for XML parsing logic (no PNG rendering needed)."""

    def setUp(self):
        try:
            from video_chat_ui.preprocessing.ecg_xml import (
                parse_ge_muse_xml,
                parse_philips_xml,
                parse_ecg_xml,
                xml_to_png,
                xml_to_png_from_bytes,
            )
            self.parse_ge_muse_xml = parse_ge_muse_xml
            self.parse_philips_xml = parse_philips_xml
            self.parse_ecg_xml = parse_ecg_xml
            self.xml_to_png = xml_to_png
            self.xml_to_png_from_bytes = xml_to_png_from_bytes
        except ImportError as e:
            self.skipTest(f"preprocessing extra not installed: {e}")

    # --- GE Muse parsing ---

    def test_ge_muse_xml_parses_to_12_leads(self):
        """GE Muse XML with 12 leads returns a (12, N) array."""
        xml_str = _make_ge_muse_xml(n_leads=12, n_samples=5000)
        tree = ET.ElementTree(ET.fromstring(xml_str))
        signal, sr = self.parse_ge_muse_xml(tree)
        self.assertEqual(signal.ndim, 2)
        self.assertEqual(signal.shape[0], 12)
        self.assertGreater(signal.shape[1], 0)

    def test_ge_muse_xml_sample_rate_detected(self):
        """Sample rate is correctly extracted from GE Muse XML <SampleBase>."""
        xml_str = _make_ge_muse_xml(n_leads=12, n_samples=5000, sample_rate=250)
        tree = ET.ElementTree(ET.fromstring(xml_str))
        _, sr = self.parse_ge_muse_xml(tree)
        self.assertEqual(sr, 250)

    def test_ge_muse_xml_signal_values_scaled(self):
        """GE Muse: WaveFormData int16 values are multiplied by LeadAmplitudeUnitsPerBit."""
        # Our fixture uses scale 0.005; raw int16 values up to ±200 → ±1.0 mV
        xml_str = _make_ge_muse_xml(n_leads=12, n_samples=5000, sample_rate=500)
        tree = ET.ElementTree(ET.fromstring(xml_str))
        signal, _ = self.parse_ge_muse_xml(tree)
        # Maximum absolute value should be approximately 1.0 mV
        max_abs = float(np.nanmax(np.abs(signal)))
        self.assertAlmostEqual(max_abs, 1.0, delta=0.1)

    def test_ge_muse_xml_partial_leads_filled_with_zeros(self):
        """GE Muse XML with fewer than 12 leads fills missing leads with zeros."""
        xml_str = _make_ge_muse_xml(n_leads=6, n_samples=5000)
        tree = ET.ElementTree(ET.fromstring(xml_str))
        signal, _ = self.parse_ge_muse_xml(tree)
        self.assertEqual(signal.shape[0], 12)
        # Last 6 leads (V1–V6) should be all zeros
        np.testing.assert_array_equal(signal[6:], 0.0)

    # --- Philips parsing ---

    def test_philips_xml_parses_to_12_leads(self):
        """Philips XML with 12 leads returns a (12, N) array."""
        xml_str = _make_philips_xml(n_leads=12, n_samples=5000)
        tree = ET.ElementTree(ET.fromstring(xml_str))
        signal, sr = self.parse_philips_xml(tree)
        self.assertEqual(signal.ndim, 2)
        self.assertEqual(signal.shape[0], 12)
        self.assertGreater(signal.shape[1], 0)

    def test_philips_xml_sample_rate_detected(self):
        """Sample rate is correctly extracted from Philips XML."""
        xml_str = _make_philips_xml(n_leads=12, n_samples=5000, sample_rate=250)
        tree = ET.ElementTree(ET.fromstring(xml_str))
        _, sr = self.parse_philips_xml(tree)
        self.assertEqual(sr, 250)

    # --- Auto-detection ---

    def test_auto_detect_ge_muse(self):
        """parse_ecg_xml auto-detects GE Muse format from file."""
        xml_str = _make_ge_muse_xml(n_leads=12, n_samples=5000)
        tmp = _write_tmp_xml(xml_str)
        try:
            signal, sr = self.parse_ecg_xml(tmp)
            self.assertEqual(signal.shape[0], 12)
            self.assertEqual(sr, 500)
        finally:
            tmp.unlink(missing_ok=True)

    def test_auto_detect_philips(self):
        """parse_ecg_xml auto-detects Philips format from file."""
        xml_str = _make_philips_xml(n_leads=12, n_samples=5000)
        tmp = _write_tmp_xml(xml_str)
        try:
            signal, sr = self.parse_ecg_xml(tmp)
            self.assertEqual(signal.shape[0], 12)
        finally:
            tmp.unlink(missing_ok=True)

    def test_unrecognised_xml_raises_value_error(self):
        """Completely foreign XML should raise ValueError."""
        xml_str = '<foo><bar>hello</bar></foo>'
        tmp = _write_tmp_xml(xml_str)
        try:
            with self.assertRaises(ValueError):
                self.parse_ecg_xml(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    # --- Sample rate detection (end-to-end) ---

    def test_sample_rate_detection(self):
        """Sample rate is correctly extracted and returned from XML."""
        for expected_sr in (250, 500, 1000):
            with self.subTest(sample_rate=expected_sr):
                xml_str = _make_ge_muse_xml(n_leads=12, n_samples=2000, sample_rate=expected_sr)
                tmp = _write_tmp_xml(xml_str)
                try:
                    _, sr = self.parse_ecg_xml(tmp)
                    self.assertEqual(sr, expected_sr)
                finally:
                    tmp.unlink(missing_ok=True)


class TestEcgXmlToPng(unittest.TestCase):
    """Integration tests for xml_to_png / xml_to_png_from_bytes."""

    def setUp(self):
        try:
            from video_chat_ui.preprocessing.ecg_xml import xml_to_png, xml_to_png_from_bytes
            self.xml_to_png = xml_to_png
            self.xml_to_png_from_bytes = xml_to_png_from_bytes
        except ImportError as e:
            self.skipTest(f"preprocessing extra not installed: {e}")
        try:
            import PIL  # noqa: F401
            import matplotlib  # noqa: F401
        except ImportError as e:
            self.skipTest(f"PIL/matplotlib not installed: {e}")

    def test_xml_to_png_creates_file(self):
        """xml_to_png creates a non-empty PNG file from GE Muse XML."""
        xml_str = _make_ge_muse_xml(n_leads=12, n_samples=5000)
        tmp_xml = _write_tmp_xml(xml_str)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "ecg.png"
            result = self.xml_to_png(tmp_xml, out)
            self.assertTrue(out.is_file(), "Output PNG file was not created")
            self.assertGreater(out.stat().st_size, 1000, "Output PNG is suspiciously small")
            self.assertEqual(result, out)
        tmp_xml.unlink(missing_ok=True)

    def test_xml_to_png_from_bytes(self):
        """xml_to_png_from_bytes works with raw bytes input."""
        xml_str = _make_ge_muse_xml(n_leads=12, n_samples=5000)
        xml_bytes = xml_str.encode("utf-8")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "ecg_from_bytes.png"
            result = self.xml_to_png_from_bytes(xml_bytes, out)
            self.assertTrue(out.is_file(), "Output PNG file was not created from bytes")
            self.assertGreater(out.stat().st_size, 1000)

    def test_xml_to_png_sample_rate_override(self):
        """xml_to_png respects a manually specified sample rate."""
        xml_str = _make_ge_muse_xml(n_leads=12, n_samples=2500, sample_rate=250)
        tmp_xml = _write_tmp_xml(xml_str)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "ecg_sr_override.png"
            # Override with 500 Hz — should not raise
            result = self.xml_to_png(tmp_xml, out, sample_rate=500)
            self.assertTrue(out.is_file())
        tmp_xml.unlink(missing_ok=True)

    def test_xml_to_png_philips_format(self):
        """xml_to_png also works with Philips-format XML."""
        xml_str = _make_philips_xml(n_leads=12, n_samples=5000)
        tmp_xml = _write_tmp_xml(xml_str)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "ecg_philips.png"
            result = self.xml_to_png(tmp_xml, out)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 1000)
        tmp_xml.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
