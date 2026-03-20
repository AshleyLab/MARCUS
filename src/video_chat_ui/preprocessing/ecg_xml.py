"""
ECG XML to Hospital-Style PNG Converter

Reads 12-lead ECG data from institutional XML files (GE MuseXML / Philips XML format)
and converts to hospital-style PNG grid images compatible with the MARCUS ECG model.

Supports two common formats:
1. GE Muse XML: <RestingECG><Waveform><LeadData>...</LeadData></Waveform></RestingECG>
2. Philips XML: <ClinicalDocument><component><series>...</series></component></ClinicalDocument>

Usage:
    from video_chat_ui.preprocessing.ecg_xml import xml_to_png

    png_path = xml_to_png("ecg.xml", "output.png")
    # Or from bytes
    png_path = xml_to_png_from_bytes(xml_bytes, "output.png")
"""
from __future__ import annotations

import base64
import io
import struct
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

# Standard 12-lead ECG lead names (canonical order)
STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# GE Muse uses these exact strings; map common aliases to canonical names
_GE_LEAD_ALIAS: dict[str, str] = {
    "I": "I",
    "II": "II",
    "III": "III",
    "AVR": "aVR",
    "AVL": "aVL",
    "AVF": "aVF",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "V4": "V4",
    "V5": "V5",
    "V6": "V6",
}


def _decode_ge_waveform(b64_text: str, scale: float = 1.0) -> np.ndarray:
    """Decode a base64-encoded sequence of little-endian int16 samples.

    GE Muse stores WaveFormData as raw int16 LE bytes, base64-encoded.
    The optional ``scale`` converts ADC counts to mV (default 1.0 = pass through).
    """
    raw = base64.b64decode(b64_text.strip())
    n_samples = len(raw) // 2
    samples = struct.unpack_from(f"<{n_samples}h", raw)
    return np.array(samples, dtype=np.float32) * scale


def parse_ge_muse_xml(tree: ET.ElementTree) -> tuple[np.ndarray, int]:
    """Parse GE Muse/CardioSoft XML format.

    Returns ``(signal_array, sample_rate)`` where ``signal_array`` is ``(12, N)``.

    The expected structure is::

        <RestingECG>
          <Waveform>
            <SampleBase>500</SampleBase>           # sample rate (Hz)
            <LeadData>
              <LeadID>I</LeadID>
              <LeadAmplitudeUnitsPerBit>0.005</LeadAmplitudeUnitsPerBit>
              <WaveFormData>BASE64...</WaveFormData>
            </LeadData>
            ...
          </Waveform>
        </RestingECG>
    """
    root = tree.getroot()

    # Strip namespace if present (e.g. {urn:hl7-org:v3}RestingECG)
    def _tag(el: ET.Element) -> str:
        return el.tag.split("}")[-1] if "}" in el.tag else el.tag

    # Locate the Waveform element that contains LeadData children
    waveform_el: ET.Element | None = None
    for el in root.iter():
        if _tag(el) == "Waveform":
            # Prefer the one that actually contains lead data
            if any(_tag(c) == "LeadData" for c in el):
                waveform_el = el
                break

    if waveform_el is None:
        raise ValueError("GE Muse XML: no <Waveform> element with <LeadData> children found")

    # Sample rate
    sample_rate = 500  # common default
    for child in waveform_el:
        if _tag(child) in ("SampleBase", "SampleRate") and child.text:
            try:
                sample_rate = int(child.text.strip())
            except ValueError:
                pass
            break

    # Collect lead signals
    lead_signals: dict[str, np.ndarray] = {}
    for lead_el in waveform_el:
        if _tag(lead_el) != "LeadData":
            continue
        lead_id: str | None = None
        waveform_data: str | None = None
        scale = 1.0  # mV per ADC count; default pass-through
        for child in lead_el:
            t = _tag(child)
            if t == "LeadID" and child.text:
                lead_id = child.text.strip().upper()
            elif t == "WaveFormData" and child.text:
                waveform_data = child.text
            elif t in ("LeadAmplitudeUnitsPerBit", "ScaleFactor") and child.text:
                try:
                    scale = float(child.text.strip())
                except ValueError:
                    pass

        if lead_id is None or waveform_data is None:
            continue

        canonical = _GE_LEAD_ALIAS.get(lead_id)
        if canonical is None:
            continue  # unknown lead — skip

        try:
            signal = _decode_ge_waveform(waveform_data, scale=scale)
        except Exception:
            signal = np.zeros(0, dtype=np.float32)

        lead_signals[canonical] = signal

    return _assemble_12_lead(lead_signals), sample_rate


def parse_philips_xml(tree: ET.ElementTree) -> tuple[np.ndarray, int]:
    """Parse Philips IntelliVue / TraceMaster XML format.

    Returns ``(signal_array, sample_rate)``.

    Philips uses a CDA-like structure::

        <ClinicalDocument>
          <component>
            <series>
              <code code="MDC_ECG_LEAD_I" .../>
              <component>
                <sequence>
                  <value><digits>...</digits></value>
                </sequence>
              </component>
            </series>
          </component>
        </ClinicalDocument>

    Also handles the simpler Philips PageWriter format::

        <Ecg>
          <Signals>
            <Signal>
              <Name>Lead I</Name>
              <SampleRate>500</SampleRate>
              <Data>space-separated integers</Data>
            </Signal>
          </Signals>
        </Ecg>
    """
    root = tree.getroot()

    def _tag(el: ET.Element) -> str:
        return el.tag.split("}")[-1] if "}" in el.tag else el.tag

    # --- Strategy 1: Philips PageWriter / simple <Signal> structure ---
    lead_signals: dict[str, np.ndarray] = {}
    sample_rate = 500

    signals_el: ET.Element | None = None
    for el in root.iter():
        if _tag(el) == "Signals":
            signals_el = el
            break

    if signals_el is not None:
        for signal_el in signals_el:
            if _tag(signal_el) != "Signal":
                continue
            name: str | None = None
            data_text: str | None = None
            sr_text: str | None = None
            for child in signal_el:
                t = _tag(child)
                if t == "Name" and child.text:
                    name = child.text.strip()
                elif t in ("Data", "WaveformData") and child.text:
                    data_text = child.text.strip()
                elif t == "SampleRate" and child.text:
                    sr_text = child.text.strip()
            if sr_text:
                try:
                    sample_rate = int(sr_text)
                except ValueError:
                    pass
            if name is None or data_text is None:
                continue
            canonical = _resolve_lead_name(name)
            if canonical is None:
                continue
            try:
                # Try base64 first, fall back to space-separated integers
                try:
                    arr = _decode_ge_waveform(data_text)
                except Exception:
                    values = list(map(float, data_text.split()))
                    arr = np.array(values, dtype=np.float32)
                lead_signals[canonical] = arr
            except Exception:
                pass
        if lead_signals:
            return _assemble_12_lead(lead_signals), sample_rate

    # --- Strategy 2: CDA series structure ---
    # Philips CDA: each <series> has a <code> with displayName like "Lead I"
    # and nested <sequence><value><digits>...</digits></value></sequence>
    lead_signals = {}
    for series_el in root.iter():
        if _tag(series_el) != "series":
            continue
        lead_name: str | None = None
        for code_el in series_el:
            if _tag(code_el) == "code":
                display = code_el.get("displayName") or code_el.get("code") or ""
                lead_name = _resolve_lead_name(display)
                break
        if lead_name is None:
            continue
        for digits_el in series_el.iter():
            if _tag(digits_el) == "digits" and digits_el.text:
                try:
                    values = list(map(float, digits_el.text.split()))
                    lead_signals[lead_name] = np.array(values, dtype=np.float32)
                except ValueError:
                    pass
                break

    # Sample rate from Philips CDA frequencyQuantity
    for el in root.iter():
        if _tag(el) == "frequencyQuantity":
            val = el.get("value")
            if val:
                try:
                    sample_rate = int(float(val))
                except ValueError:
                    pass
            break

    return _assemble_12_lead(lead_signals), sample_rate


def _resolve_lead_name(raw: str) -> str | None:
    """Map a raw lead name string (from various vendor formats) to a canonical name."""
    s = raw.strip().upper()
    # Remove common prefixes
    for prefix in ("MDC_ECG_LEAD_", "LEAD ", "ECG "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    # Map augmented leads
    augmented_map = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}
    if s in augmented_map:
        return augmented_map[s]
    # Canonical check (case-insensitive match)
    for canonical in STANDARD_LEADS:
        if s == canonical.upper():
            return canonical
    return None


def _assemble_12_lead(lead_signals: dict[str, np.ndarray]) -> np.ndarray:
    """Assemble a ``(12, N)`` array from a dict of lead arrays.

    Missing leads are filled with zeros. All leads are trimmed/padded to the
    length of the longest lead present.

    Raises ``ValueError`` if no lead data is present at all.
    """
    if not lead_signals:
        raise ValueError("No recognizable ECG lead data found in XML")

    # Determine target length from the longest signal
    n = max(len(arr) for arr in lead_signals.values() if len(arr) > 0)
    if n == 0:
        n = 5000

    out = np.zeros((12, n), dtype=np.float32)
    for idx, lead_name in enumerate(STANDARD_LEADS):
        arr = lead_signals.get(lead_name)
        if arr is None or len(arr) == 0:
            continue  # leave as zeros
        # Trim or pad to length n
        if len(arr) >= n:
            out[idx] = arr[:n]
        else:
            out[idx, : len(arr)] = arr
    return out


def parse_ecg_xml(xml_path: str | Path) -> tuple[np.ndarray, int]:
    """Auto-detect and parse ECG XML format.

    Returns ``(signal_array, sample_rate)`` where ``signal_array`` is ``(12, N)``.
    Raises ``ValueError`` if the format is not recognized.

    Detection order:
    1. If the root tag contains "RestingECG" or a child Waveform with LeadData
       is present → GE Muse.
    2. If the root tag contains "ClinicalDocument", "Ecg", or "Signals"
       → Philips.
    3. Try GE Muse parser, then Philips parser; return whichever succeeds first.
    """
    xml_path = Path(xml_path)
    tree = ET.parse(str(xml_path))
    return _parse_tree(tree)


def _parse_tree(tree: ET.ElementTree) -> tuple[np.ndarray, int]:
    """Internal: detect format from a parsed ElementTree and dispatch."""
    root = tree.getroot()

    def _bare_tag(el: ET.Element) -> str:
        return el.tag.split("}")[-1] if "}" in el.tag else el.tag

    root_tag = _bare_tag(root).lower()

    # Detection heuristics
    is_ge = False
    is_philips = False

    if "restingecg" in root_tag:
        is_ge = True
    elif any(k in root_tag for k in ("clinicaldocument", "ecg", "signals")):
        is_philips = True
    else:
        # Scan top-level children for clues
        for child in root:
            ct = _bare_tag(child).lower()
            if ct == "waveform":
                is_ge = True
                break
            if ct in ("component", "signals"):
                is_philips = True
                break

    if is_ge:
        try:
            return parse_ge_muse_xml(tree)
        except Exception as ge_err:
            # Fallback to Philips
            try:
                return parse_philips_xml(tree)
            except Exception:
                raise ValueError(f"GE Muse parse failed: {ge_err}") from ge_err

    if is_philips:
        try:
            return parse_philips_xml(tree)
        except Exception as ph_err:
            try:
                return parse_ge_muse_xml(tree)
            except Exception:
                raise ValueError(f"Philips parse failed: {ph_err}") from ph_err

    # Unknown root — try both
    errors: list[str] = []
    for parser, label in ((parse_ge_muse_xml, "GE Muse"), (parse_philips_xml, "Philips")):
        try:
            result = parser(tree)
            return result
        except Exception as e:
            errors.append(f"{label}: {e}")

    raise ValueError(
        "ECG XML format not recognized. Tried GE Muse and Philips parsers.\n"
        + "\n".join(errors)
    )


def xml_to_png(
    xml_path: str | Path,
    out_path: str | Path,
    sample_rate: Optional[int] = None,
    paper_speed: float = 25.0,
    gain: float = 10.0,
) -> Path:
    """Convert an institutional ECG XML file to a hospital-style PNG.

    Args:
        xml_path: Path to ECG XML file.
        out_path: Output PNG path.
        sample_rate: Override detected sample rate (Hz).
        paper_speed: Paper speed in mm/s (default 25).
        gain: Gain in mm/mV (default 10).

    Returns:
        Path to the created PNG file.
    """
    signal, detected_rate = parse_ecg_xml(xml_path)
    sr = sample_rate if sample_rate is not None else detected_rate
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from .ecg import plot_hospital_ecg

    plot_hospital_ecg(signal, sampling_rate=sr, paper_speed=paper_speed, amplitude_scale=gain, save_path=str(out_path))
    # Resize to 700×700 thumbnail, consistent with npy_to_png behaviour
    try:
        from PIL import Image

        img = Image.open(out_path)
        img.thumbnail((700, 700), Image.Resampling.LANCZOS)
        img.save(out_path)
    except ImportError:
        pass
    return out_path


def xml_to_png_from_bytes(
    xml_bytes: bytes,
    out_path: str | Path,
    **kwargs,
) -> Path:
    """Convert ECG XML from bytes (e.g., from an HTTP upload).

    Writes the bytes to a temporary file, parses the XML, then renders PNG.

    Args:
        xml_bytes: Raw XML bytes.
        out_path: Destination PNG path.
        **kwargs: Forwarded to :func:`xml_to_png` (``sample_rate``, ``paper_speed``,
            ``gain``).

    Returns:
        Path to the created PNG file.
    """
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tf:
        tf.write(xml_bytes)
        tmp_path = Path(tf.name)
    try:
        return xml_to_png(tmp_path, out_path, **kwargs)
    finally:
        tmp_path.unlink(missing_ok=True)
