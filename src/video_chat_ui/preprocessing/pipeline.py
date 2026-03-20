"""TGZ -> CMR/Echo grid MP4; NPY -> ECG PNG."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Literal

from video_chat_ui.preprocessing.config import PreprocessConfig


def preprocess_tgz_for_expert(
    tgz_path: Path,
    expert: Literal["cmr", "echo"],
    cfg: PreprocessConfig | None = None,
) -> Path:
    """
    Extract DICOM study from .tgz, build modality-specific grid video.
    Returns (path to final video file, scratch work dir to delete after copy).
    """
    cfg = cfg or PreprocessConfig()
    work = Path(cfg.resolved_workdir()) / f"prep_{uuid.uuid4().hex}"
    work.mkdir(parents=True, exist_ok=True)
    extracted_root = work / "extracted"
    extracted_root.mkdir(exist_ok=True)

    import video_chat_ui.preprocessing.dicom_processor as dp

    dp.TEMP_BASE_DIR = str(work / "dicom_tmp")
    dp.OUTPUT_BASE_DIR = str(extracted_root)
    Path(dp.TEMP_BASE_DIR).mkdir(parents=True, exist_ok=True)

    proc = dp.DicomStudyProcessor(str(tgz_path))
    if not proc.process():
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError("DICOM extraction or processing failed")

    study_name = proc.study_name
    study_dir = extracted_root / study_name
    if not study_dir.is_dir():
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError(f"Expected study dir missing: {study_dir}")

    out_mp4 = work / f"{study_name}_{expert}_grid.mp4"

    try:
        if expert == "cmr":
            from video_chat_ui.preprocessing.cmr_grid import main as cmr_main

            csvs = list(study_dir.glob("*_metadata.csv"))
            if not csvs:
                raise FileNotFoundError(f"No *_metadata.csv under {study_dir}")
            cmr_main(str(csvs[0]), str(out_mp4))
            final = out_mp4
        else:
            from video_chat_ui.preprocessing.echo_grid import (
                _process_study_with_compression,
            )

            grid_sub = work / "echo_grids"
            comp_sub = work / "echo_out"
            grid_sub.mkdir(exist_ok=True)
            comp_sub.mkdir(exist_ok=True)
            _name, path, msg = _process_study_with_compression(
                str(study_dir),
                str(grid_sub),
                str(comp_sub),
                5,
                (256, 256),
                24,
                10,
                "freeze_last",
                "MJPG",
                42,
                700,
                5,
                5,
                True,
                False,
            )
            if not path:
                _name2, path2, msg2 = _process_study_with_compression(
                    str(study_dir),
                    str(grid_sub),
                    str(comp_sub),
                    5,
                    (256, 256),
                    24,
                    10,
                    "freeze_last",
                    "MJPG",
                    42,
                    700,
                    5,
                    5,
                    False,
                    True,
                )
                path = path2
                msg = msg2
            if not path:
                raise RuntimeError(msg or "Echo grid failed")
            final = Path(path)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise

    if not final.is_file():
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError("Grid video not produced")
    return final.resolve(), work


def preprocess_ecg_npy(npy_path: Path | None, npy_bytes: bytes | None, cfg: PreprocessConfig | None = None) -> tuple[Path, Path]:
    from video_chat_ui.preprocessing.ecg import npy_to_png

    cfg = cfg or PreprocessConfig()
    work = Path(cfg.resolved_workdir()) / f"ecg_{uuid.uuid4().hex}"
    work.mkdir(parents=True, exist_ok=True)
    out_png = work / "ecg.png"
    if npy_bytes is not None:
        npy_to_png(npy_bytes, out_png)
    elif npy_path is not None:
        npy_to_png(Path(npy_path), out_png)
    else:
        raise ValueError("npy_path or npy_bytes required")
    return out_png.resolve(), work


def preprocess_ecg_xml(
    xml_path: Path | None,
    xml_bytes: bytes | None,
    cfg: PreprocessConfig | None = None,
    sample_rate: int | None = None,
) -> tuple[Path, Path]:
    """Convert an institutional ECG XML file (GE Muse / Philips) to a PNG.

    Accepts either a file path or raw bytes (e.g., from an HTTP upload).
    Returns ``(png_path, scratch_work_dir)`` — caller is responsible for
    removing the scratch directory after copying the output.
    """
    from video_chat_ui.preprocessing.ecg_xml import xml_to_png, xml_to_png_from_bytes

    cfg = cfg or PreprocessConfig()
    work = Path(cfg.resolved_workdir()) / f"ecg_xml_{uuid.uuid4().hex}"
    work.mkdir(parents=True, exist_ok=True)
    out_png = work / "ecg.png"
    if xml_bytes is not None:
        xml_to_png_from_bytes(xml_bytes, out_png, sample_rate=sample_rate)
    elif xml_path is not None:
        xml_to_png(Path(xml_path), out_png, sample_rate=sample_rate)
    else:
        raise ValueError("xml_path or xml_bytes required")
    return out_png.resolve(), work
