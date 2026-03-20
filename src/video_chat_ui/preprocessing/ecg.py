"""12-lead ECG .npy to hospital-style PNG (from test_UCSF/ecg_img_convert_v2)."""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def create_ecg_grid(ax, duration_s, y_low, y_high, paper_speed_mm_s=25, gain_mm_mV=10):
    minor_time_s = 1.0 / paper_speed_mm_s
    major_time_s = 5.0 / paper_speed_mm_s
    minor_voltage_mV = 1.0 / gain_mm_mV
    major_voltage_mV = 5.0 / gain_mm_mV
    t = 0.0
    while t <= duration_s + 1e-12:
        ax.axvline(x=t, color="#ffb3b3", lw=0.3, alpha=0.7, zorder=0)
        t += minor_time_s
    v = y_low
    while v <= y_high + 1e-12:
        ax.axhline(y=v, color="#ffb3b3", lw=0.3, alpha=0.7, zorder=0)
        v += minor_voltage_mV
    t = 0.0
    while t <= duration_s + 1e-12:
        ax.axvline(x=t, color="#ff6666", lw=0.6, alpha=0.9, zorder=0)
        t += major_time_s
    v = y_low
    while v <= y_high + 1e-12:
        ax.axhline(y=v, color="#ff6666", lw=0.6, alpha=0.9, zorder=0)
        v += major_voltage_mV
    ax.set_facecolor("#fff5f5")


def draw_calibration_pulse(ax, start_s=0.2, height_mV=1.0, width_s=0.2):
    y0 = ax.get_ylim()[0]
    t0, t1 = start_s, start_s + width_s
    y_step = y0 + height_mV
    ax.plot([t0, t0, t1, t1], [y0, y_step, y_step, y0], color="black", lw=1.6, solid_joinstyle="miter", zorder=3)


def annotate_seconds(ax, duration_s, every_s=1.0):
    ticks = np.arange(0, duration_s + 1e-9, every_s)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}" for t in ticks], fontsize=9)
    for t in ticks:
        ax.axvline(x=t, color="black", lw=0.6, alpha=0.25, zorder=1)


def add_six_second_bracket(ax, y_frac=0.1, start_s=0.0, width_s=6.0):
    y0, y1 = ax.get_ylim()
    y = y0 + (y1 - y0) * y_frac
    x0, x1 = start_s, start_s + width_s
    ax.plot([x0, x0, x1, x1], [y, y * 0.999, y * 0.999, y], color="black", lw=1.0, zorder=3)
    ax.text((x0 + x1) / 2, y, "6 s", ha="center", va="bottom", fontsize=9)


def plot_hospital_ecg(
    ecg_data,
    sampling_rate=500,
    paper_speed=25,
    amplitude_scale=10,
    lead_names=None,
    save_path=None,
    rhythm_lead_idx=1,
    rhythm_duration_s=10,
):
    if lead_names is None:
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    n_samples = ecg_data.shape[1]
    duration_s = n_samples / float(sampling_rate)
    t = np.linspace(0, duration_s, n_samples)
    mn, mx = np.nanmin(ecg_data), np.nanmax(ecg_data)
    pad = 0.1 * (mx - mn + 1e-12)
    y_low, y_high = mn - pad, mx + pad
    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        5, 3, height_ratios=[1, 1, 1, 1, 0.8], hspace=0.15, wspace=0.1, left=0.05, right=0.95, top=0.9, bottom=0.08
    )
    lead_positions = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 1),
        (3, 2),
    ]
    for i, (r, c) in enumerate(lead_positions):
        ax = fig.add_subplot(gs[r, c])
        create_ecg_grid(ax, duration_s, y_low, y_high, paper_speed_mm_s=paper_speed, gain_mm_mV=amplitude_scale)
        ax.plot(t, ecg_data[i], color="black", lw=0.9, zorder=2)
        ax.set_xlim(0, duration_s)
        ax.set_ylim(y_low, y_high)
        ax.text(
            0.02,
            0.93,
            lead_names[i],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, lw=0),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if lead_names[i] in ("I", "II", "V1"):
            draw_calibration_pulse(ax)
    ax_r = fig.add_subplot(gs[4, :])
    n_samples_r = min(int(rhythm_duration_s * sampling_rate), n_samples)
    t_r = np.linspace(0, n_samples_r / float(sampling_rate), n_samples_r)
    strip = ecg_data[rhythm_lead_idx, :n_samples_r]
    mn_r, mx_r = np.nanmin(strip), np.nanmax(strip)
    pad_r = 0.1 * (mx_r - mn_r + 1e-12)
    y_low_r, y_high_r = mn_r - pad_r, mx_r + pad_r
    create_ecg_grid(ax_r, t_r[-1], y_low_r, y_high_r, paper_speed_mm_s=paper_speed, gain_mm_mV=amplitude_scale)
    ax_r.plot(t_r, strip, color="black", lw=1.0, zorder=2)
    ax_r.set_xlim(0, t_r[-1])
    ax_r.set_ylim(y_low_r, y_high_r)
    annotate_seconds(ax_r, t_r[-1], every_s=1.0)
    draw_calibration_pulse(ax_r)
    add_six_second_bracket(ax_r)
    ax_r.set_xlabel("Time (s)")
    ax_r.set_ylabel("Amplitude (mV)")
    ax_r.set_title(f"Lead {lead_names[rhythm_lead_idx]} — Rhythm Strip", fontsize=12, fontweight="bold")
    tech = f"Speed: {paper_speed} mm/s | Gain: {amplitude_scale} mm/mV | Duration: {duration_s:.1f} s | Fs: {sampling_rate} Hz"
    fig.text(0.5, 0.04, tech, ha="center", fontsize=10, style="italic")
    if save_path:
        plt.savefig(save_path, dpi=72, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        from PIL import Image

        img = Image.open(save_path)
        img.thumbnail((700, 700), Image.Resampling.LANCZOS)
        img.save(save_path)
    else:
        plt.close(fig)
    return None


def npy_to_png(data: bytes | Path, out_path: Path, sampling_rate: int = 500) -> Path:
    """Load 12 x N ECG from .npy bytes or path; write PNG."""
    try:
        import PIL  # noqa: F401
    except ImportError as e:
        raise ImportError("pip install 'video-chat-ui[preprocessing]'") from e
    if isinstance(data, bytes):
        arr = np.load(io.BytesIO(data))
    else:
        arr = np.load(Path(data))
    if arr.ndim != 2 or arr.shape[0] != 12:
        raise ValueError(f"ECG array must be shape (12, N); got {arr.shape}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_hospital_ecg(arr, sampling_rate=sampling_rate, save_path=str(out_path))
    return out_path
