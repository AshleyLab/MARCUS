"""
MARCUS Demo — Professional Gradio UI.

Showcases single-modality analysis (ECG, Echo, CMR), multimodal
orchestrated synthesis, and counterfactual mirage detection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

from video_chat_ui.demo_examples import (
    EXAMPLES,
    MULTIMODAL_EXAMPLE,
    TEMPLATE_QUESTIONS,
    get_example,
    get_example_choices,
)

logger = logging.getLogger(__name__)


_GIF_CACHE: dict[str, str] = {}


def _video_to_gif(video_path: str, width: int = 480) -> str | None:
    """Convert a video to an animated GIF for browser preview.

    Video streaming often fails over SSH tunnels / port-forwarded connections,
    so we convert to GIF which loads as a normal image and animates natively.
    Caches the result so each source file is only converted once.
    """
    if video_path in _GIF_CACHE:
        cached = _GIF_CACHE[video_path]
        if Path(cached).exists():
            return cached
    try:
        import subprocess
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

        out_path = tempfile.mktemp(suffix=".gif", dir="/tmp")
        result = subprocess.run(
            [ffmpeg_bin, "-i", video_path,
             "-vf", f"fps=5,scale={width}:-1:flags=lanczos",
             "-loop", "0",
             "-y", out_path],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0 or not Path(out_path).exists():
            return None

        _GIF_CACHE[video_path] = out_path
        return out_path
    except Exception as exc:
        logger.warning("Video to GIF failed for %s: %s", video_path, exc)
        return None

# ---------------------------------------------------------------------------
# Expert endpoint configuration
# ---------------------------------------------------------------------------

EXPERTS = {
    "ecg": {"api_port": 8020, "ui_port": 8775, "media_kind": "image", "label": "ECG"},
    "echo": {"api_port": 8010, "ui_port": 8770, "media_kind": "video", "label": "Echo"},
    "cmr": {"api_port": 8000, "ui_port": 8765, "media_kind": "video", "label": "CMR"},
}

FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "figures"

# ---------------------------------------------------------------------------
# CSS — Professional design inspired by AlphaFold, EchoNet, Google Health
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ══════════════════════════════════════════════════════════════════════
   MARCUS — Professional UI Stylesheet
   Design: Clean academic aesthetic
   ══════════════════════════════════════════════════════════════════════ */

/* ── Force light mode ────────────────────────────────────────────────── */
:root { color-scheme: light !important; }
body.dark { background: #ffffff !important; }

/* ── Google Fonts import ────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ──────────────────────────────────────────────────────────── */
.gradio-container {
    max-width: 1100px !important;
    margin: auto;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: #ffffff !important;
}
.gradio-container .prose { font-size: 0.9rem; }
.gradio-container .prose p { color: #3b4252; line-height: 1.7; }

/* ── Header ──────────────────────────────────────────────────────────── */
.marcus-header {
    text-align: center;
    padding: 48px 24px 32px;
    margin-bottom: 4px;
    position: relative;
}
.marcus-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 80px;
    height: 3px;
    background: linear-gradient(90deg, #1e3a5f, #2563eb);
    border-radius: 2px;
}
.marcus-header h1 {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: 0.18em;
    margin: 0 0 8px;
}
.marcus-header .subtitle {
    font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif;
    font-size: 1.05rem;
    color: #475569;
    font-weight: 400;
    margin: 0 0 14px;
    letter-spacing: 0.005em;
    font-style: italic;
}
.marcus-header .authors {
    font-size: 0.8rem;
    color: #64748b;
    margin: 0;
    line-height: 1.7;
    letter-spacing: 0.01em;
}
.marcus-header .authors strong {
    color: #334155;
    font-weight: 600;
}
.marcus-header .affiliation {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 4px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

/* ── Tab styling ─────────────────────────────────────────────────────── */
.tab-nav {
    border-bottom: 2px solid #e5e7eb !important;
    margin-top: 8px !important;
}
.tab-nav button {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: #64748b !important;
    padding: 10px 20px !important;
    border: none !important;
    background: transparent !important;
    transition: color 0.2s, border-color 0.2s;
}
.tab-nav button.selected {
    color: #0f172a !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #1e3a5f !important;
}
.tab-nav button:hover {
    color: #1e293b !important;
}

/* ── Section headings ────────────────────────────────────────────────── */
h2 {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #0f172a;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    padding-bottom: 0;
    border-bottom: none;
}
h3 {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #1e293b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 1.2em;
}

/* ── Status badges ───────────────────────────────────────────────────── */
.status-row {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 16px 0 8px;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
}
.status-online {
    background: #f0fdf4;
    color: #166534;
    border: 1px solid #bbf7d0;
}
.status-offline {
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
}
.status-online .status-dot {
    background: #22c55e;
    box-shadow: 0 0 4px rgba(34, 197, 94, 0.4);
}
.status-offline .status-dot { background: #ef4444; }

/* ── Inputs & controls ───────────────────────────────────────────────── */
.gradio-container input[type="text"],
.gradio-container textarea {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.88rem !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    transition: border-color 0.2s !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.08) !important;
}
.gradio-container button.primary {
    background: #1e3a5f !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    padding: 10px 28px !important;
    transition: background 0.2s, box-shadow 0.2s !important;
}
.gradio-container button.primary:hover {
    background: #15304f !important;
    box-shadow: 0 2px 8px rgba(30, 58, 95, 0.25) !important;
}
.gradio-container button.secondary {
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    color: #374151 !important;
    background: #ffffff !important;
}
.gradio-container button.secondary:hover {
    background: #f9fafb !important;
    border-color: #9ca3af !important;
}

/* ── Result boxes ────────────────────────────────────────────────────── */
.result-box textarea {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.87rem !important;
    line-height: 1.72 !important;
    color: #1e293b !important;
    background: #fafbfc !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
}

/* ── Score cards ──────────────────────────────────────────────────────── */
.score-card {
    background: #fafbfc;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.score-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #0f172a;
    font-family: 'JetBrains Mono', monospace;
}
.score-label {
    font-size: 0.68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
    font-weight: 500;
}

/* ── Verdict banners ─────────────────────────────────────────────────── */
.verdict-pass {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 14px 20px;
    color: #166534;
    font-weight: 500;
    font-size: 0.88rem;
    text-align: left;
}
.verdict-fail {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 14px 20px;
    color: #78350f;
    font-weight: 500;
    font-size: 0.88rem;
    text-align: left;
}

/* ── Timing pill ─────────────────────────────────────────────────────── */
.timing-pill {
    display: inline-block;
    background: #f1f5f9;
    color: #475569;
    font-size: 0.73rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 12px;
    border-radius: 4px;
    margin-top: 8px;
    letter-spacing: 0.01em;
}

/* ── Figure captions ─────────────────────────────────────────────────── */
.figure-caption {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 0.78rem;
    color: #64748b;
    text-align: center;
    margin-top: 6px;
    font-style: italic;
}

/* ── Table refinements ───────────────────────────────────────────────── */
table {
    font-size: 0.84rem !important;
    border-collapse: collapse !important;
}
table th {
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #475569 !important;
    border-bottom: 2px solid #1e3a5f !important;
}
table td {
    border-bottom: 1px solid #f1f5f9 !important;
    padding: 8px 12px !important;
}

/* ── Hide Gradio footer ──────────────────────────────────────────────── */
footer { display: none !important; }

/* ── Accordion styling ───────────────────────────────────────────────── */
.gradio-container .accordion {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    overflow: hidden;
}
.gradio-container .label-wrap {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

/* ── Pipeline step indicator ─────────────────────────────────────────── */
.pipeline-steps {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0;
    margin: 20px 0 12px;
    flex-wrap: wrap;
}
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0;
}
.step-box {
    background: #ffffff;
    border: 1.5px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 0.76rem;
    font-weight: 600;
    color: #334155;
    white-space: nowrap;
    letter-spacing: 0.02em;
    transition: all 0.2s;
}
.step-box:hover {
    border-color: #1e3a5f;
    color: #1e3a5f;
}
.step-arrow {
    color: #94a3b8;
    font-size: 1.1rem;
    padding: 0 8px;
    font-weight: 300;
}

/* ── Disclaimer ──────────────────────────────────────────────────────── */
.disclaimer {
    text-align: center;
    font-size: 0.72rem;
    color: #94a3b8;
    padding: 20px 24px;
    border-top: 1px solid #e5e7eb;
    margin-top: 32px;
    letter-spacing: 0.01em;
    line-height: 1.6;
}

/* ── Mirage example cards ────────────────────────────────────────────── */
.mirage-example {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    background: #fafbfc;
}
.mirage-example-pass { border-left: 3px solid #22c55e; }
.mirage-example-fail { border-left: 3px solid #f59e0b; }

/* ── Image/media containers ──────────────────────────────────────────── */
.gradio-container .image-container {
    border-radius: 8px !important;
    overflow: hidden;
    border: 1px solid #e5e7eb !important;
}

/* ── File upload styling ─────────────────────────────────────────────── */
.gradio-container .upload-button {
    border: 2px dashed #d1d5db !important;
    border-radius: 8px !important;
    background: #fafbfc !important;
    transition: border-color 0.2s !important;
}
.gradio-container .upload-button:hover {
    border-color: #2563eb !important;
}

/* ── Checkbox styling ────────────────────────────────────────────────── */
.gradio-container input[type="checkbox"] {
    accent-color: #1e3a5f !important;
}

/* ── Radio button styling ────────────────────────────────────────────── */
.gradio-container .radio-group label {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── Dropdown styling ────────────────────────────────────────────────── */
.gradio-container .dropdown-container {
    border-radius: 6px !important;
}

/* ── Citation code block ─────────────────────────────────────────────── */
.gradio-container .code-block {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Smooth scrolling ────────────────────────────────────────────────── */
html { scroll-behavior: smooth; }

/* ── Selection color ─────────────────────────────────────────────────── */
::selection {
    background: rgba(37, 99, 235, 0.15);
    color: #0f172a;
}
"""

# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


async def _check_expert(modality: str) -> bool:
    """Ping a single expert API."""
    port = EXPERTS[modality]["api_port"]
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"http://127.0.0.1:{port}/v1/models", timeout=3)
            return r.status_code == 200
    except Exception:
        return False


async def check_all_health() -> str:
    """Return Markdown status badges for all experts."""
    results = await asyncio.gather(*[_check_expert(m) for m in EXPERTS])
    parts = ['<div class="status-row">']
    for (mod, cfg), alive in zip(EXPERTS.items(), results):
        cls = "status-online" if alive else "status-offline"
        txt = "Online" if alive else "Offline"
        parts.append(
            f'<span class="status-badge {cls}">'
            f'<span class="status-dot"></span>'
            f'{cfg["label"]}: {txt}</span>'
        )
    parts.append("</div>")
    return "".join(parts)


def _ui_url(modality: str) -> str:
    return f"http://127.0.0.1:{EXPERTS[modality]['ui_port']}"


def _api_url(modality: str) -> str:
    return f"http://127.0.0.1:{EXPERTS[modality]['api_port']}"


async def _upload_file(file_path: str, modality: str) -> tuple[str, str]:
    """Upload a file to the expert's FastAPI UI server. Returns (media_id, media_kind)."""
    p = Path(file_path)
    ext = p.suffix.lower()
    base = _ui_url(modality)

    # Raw formats go through /preprocess
    if ext in (".npy", ".xml", ".tgz", ".tar.gz"):
        async with httpx.AsyncClient(timeout=300) as c:
            with open(file_path, "rb") as f:
                resp = await c.post(
                    f"{base}/preprocess",
                    files={"file": (p.name, f)},
                    data={"expert": modality},
                )
            resp.raise_for_status()
            data = resp.json()
            return data["id"], data["kind"]

    # Already-processed files go through /upload
    async with httpx.AsyncClient(timeout=120) as c:
        with open(file_path, "rb") as f:
            resp = await c.post(
                f"{base}/upload",
                files={"video": (p.name, f)},
            )
        resp.raise_for_status()
        data = resp.json()
        kind = "image" if ext in (".png", ".jpg", ".jpeg") else "video"
        return data["id"], kind


async def _chat_query(modality: str, media_id: str, media_kind: str, question: str) -> str:
    """Send a chat query to the expert's FastAPI UI and collect the full response."""
    full_text = ""
    async for token in _stream_chat_query(modality, media_id, media_kind, question):
        full_text += token
    return full_text


async def _stream_chat_query(modality: str, media_id: str, media_kind: str, question: str):
    """Stream tokens from the expert's FastAPI UI via SSE. Yields each token as it arrives."""
    base = _ui_url(modality)
    payload = {
        "video_id": media_id,
        "media_kind": media_kind,
        "messages": [{"role": "user", "content": question}],
    }
    async with httpx.AsyncClient(timeout=180) as c:
        async with c.stream("POST", f"{base}/chat", json=payload) as resp:
            resp.raise_for_status()
            buf = ""
            async for chunk in resp.aiter_text():
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        parsed = json.loads(data_str)
                        delta = parsed.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    import re
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Attention map visualization
# ---------------------------------------------------------------------------


async def _attention_query(modality: str, media_id: str, media_kind: str, question: str) -> dict:
    """Query the attention endpoint to get per-token attention maps."""
    base = _ui_url(modality)
    payload = {
        "video_id": media_id,
        "media_kind": media_kind,
        "messages": [{"role": "user", "content": question}],
    }
    async with httpx.AsyncClient(timeout=300) as c:
        resp = await c.post(f"{base}/chat_attention", json=payload)
        resp.raise_for_status()
        return resp.json()


def _load_base_image(path: str):
    """Load the base image for attention overlay.  For videos, extracts the middle frame.

    Returns a PIL Image or None.
    """
    from PIL import Image

    if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        return _extract_middle_frame(path)
    return Image.open(path).convert("RGB")


_HEATMAP_PROFILES = {
    "ecg": {
        "power_exponent": 0.5,       # broadens attention (sqrt) → smooth blobs
        "upsample_method": "bilinear",
        "sharpen_passes": 0,          # no sharpening
        "final_upsample": "bilinear",
        "alpha_floor": 0.15,
        "alpha_ceil": 0.60,
        "alpha_threshold": 0.0,       # overlay everywhere
    },
    "echo": {
        "power_exponent": 1.0,       # linear (between blobby 0.5 and sharp 1.5)
        "upsample_method": "bicubic",
        "sharpen_passes": 1,          # one pass (softer than two)
        "final_upsample": "bicubic",  # softer than Lanczos
        "alpha_floor": 0.10,
        "alpha_ceil": 0.60,
        "alpha_threshold": 0.10,
    },
    "cmr": {
        "power_exponent": 1.0,
        "upsample_method": "bicubic",
        "sharpen_passes": 1,
        "final_upsample": "bicubic",
        "alpha_floor": 0.10,
        "alpha_ceil": 0.60,
        "alpha_threshold": 0.10,
    },
}
_HEATMAP_DEFAULT = _HEATMAP_PROFILES["echo"]


def _create_attention_heatmap(
    attention_weights: list[float],
    grid_h: int,
    grid_w: int,
    base_image,
    grid_t: int = 1,
    modality: str = "",
) -> "np.ndarray | None":
    """Create a heatmap overlay on the base image from attention weights.

    For videos (grid_t > 1), averages attention over the temporal dimension.
    ``base_image`` should be a pre-loaded PIL Image (use ``_load_base_image``).
    ``modality`` selects rendering profile (ecg/echo/cmr).
    Returns numpy array (H, W, 3) or None on failure.
    """
    try:
        import numpy as np
        from PIL import Image as PILImage
        from PIL import ImageFilter

        if base_image is None:
            return None

        prof = _HEATMAP_PROFILES.get(modality.lower(), _HEATMAP_DEFAULT)

        # Reshape attention to spatial grid (averaging over temporal dim for video)
        n_spatial = grid_h * grid_w
        n_total = n_spatial * grid_t
        w_arr = np.array(attention_weights, dtype=np.float32)
        if len(w_arr) < n_total:
            w_arr = np.pad(w_arr, (0, n_total - len(w_arr)))
        else:
            w_arr = w_arr[:n_total]
        if grid_t > 1:
            attn_map = w_arr.reshape(grid_t, grid_h, grid_w).mean(axis=0)
        else:
            attn_map = w_arr[:n_spatial].reshape(grid_h, grid_w)

        # Normalize to [0, 1]
        vmin, vmax = attn_map.min(), attn_map.max()
        if vmax > vmin:
            attn_map = (attn_map - vmin) / (vmax - vmin)
        else:
            attn_map = np.zeros_like(attn_map)

        w, h = base_image.size

        # --- Upscale with modality-specific pipeline ---
        _resample = {
            "bilinear": PILImage.BILINEAR,
            "bicubic": PILImage.BICUBIC,
            "lanczos": PILImage.LANCZOS,
        }
        mid_w, mid_h = grid_w * 4, grid_h * 4
        attn_uint8 = (attn_map * 255).astype(np.uint8)
        attn_pil = PILImage.fromarray(attn_uint8, mode="L")
        attn_mid = attn_pil.resize(
            (mid_w, mid_h),
            _resample.get(prof["upsample_method"], PILImage.BICUBIC),
        )

        for _ in range(prof["sharpen_passes"]):
            attn_mid = attn_mid.filter(ImageFilter.SHARPEN)

        attn_resized = attn_mid.resize(
            (w, h),
            _resample.get(prof["final_upsample"], PILImage.LANCZOS),
        )
        attn_np = np.array(attn_resized, dtype=np.float32) / 255.0
        attn_np = np.clip(attn_np, 0, 1)

        # Power-law contrast (modality-specific)
        attn_np = np.power(attn_np, prof["power_exponent"])
        a_max = attn_np.max()
        if a_max > 0:
            attn_np = attn_np / a_max

        # Jet-like colormap: blue -> cyan -> green -> yellow -> red
        heatmap_rgb = np.zeros((h, w, 3), dtype=np.float32)
        heatmap_rgb[:, :, 0] = np.clip(1.5 * attn_np - 0.25, 0, 1)
        heatmap_rgb[:, :, 1] = np.clip(1.0 - 2.0 * np.abs(attn_np - 0.5), 0, 1)
        heatmap_rgb[:, :, 2] = np.clip(1.25 - 1.5 * attn_np, 0, 1)

        heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
        heatmap_img = PILImage.fromarray(heatmap_rgb, mode="RGB")

        # Blend with modality-specific alpha
        base_np = np.array(base_image, dtype=np.float32)
        heat_np = np.array(heatmap_img, dtype=np.float32)
        threshold = prof["alpha_threshold"]
        floor = prof["alpha_floor"]
        ceil = prof["alpha_ceil"]
        alpha_map = np.where(
            attn_np < threshold, 0.0, floor + (ceil - floor) * attn_np,
        )
        alpha_map = np.clip(alpha_map, 0.0, ceil)
        alpha_3ch = np.stack([alpha_map] * 3, axis=-1)
        blended_np = (1 - alpha_3ch) * base_np + alpha_3ch * heat_np
        return blended_np.astype(np.uint8)

    except Exception as exc:
        logger.warning("Attention heatmap creation failed: %s", exc)
        return None


def _extract_middle_frame(video_path: str):
    """Extract the middle frame from a video file."""
    try:
        import subprocess
        import imageio_ffmpeg
        from PIL import Image
        import io

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

        # Get video duration
        probe = subprocess.run(
            [ffmpeg_bin, "-i", video_path],
            capture_output=True, timeout=10,
        )
        # Parse duration from stderr
        import re
        duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", probe.stderr.decode())
        if duration_match:
            h, m, s = duration_match.groups()
            total_s = int(h) * 3600 + int(m) * 60 + float(s)
            mid_s = total_s / 2
        else:
            mid_s = 1.0  # fallback

        # Extract frame at middle timestamp
        out_path = tempfile.mktemp(suffix=".png", dir="/tmp")
        result = subprocess.run(
            [ffmpeg_bin, "-ss", str(mid_s), "-i", video_path,
             "-vframes", "1", "-y", out_path],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and Path(out_path).exists():
            return Image.open(out_path).convert("RGB")
        return None
    except Exception as exc:
        logger.warning("Middle frame extraction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Single-modality handler
# ---------------------------------------------------------------------------


async def run_single_analysis(
    file_obj: Any,
    modality: str,
    question: str,
    enable_mirage: bool,
    enable_attention: bool = False,
    template: str | None = None,
):
    """Run single-modality analysis. Yields (answer, confidence_info, timing, attn_map) with streaming."""
    if file_obj is None:
        yield "Please upload a file or select an example.", "", "", None
        return
    # Fall back to template dropdown if textbox is empty
    if not question.strip() and template:
        question = template
    if not question.strip():
        yield "Please enter a clinical question.", "", "", None
        return

    modality = modality.lower()
    t0 = time.time()

    # Get file path
    file_path = file_obj if isinstance(file_obj, str) else file_obj.name if hasattr(file_obj, "name") else str(file_obj)

    try:
        alive = await _check_expert(modality)
        if not alive:
            yield f"The {EXPERTS[modality]['label']} expert is not available. Start the model server first.", "", "", None
            return

        media_id, media_kind = await _upload_file(file_path, modality)

        if enable_attention:
            # --- Attention map mode: non-streaming with per-token attention ---
            yield "Generating with attention extraction (slower)...", "", "", None

            try:
                attn_data = await _attention_query(modality, media_id, media_kind, question)
            except Exception as exc:
                yield f"Attention extraction failed: {exc}", "", "", None
                return

            tokens = attn_data.get("tokens", [])
            attn_maps = attn_data.get("attention_maps", [])
            grid_h = attn_data.get("grid_h", 0)
            grid_w = attn_data.get("grid_w", 0)
            grid_t = attn_data.get("grid_t", 1)
            full_response = attn_data.get("response", "")
            full_answer = _strip_think_tags(full_response)

            # Pre-load base image once (avoids repeated ffmpeg calls for video)
            base_img = _load_base_image(file_path) if grid_h > 0 and grid_w > 0 else None

            # Replay tokens with attention heatmaps
            text_so_far = ""
            last_heatmap = None
            skip_think = False
            for i, tok in enumerate(tokens):
                text_so_far += tok
                # Handle <think> tags
                if "<think>" in text_so_far and "</think>" not in text_so_far:
                    skip_think = True
                if "</think>" in text_so_far:
                    skip_think = False

                display = _strip_think_tags(text_so_far)
                if skip_think:
                    display = display + " _(thinking...)_" if display else "_(thinking...)_"

                # Generate heatmap for this token
                if i < len(attn_maps) and base_img is not None:
                    heatmap = _create_attention_heatmap(
                        attn_maps[i], grid_h, grid_w, base_img, grid_t,
                        modality=modality,
                    )
                    if heatmap is not None:
                        last_heatmap = heatmap

                yield display, "", "", last_heatmap
                # Brief pause so users can see the attention shift per token
                await asyncio.sleep(0.08)

            elapsed = time.time() - t0
            timing = f'<span class="timing-pill">Completed in {elapsed:.1f}s (with attention)</span>'
            yield full_answer, "", timing, last_heatmap

        else:
            # --- Standard streaming mode ---
            answer_parts = []
            in_think = False
            async for token in _stream_chat_query(modality, media_id, media_kind, question):
                answer_parts.append(token)
                raw_so_far = "".join(answer_parts)
                display = _strip_think_tags(raw_so_far)
                if "<think>" in raw_so_far:
                    open_count = raw_so_far.count("<think>")
                    close_count = raw_so_far.count("</think>")
                    in_think = open_count > close_count
                if in_think:
                    yield display + " _(thinking...)_" if display else "_(thinking...)_", "", "", None
                else:
                    yield display, "", "", None

            full_answer = _strip_think_tags("".join(answer_parts))

            elapsed = time.time() - t0
            timing = f'<span class="timing-pill">Completed in {elapsed:.1f}s</span>'

            confidence_info = ""
            if enable_mirage:
                yield full_answer, "Running mirage probe...", timing, None
                try:
                    from video_chat_ui.orchestrator.mirage import MirageProbe
                    probe = MirageProbe(timeout=120.0)
                    result = await probe.probe_expert(
                        question=question,
                        media_id=media_id,
                        expert_api_url=_api_url(modality),
                        expert=modality,
                        media_kind=media_kind,
                        media_base_url=_ui_url(modality),
                    )
                    flag = "MIRAGE DETECTED" if result.mirage_flag else "No mirage detected"
                    confidence_info = (
                        f"**Mirage probe:** {flag}\n\n"
                        f"Consistency: {result.consistency_score:.3f} &middot; "
                        f"Divergence: {result.divergence_score:.3f} &middot; "
                        f"Confidence: {result.confidence_score:.3f}"
                    )
                    elapsed = time.time() - t0
                    timing = f'<span class="timing-pill">Analysis + mirage probe in {elapsed:.1f}s</span>'
                except Exception as e:
                    confidence_info = f"Mirage probing failed: {e}"

            yield full_answer, confidence_info, timing, None

    except Exception as e:
        yield f"Error: {e}", "", "", None


# ---------------------------------------------------------------------------
# Multimodal handler
# ---------------------------------------------------------------------------


async def run_multimodal_analysis(
    ecg_file: Any,
    echo_file: Any,
    cmr_file: Any,
    question: str,
    enable_attention: bool = False,
    template: str | None = None,
):
    """Run multimodal analysis with optional attention maps.

    Returns (synth_answer, ecg_resp, echo_resp, cmr_resp, confidence_md, timing,
             ecg_attn, echo_attn, cmr_attn).
    """
    empty = ("", "", "", "", "", "", None, None, None)
    # Fall back to template dropdown if textbox is empty
    if not question.strip() and template:
        question = template
    if not question.strip():
        yield ("Please enter a clinical question.",) + ("",) * 5 + (None, None, None)
        return

    files = {"ecg": ecg_file, "echo": echo_file, "cmr": cmr_file}
    provided = {m: f for m, f in files.items() if f is not None}
    if len(provided) < 2:
        yield ("Please provide at least two modalities for multimodal analysis.",) + ("",) * 5 + (None, None, None)
        return

    t0 = time.time()

    # Check health of needed experts
    for mod in provided:
        alive = await _check_expert(mod)
        if not alive:
            yield (f"The {EXPERTS[mod]['label']} expert is not available.",) + ("",) * 5 + (None, None, None)
            return

    try:
        # Upload all files — keep file paths for heatmap base images
        media_ids: dict[str, str] = {}
        media_kinds: dict[str, str] = {}
        file_paths: dict[str, str] = {}
        for mod, f in provided.items():
            fp = f if isinstance(f, str) else f.name if hasattr(f, "name") else str(f)
            mid, mk = await _upload_file(fp, mod)
            media_ids[mod] = mid
            media_kinds[mod] = mk
            file_paths[mod] = fp

        yield ("Running orchestrator...", "", "", "", "", "", None, None, None)

        # Run orchestrator — route_all=True so every provided modality is queried
        from video_chat_ui.orchestrator.orchestrator import MARCUSOrchestrator
        orch = MARCUSOrchestrator(enable_mirage_probing=True, timeout=180.0)
        result = await orch.synthesize(question=question, media_ids=media_ids, route_all=True)

        ecg_resp = _strip_think_tags(result.modality_responses.get("ecg", "—"))
        echo_resp = _strip_think_tags(result.modality_responses.get("echo", "—"))
        cmr_resp = _strip_think_tags(result.modality_responses.get("cmr", "—"))

        # Confidence info
        conf_lines = []
        for mod in ["ecg", "echo", "cmr"]:
            if mod in result.confidence_scores:
                score = result.confidence_scores[mod]
                flag = " ⚠ mirage" if result.mirage_flags.get(mod) else ""
                conf_lines.append(f"**{mod.upper()}**: {score:.3f}{flag}")
        confidence_md = " &middot; ".join(conf_lines) if conf_lines else ""

        synth_answer = _strip_think_tags(result.answer)

        if not enable_attention:
            elapsed = time.time() - t0
            timing = f'<span class="timing-pill">Multimodal synthesis in {elapsed:.1f}s</span>'
            yield (synth_answer, ecg_resp, echo_resp, cmr_resp, confidence_md, timing, None, None, None)
            return

        # --- Attention extraction for each modality (in parallel) ---
        yield (synth_answer, ecg_resp, echo_resp, cmr_resp, confidence_md,
               '<span class="timing-pill">Extracting attention maps...</span>', None, None, None)

        async def _get_attn(mod: str) -> dict | None:
            try:
                return await _attention_query(mod, media_ids[mod], media_kinds[mod], question)
            except Exception as exc:
                logger.warning("Multimodal attention failed for %s: %s", mod, exc)
                return None

        attn_tasks = {mod: asyncio.create_task(_get_attn(mod)) for mod in provided}
        attn_results: dict[str, dict | None] = {}
        for mod, task in attn_tasks.items():
            attn_results[mod] = await task

        # Generate heatmaps
        heatmaps: dict[str, Any] = {"ecg": None, "echo": None, "cmr": None}
        for mod in provided:
            data = attn_results.get(mod)
            if not data:
                continue
            attn_maps = data.get("attention_maps", [])
            grid_h = data.get("grid_h", 0)
            grid_w = data.get("grid_w", 0)
            grid_t = data.get("grid_t", 1)
            if attn_maps and grid_h > 0 and grid_w > 0:
                base_img = _load_base_image(file_paths[mod])
                # Use last token's attention (final answer token)
                last_attn = attn_maps[-1] if attn_maps else []
                hm = _create_attention_heatmap(last_attn, grid_h, grid_w, base_img, grid_t, modality=mod)
                if hm is not None:
                    heatmaps[mod] = hm

        elapsed = time.time() - t0
        timing = f'<span class="timing-pill">Multimodal synthesis in {elapsed:.1f}s (with attention)</span>'
        yield (synth_answer, ecg_resp, echo_resp, cmr_resp, confidence_md, timing,
               heatmaps["ecg"], heatmaps["echo"], heatmaps["cmr"])

    except Exception as e:
        yield (f"Error: {e}",) + ("",) * 5 + (None, None, None)


# ---------------------------------------------------------------------------
# Mirage detection handler
# ---------------------------------------------------------------------------


async def run_mirage_probe(
    file_obj: Any,
    modality: str,
    question: str,
) -> tuple[str, str, str, str, str, float, float, float, str]:
    """Run full mirage probing pipeline."""
    if file_obj is None:
        return "Upload a file first.", "", "", "", "", 0.0, 0.0, 0.0, ""
    if not question.strip():
        return "Enter a question.", "", "", "", "", 0.0, 0.0, 0.0, ""

    modality = modality.lower()
    alive = await _check_expert(modality)
    if not alive:
        return f"{EXPERTS[modality]['label']} expert offline.", "", "", "", "", 0.0, 0.0, 0.0, ""

    file_path = file_obj if isinstance(file_obj, str) else file_obj.name if hasattr(file_obj, "name") else str(file_obj)

    try:
        media_id, media_kind = await _upload_file(file_path, modality)

        from video_chat_ui.orchestrator.mirage import MirageProbe
        probe = MirageProbe(timeout=120.0)

        # Get rephrased questions for display
        rephrases = probe.rephrase_question(question, modality=modality)
        rephrase_display = "\n\n".join(f"**Query {i+1}:** {q}" for i, q in enumerate(rephrases))

        result = await probe.probe_expert(
            question=question,
            media_id=media_id,
            expert_api_url=_api_url(modality),
            expert=modality,
            media_kind=media_kind,
            media_base_url=_ui_url(modality),
        )

        r1 = _strip_think_tags(result.image_present_responses[0]) if len(result.image_present_responses) > 0 else ""
        r2 = _strip_think_tags(result.image_present_responses[1]) if len(result.image_present_responses) > 1 else ""
        r3 = _strip_think_tags(result.image_present_responses[2]) if len(result.image_present_responses) > 2 else ""

        if result.mirage_flag:
            verdict = '<div class="verdict-fail">MIRAGE DETECTED &mdash; Responses are similar with and without visual input, suggesting the model may not be genuinely referencing the provided data.</div>'
        else:
            verdict = '<div class="verdict-pass">NO MIRAGE &mdash; Responses are grounded in the provided visual data. The model\'s analysis changes substantively when the image is removed.</div>'

        return (
            rephrase_display,
            r1, r2, r3,
            _strip_think_tags(result.image_absent_response),
            result.consistency_score,
            result.divergence_score,
            result.confidence_score,
            verdict,
        )
    except Exception as e:
        return f"Error: {e}", "", "", "", "", 0.0, 0.0, 0.0, ""


# ---------------------------------------------------------------------------
# Example loading helpers
# ---------------------------------------------------------------------------


def load_single_example(modality: str, example_name: str):
    """Load a pre-loaded example for single-modality tab."""
    ex = get_example(modality.lower(), example_name)
    if ex is None or not Path(ex["path"]).is_file():
        return None, "", None

    p = ex["path"]
    question = ex["default_question"]
    if p.lower().endswith((".png", ".jpg", ".jpeg")):
        return p, question, p
    else:
        gif = _video_to_gif(p)
        return p, question, gif


def load_multimodal_example():
    """Load the pre-loaded multimodal example."""
    return MULTIMODAL_EXAMPLE["ecg"], MULTIMODAL_EXAMPLE["echo"], MULTIMODAL_EXAMPLE["cmr"]


def on_modality_change(modality: str):
    """Update UI components when modality changes."""
    mod = modality.lower()
    choices = get_example_choices(mod)
    templates = TEMPLATE_QUESTIONS.get(mod, [])
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=templates, value=None),
        None,  # clear preview
    )


def on_template_select(template: str):
    """Fill question input from template."""
    return template or ""


def on_file_upload(file_obj, modality: str):
    """Show preview when user uploads a file."""
    if file_obj is None:
        return None
    fp = file_obj if isinstance(file_obj, str) else file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    mod = modality.lower()
    if mod == "ecg":
        return fp
    else:
        return _video_to_gif(fp)


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="MARCUS | Cardiac AI",
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.HTML("""
        <div class="marcus-header">
            <h1>MARCUS</h1>
            <p class="subtitle">
                Multimodal Autonomous Reasoning and Chat for Ultrasound and Signals
            </p>
            <p class="authors">
                <strong>O'Sullivan JW*</strong>, <strong>Asadi M*</strong>,
                Elbe L, Chaudhari A, Nedaee T,
                Haddad F, Salerno M, Fei-Fei L, Adeli E, Arnaout R, Ashley EA
            </p>
            <p class="affiliation">Stanford University &middot; UCSF</p>
        </div>
        """)

        # System status — inline at top
        status_html = gr.HTML(
            '<div class="status-row">'
            '<span class="status-badge status-offline"><span class="status-dot"></span>Checking...</span>'
            '</div>'
        )
        demo.load(check_all_health, outputs=status_html)

        with gr.Tabs():

            # ════════════════════════════════════════════════════════════
            # TAB 1: Overview
            # ════════════════════════════════════════════════════════════
            with gr.Tab("Overview"):
                gr.Markdown(
                    "MARCUS is an agentic vision-language system for end-to-end "
                    "interpretation of **electrocardiograms**, **echocardiograms**, "
                    "and **cardiac MRI**. It reasons across modalities independently "
                    "and jointly, with built-in counterfactual safeguards against "
                    "hallucinated reasoning. Trained on 13.5 million images from "
                    "270,000 clinical studies with physician-verified ground truth."
                )

                # Pipeline visualisation
                gr.HTML("""
                <div class="pipeline-steps">
                    <div class="pipeline-step">
                        <span class="step-box">Upload Data</span>
                        <span class="step-arrow">→</span>
                    </div>
                    <div class="pipeline-step">
                        <span class="step-box">Expert Routing</span>
                        <span class="step-arrow">→</span>
                    </div>
                    <div class="pipeline-step">
                        <span class="step-box">Mirage Probing</span>
                        <span class="step-arrow">→</span>
                    </div>
                    <div class="pipeline-step">
                        <span class="step-box">Synthesis</span>
                    </div>
                </div>
                """)

                fig1 = FIGURES_DIR / "fig1_architecture.png"
                if fig1.is_file():
                    gr.Image(str(fig1), label="Architecture and training pipeline",
                             show_label=True, interactive=False, height=480)

                refresh_btn = gr.Button("Refresh Status", size="sm", variant="secondary")
                refresh_btn.click(check_all_health, outputs=status_html)

            # ════════════════════════════════════════════════════════════
            # TAB 2: Single-Modality Analysis
            # ════════════════════════════════════════════════════════════
            with gr.Tab("Single-Modality Analysis"):
                gr.Markdown(
                    "Upload an ECG image, echocardiogram video, or cardiac MRI video "
                    "and ask a clinical question. The corresponding expert model "
                    "will interpret the data."
                )

                modality_radio = gr.Radio(
                    choices=["ECG", "Echo", "CMR"], value="ECG",
                    label="Modality", interactive=True,
                )

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        single_file = gr.File(
                            label="Upload file",
                            file_types=[".png", ".jpg", ".jpeg", ".npy", ".xml",
                                        ".mp4", ".avi", ".mov", ".tgz"],
                        )
                        example_dd = gr.Dropdown(
                            choices=get_example_choices("ecg"),
                            label="Or select an example",
                            interactive=True,
                        )
                        load_ex_btn = gr.Button("Load Example", size="sm", variant="secondary")

                        img_preview = gr.Image(label="Preview", height=280)

                    with gr.Column(scale=1):
                        template_dd = gr.Dropdown(
                            choices=TEMPLATE_QUESTIONS["ecg"],
                            label="Template questions",
                            interactive=True,
                        )
                        question_input = gr.Textbox(
                            label="Clinical question",
                            placeholder="Ask a clinical question about this study...",
                            lines=2,
                        )
                        with gr.Row():
                            mirage_cb = gr.Checkbox(
                                label="Enable mirage probing",
                                value=False,
                            )
                            attention_cb = gr.Checkbox(
                                label="Show attention maps",
                                value=False,
                            )
                        submit_btn = gr.Button("Analyse", variant="primary")

                        gr.Markdown("### Response")
                        answer_box = gr.Textbox(
                            label="Model interpretation", lines=10, interactive=False,
                            elem_classes=["result-box"],
                        )
                        confidence_md = gr.Markdown("")
                        timing_md = gr.HTML("")

                        gr.Markdown("### Attention Map")
                        attn_image = gr.Image(
                            label="Per-token attention over input",
                            height=280,
                            visible=True,
                        )

                # Wiring
                modality_radio.change(
                    on_modality_change, modality_radio,
                    [example_dd, template_dd, img_preview],
                )
                template_dd.change(on_template_select, template_dd, question_input)
                single_file.change(
                    on_file_upload, [single_file, modality_radio],
                    [img_preview],
                )
                load_ex_btn.click(
                    load_single_example, [modality_radio, example_dd],
                    [single_file, question_input, img_preview],
                )
                submit_btn.click(
                    run_single_analysis,
                    [single_file, modality_radio, question_input, mirage_cb, attention_cb, template_dd],
                    [answer_box, confidence_md, timing_md, attn_image],
                )

            # ════════════════════════════════════════════════════════════
            # TAB 3: Multimodal Analysis
            # ════════════════════════════════════════════════════════════
            with gr.Tab("Multimodal Analysis"):
                gr.Markdown(
                    "Upload studies from two or three modalities. The MARCUS "
                    "orchestrator decomposes the query, routes sub-questions to "
                    "each expert, probes for mirage reasoning, and synthesises "
                    "a unified clinical answer."
                )

                gr.HTML("""
                <div class="pipeline-steps">
                    <div class="pipeline-step">
                        <span class="step-box">ECG Expert</span>
                    </div>
                    <div class="pipeline-step">
                        <span class="step-box">Echo Expert</span>
                    </div>
                    <div class="pipeline-step">
                        <span class="step-box">CMR Expert</span>
                    </div>
                    <div class="pipeline-step">
                        <span class="step-arrow">&rarr;</span>
                        <span class="step-box" style="background:#f0f4ff; border-color:#93a8d4;">Orchestrator</span>
                        <span class="step-arrow">&rarr;</span>
                        <span class="step-box" style="background:#1e3a5f; color:#fff; border-color:#1e3a5f;">Report</span>
                    </div>
                </div>
                """)

                with gr.Row():
                    multi_ecg = gr.File(label="ECG", file_types=[".png", ".jpg", ".npy", ".xml"])
                    multi_echo = gr.File(label="Echocardiogram", file_types=[".mp4", ".avi", ".tgz"])
                    multi_cmr = gr.File(label="Cardiac MRI", file_types=[".mp4", ".avi", ".tgz"])

                with gr.Row():
                    multi_ecg_prev = gr.Image(label="ECG Preview", height=160)
                    multi_echo_prev = gr.Image(label="Echo Preview", height=160)
                    multi_cmr_prev = gr.Image(label="CMR Preview", height=160)

                load_multi_btn = gr.Button("Load Example Patient", size="sm", variant="secondary")

                multi_template = gr.Dropdown(
                    choices=TEMPLATE_QUESTIONS["multimodal"],
                    value=TEMPLATE_QUESTIONS["multimodal"][0],
                    label="Template questions",
                    interactive=True,
                )
                multi_question = gr.Textbox(
                    label="Clinical question", lines=2,
                    placeholder="Ask a question requiring multimodal reasoning...",
                )
                multi_attn_cb = gr.Checkbox(label="Show attention maps", value=False)
                multi_submit = gr.Button("Run Multimodal Analysis", variant="primary")

                gr.Markdown("### Synthesised Report")
                multi_answer = gr.Textbox(
                    label="Orchestrator synthesis", lines=8, interactive=False,
                    elem_classes=["result-box"],
                )
                multi_timing = gr.HTML("")
                multi_confidence = gr.Markdown("")

                with gr.Accordion("Individual Expert Responses", open=False):
                    with gr.Row():
                        ecg_resp_box = gr.Textbox(label="ECG Expert", lines=5, interactive=False,
                                                   elem_classes=["result-box"])
                        echo_resp_box = gr.Textbox(label="Echo Expert", lines=5, interactive=False,
                                                    elem_classes=["result-box"])
                        cmr_resp_box = gr.Textbox(label="CMR Expert", lines=5, interactive=False,
                                                   elem_classes=["result-box"])

                with gr.Accordion("Attention Maps", open=False) as multi_attn_accordion:
                    with gr.Row():
                        multi_ecg_attn = gr.Image(label="ECG Attention", interactive=False)
                        multi_echo_attn = gr.Image(label="Echo Attention", interactive=False)
                        multi_cmr_attn = gr.Image(label="CMR Attention", interactive=False)

                # Wiring
                multi_template.change(on_template_select, multi_template, multi_question)

                def _load_multi():
                    e = MULTIMODAL_EXAMPLE
                    echo_gif = _video_to_gif(e["echo"]) or e["echo"]
                    cmr_gif = _video_to_gif(e["cmr"]) or e["cmr"]
                    return e["ecg"], e["echo"], e["cmr"], e["ecg"], echo_gif, cmr_gif

                load_multi_btn.click(
                    _load_multi, outputs=[
                        multi_ecg, multi_echo, multi_cmr,
                        multi_ecg_prev, multi_echo_prev, multi_cmr_prev,
                    ],
                )

                multi_submit.click(
                    run_multimodal_analysis,
                    [multi_ecg, multi_echo, multi_cmr, multi_question,
                     multi_attn_cb, multi_template],
                    [multi_answer, ecg_resp_box, echo_resp_box, cmr_resp_box,
                     multi_confidence, multi_timing,
                     multi_ecg_attn, multi_echo_attn, multi_cmr_attn],
                )

            # ════════════════════════════════════════════════════════════
            # TAB 4: Mirage Detection
            # ════════════════════════════════════════════════════════════
            with gr.Tab("Mirage Detection"):
                gr.Markdown(
                    "MARCUS detects *mirage reasoning* — the phenomenon whereby "
                    "vision-language models generate plausible clinical descriptions "
                    "without genuinely referencing the provided image or video. "
                    "The counterfactual probing protocol sends the same question "
                    "multiple times with different phrasings (with the image) and "
                    "once without the image, then compares the responses to determine "
                    "whether the model is truly grounded in visual data."
                )

                with gr.Accordion("How it works — examples", open=False):
                    gr.Markdown("""
**No mirage (grounded):** The model is asked "What is the cardiac rhythm?" about an ECG
showing atrial fibrillation. With the image, it consistently reports "irregularly irregular
rhythm, no P waves, consistent with atrial fibrillation." Without the image, it gives a
generic response: "The rhythm is sinus at 70 bpm." The responses are substantially different
(high divergence), confirming the model is referencing the actual image.

**Mirage detected:** A generic model is asked "Describe the left ventricular function" about
an echocardiogram. With and without the video, it produces nearly identical boilerplate:
"The left ventricle shows normal systolic function with an ejection fraction of 55-60%."
The low divergence between image-present and image-absent responses indicates the model is
generating plausible text from its language prior rather than interpreting the actual video.

**Scoring:**
- **Consistency** — How similar are the 3 image-present responses to each other? High = coherent.
- **Divergence** — How different are the image-present responses from the image-absent baseline? High = grounded, low = potential mirage.
- **Confidence** — Combined score (0-1). Higher is better.
""")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        mirage_modality = gr.Radio(
                            ["ECG", "Echo", "CMR"], value="ECG", label="Modality",
                        )
                        mirage_file = gr.File(
                            label="Upload study",
                            file_types=[".png", ".jpg", ".npy", ".xml", ".mp4", ".avi", ".tgz"],
                        )
                        mirage_example_dd = gr.Dropdown(
                            choices=get_example_choices("ecg"),
                            label="Or select an example",
                        )
                        mirage_load_btn = gr.Button("Load Example", size="sm", variant="secondary")
                        mirage_question = gr.Textbox(
                            label="Clinical question", lines=2,
                            value="What is the cardiac rhythm shown in this ECG?",
                        )
                        mirage_run = gr.Button("Run Mirage Probe", variant="primary")

                    with gr.Column(scale=1):
                        rephrase_display = gr.Markdown("", label="Rephrased queries")

                gr.Markdown("### Image-present responses (3 phrasings)")
                with gr.Row():
                    resp_1 = gr.Textbox(label="Response 1 (original)", lines=4,
                                         interactive=False, elem_classes=["result-box"])
                    resp_2 = gr.Textbox(label="Response 2 (rephrase)", lines=4,
                                         interactive=False, elem_classes=["result-box"])
                    resp_3 = gr.Textbox(label="Response 3 (rephrase)", lines=4,
                                         interactive=False, elem_classes=["result-box"])

                gr.Markdown("### Image-absent response (counterfactual)")
                counterfactual_box = gr.Textbox(
                    label="Response without image/video", lines=4,
                    interactive=False, elem_classes=["result-box"],
                )

                gr.Markdown("### Scoring")
                with gr.Row():
                    score_consistency = gr.Number(label="Consistency", precision=3, interactive=False)
                    score_divergence = gr.Number(label="Divergence", precision=3, interactive=False)
                    score_confidence = gr.Number(label="Confidence", precision=3, interactive=False)

                verdict_html = gr.HTML("")

                fig6 = FIGURES_DIR / "fig6_mirage.png"
                if fig6.is_file():
                    with gr.Accordion("Mirage reasoning analysis (paper figure)", open=False):
                        gr.Image(str(fig6), label="Figure 6. Mirage reasoning analysis",
                                 show_label=True, interactive=False, height=380)

                # Wiring
                def _mirage_modality_change(mod):
                    return gr.update(choices=get_example_choices(mod.lower()))

                mirage_modality.change(_mirage_modality_change, mirage_modality, mirage_example_dd)

                def _mirage_load_ex(mod, name):
                    ex = get_example(mod.lower(), name)
                    if ex and Path(ex["path"]).is_file():
                        return ex["path"], ex["default_question"]
                    return None, ""

                mirage_load_btn.click(
                    _mirage_load_ex, [mirage_modality, mirage_example_dd],
                    [mirage_file, mirage_question],
                )

                mirage_run.click(
                    run_mirage_probe,
                    [mirage_file, mirage_modality, mirage_question],
                    [rephrase_display, resp_1, resp_2, resp_3, counterfactual_box,
                     score_consistency, score_divergence, score_confidence, verdict_html],
                )

            # ════════════════════════════════════════════════════════════
            # TAB 5: About
            # ════════════════════════════════════════════════════════════
            with gr.Tab("About"):
                gr.Markdown("## Citation")
                gr.Code(
                    value=(
                        "@article{osullivan2026marcus,\n"
                        "  title   = {MARCUS: An agentic, multimodal vision-language\n"
                        "             model for cardiac diagnosis and management},\n"
                        "  author  = {O'Sullivan, Jack W and Asadi, Mohammad and\n"
                        "             Elbe, Lennart and Chaudhari, Akshay and\n"
                        "             Nedaee, Tahoura and Haddad, Francois and\n"
                        "             Salerno, Michael and Fei-Fei, Li and\n"
                        "             Adeli, Ehsan and Arnaout, Rima and\n"
                        "             Ashley, Euan A},\n"
                        "  year    = {2026}\n"
                        "}"
                    ),
                    language=None,
                    interactive=False,
                )

                gr.Markdown(
                    "**Resources:** "
                    "[GitHub](https://github.com/masadi-99/MARCUS) "
                    "&middot; [HuggingFace Models](https://huggingface.co/stanford-cardiac-ai) "
                    "&middot; [MARCUS-Benchmark](https://huggingface.co/datasets/stanford-cardiac-ai/MARCUS-Benchmark)"
                )

                fig2 = FIGURES_DIR / "fig2_performance.png"
                if fig2.is_file():
                    with gr.Accordion("Performance comparison (paper figure)", open=False):
                        gr.Image(str(fig2), label="Figure 2. Performance comparison",
                                 show_label=True, interactive=False, height=400)

                fig3 = FIGURES_DIR / "fig3_subcategory.png"
                if fig3.is_file():
                    with gr.Accordion("Per-subcategory performance (paper figure)", open=False):
                        gr.Image(str(fig3), label="Figure 3. Per-subcategory breakdown",
                                 show_label=True, interactive=False, height=400)

                fig4 = FIGURES_DIR / "fig4_external.png"
                if fig4.is_file():
                    with gr.Accordion("External validation (paper figure)", open=False):
                        gr.Image(str(fig4), label="Figure 4. Stanford vs UCSF validation",
                                 show_label=True, interactive=False, height=400)

        # ── Footer ──────────────────────────────────────────────────────
        gr.HTML(
            '<div class="disclaimer">'
            "MARCUS is a research prototype for investigational use only. "
            "It is not intended as a substitute for professional medical judgment."
            "</div>"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    import os
    # Ensure the Gradio process uses the same upload directory as the FastAPI
    # UI servers so that _resolve_media_ref can find uploaded video files locally.
    os.environ.setdefault("UPLOAD_DIR", "/tmp/marcus_uploads")

    demo = build_demo()
    theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#f0f4ff", c100="#dbe4ff", c200="#bac8ff",
            c300="#91a7ff", c400="#748ffc", c500="#4c6ef5",
            c600="#3b5bdb", c700="#1e3a5f", c800="#1a3052",
            c900="#0f1f3a", c950="#0a1628",
        ),
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
    ).set(
        body_background_fill="#ffffff",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_border_color="#e5e7eb",
        block_radius="8px",
        block_shadow="0 1px 3px rgba(0,0,0,0.04)",
        input_background_fill="#ffffff",
        input_border_color="#d1d5db",
        input_radius="6px",
        button_primary_background_fill="#1e3a5f",
        button_primary_text_color="#ffffff",
        button_primary_border_color="#1e3a5f",
        button_secondary_background_fill="#ffffff",
        button_secondary_border_color="#d1d5db",
        button_secondary_text_color="#374151",
    )
    demo.queue(max_size=10)
    # Inject a <script> in <head> that fires early — before Gradio adds body.dark.
    # A MutationObserver on <html> catches the body element as soon as it appears
    # and then watches body.classList for the "dark" class.
    _force_light_head = """
    <script>
    (function() {
        function watchBody() {
            var body = document.body;
            if (!body) return;
            body.classList.remove('dark');
            new MutationObserver(function() {
                if (body.classList.contains('dark')) body.classList.remove('dark');
            }).observe(body, {attributes: true, attributeFilter: ['class']});
        }
        if (document.body) { watchBody(); }
        new MutationObserver(function(mutations, obs) {
            if (document.body) { watchBody(); obs.disconnect(); }
        }).observe(document.documentElement, {childList: true});
    })();
    </script>
    """
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=theme,
        css=CUSTOM_CSS,
        head=_force_light_head,
        allowed_paths=[
            str(FIGURES_DIR),
            "/tmp",
            "/home/masadi/temp_ecg.png",
            "/home/masadi/temp_ecg_2.png",
            "/home/masadi/temp_echo.mp4",
            "/home/masadi/temp_cmr.mp4",
            "/home/masadi/cmr_grid.mp4",
            "/home/masadi/ecg/ecg_imgs",
            "/home/masadi/demo_data/grid_videos_small",
            "/home/masadi/videos_cmr_50",
        ],
    )


if __name__ == "__main__":
    main()
