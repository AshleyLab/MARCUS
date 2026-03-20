"""
Create a professional animated MP4 demo video showcasing the MARCUS pipeline.

Renders frames programmatically with PIL and writes to MP4 via OpenCV.
Designed for short-form social media (LinkedIn, Twitter/X).
Style: dark navy, minimalistic, elegant, smooth animations.

Usage:
    python scripts/create_demo_video.py
"""

from __future__ import annotations

import math
import os
import pickle
import re
import subprocess
import sys
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ── Resolution & timing ───────────────────────────────────────────────
W, H = 1920, 1080
FPS = 30
CROSSFADE_FRAMES = 15  # 0.5 s

# ── Color palette ─────────────────────────────────────────────────────
BG = (15, 23, 42)           # #0F172A dark navy
BG_CARD = (30, 41, 59)      # #1E293B card surfaces
ACCENT = (59, 130, 246)     # #3B82F6 blue accent
WHITE = (255, 255, 255)
WARM = (248, 250, 252)      # #F8FAFC off-white
GRAY = (148, 163, 184)      # #94A3B8 light gray
GREEN = (34, 197, 94)       # #22C55E
AMBER = (245, 158, 11)      # #F59E0B
PURPLE = (109, 40, 217)     # #6D28D9
RED_ACCENT = (239, 68, 68)  # #EF4444

# ── Fonts ─────────────────────────────────────────────────────────────
_FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_ITALIC = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"
_FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
_FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
_FONT_SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

# ── Asset paths ───────────────────────────────────────────────────────
ASSET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ECG_IMG = "/home/masadi/temp_ecg.png"
ECHO_FRAME = "/tmp/echo_frame.png"
CMR_FRAME = "/tmp/cmr_frame_new.png"
ECG_HEATMAP = "/tmp/test_ecg_heatmap_tuned.png"
FIG_ARCH = os.path.join(ASSET_DIR, "docs/figures/fig1_architecture.png")
FIG_PERF = os.path.join(ASSET_DIR, "docs/figures/fig2_performance.png")
FIG_MIRAGE = os.path.join(ASSET_DIR, "docs/figures/fig6_mirage.png")

OUTPUT_RAW = os.path.join(ASSET_DIR, "MARCUS_Demo_raw.mp4")
OUTPUT_FINAL = os.path.join(ASSET_DIR, "MARCUS_Demo.mp4")


# ═══════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════

def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def ease_in_out(t: float) -> float:
    """Cubic ease-in-out: smooth start and end."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def ease_out(t: float) -> float:
    """Cubic ease-out: fast start, smooth end."""
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def new_frame() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


def draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    fill: tuple | None = None,
    outline: tuple | None = None,
    radius: int = 16,
    width: int = 2,
):
    """Draw a rounded rectangle."""
    draw.rounded_rectangle(bbox, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(
    draw: ImageDraw.ImageDraw,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple = WHITE,
    alpha: float = 1.0,
):
    """Draw horizontally centered text. Alpha is applied via fill tuple."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (W - tw) // 2
    if alpha < 1.0:
        fill = (*fill[:3], int(255 * alpha))
    draw.text((x, y), text, font=font, fill=fill)


def load_and_fit(path: str, max_w: int, max_h: int) -> Image.Image:
    """Load image, resize to fit within bounds, maintain aspect ratio."""
    img = Image.open(path).convert("RGBA")
    ratio = min(max_w / img.width, max_h / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size, Image.LANCZOS)


def paste_centered(canvas: Image.Image, element: Image.Image, cx: int, cy: int):
    """Paste element centered at (cx, cy)."""
    x = cx - element.width // 2
    y = cy - element.height // 2
    if element.mode == "RGBA":
        canvas.paste(element, (x, y), element)
    else:
        canvas.paste(element, (x, y))


def fade_alpha(base_frame: Image.Image, overlay_frame: Image.Image, t: float) -> Image.Image:
    """Cross-fade between two frames. t=0 → base, t=1 → overlay."""
    t = max(0.0, min(1.0, t))
    base_np = np.array(base_frame, dtype=np.float32)
    over_np = np.array(overlay_frame, dtype=np.float32)
    blended = ((1 - t) * base_np + t * over_np).astype(np.uint8)
    return Image.fromarray(blended)


def draw_arrow(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int,
               color: tuple = GRAY, width: int = 3, head_size: int = 12):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    hx1 = x2 - head_size * math.cos(angle - math.pi / 6)
    hy1 = y2 - head_size * math.sin(angle - math.pi / 6)
    hx2 = x2 - head_size * math.cos(angle + math.pi / 6)
    hy2 = y2 - head_size * math.sin(angle + math.pi / 6)
    draw.polygon([(x2, y2), (int(hx1), int(hy1)), (int(hx2), int(hy2))], fill=color)


def draw_glow_circle(canvas: Image.Image, cx: int, cy: int, radius: int,
                     color: tuple, alpha: float = 0.3):
    """Draw a soft glowing circle behind an element."""
    glow = Image.new("RGBA", (radius * 4, radius * 4), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    center = radius * 2
    for r in range(radius, 0, -1):
        a = int(255 * alpha * (r / radius) ** 0.5)
        gd.ellipse(
            [center - r, center - r, center + r, center + r],
            fill=(*color[:3], a),
        )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=radius // 2))
    px = cx - radius * 2
    py = cy - radius * 2
    canvas.paste(glow, (px, py), glow)


# ═══════════════════════════════════════════════════════════════════════
#  Scene generators — each returns a list of PIL frames
# ═══════════════════════════════════════════════════════════════════════

def scene_title(duration_s: float = 4.5) -> list[Image.Image]:
    """Scene 1: Title card with animated accent line."""
    n = int(duration_s * FPS)
    frames = []
    title_font = _font(_FONT_SERIF_BOLD, 88)
    sub_font = _font(_FONT_ITALIC, 26)
    affil_font = _font(_FONT_REGULAR, 22)

    for i in range(n):
        t = i / n
        f = new_frame()
        draw = ImageDraw.Draw(f)

        # Title fades in (first 30%)
        title_a = ease_in_out(min(1.0, t / 0.3))
        title_color = tuple(int(c * title_a) for c in WHITE)
        centered_text(draw, 340, "MARCUS", title_font, fill=title_color)

        # Accent line grows (20-50%)
        if t > 0.2:
            line_t = ease_out(min(1.0, (t - 0.2) / 0.3))
            line_w = int(360 * line_t)
            cx = W // 2
            y_line = 445
            draw.line(
                [(cx - line_w // 2, y_line), (cx + line_w // 2, y_line)],
                fill=ACCENT, width=3,
            )

        # Subtitle fades in (35-65%)
        if t > 0.35:
            sub_a = ease_in_out(min(1.0, (t - 0.35) / 0.3))
            sub_color = tuple(int(c * sub_a) for c in GRAY)
            sub_text = "Multimodal Agentic Reasoning and Clinical Understanding System"
            centered_text(draw, 470, sub_text, sub_font, fill=sub_color)

        # Affiliation fades in (55-80%)
        if t > 0.55:
            aff_a = ease_in_out(min(1.0, (t - 0.55) / 0.25))
            aff_color = tuple(int(c * aff_a) for c in GRAY)
            centered_text(draw, 540, "Stanford University  |  UCSF", affil_font, fill=aff_color)

        frames.append(f)
    return frames


def scene_problem(duration_s: float = 6.0) -> list[Image.Image]:
    """Scene 2: Three cardiac modalities — ECG static, Echo/CMR as playing video."""
    n = int(duration_s * FPS)
    frames = []

    header_font = _font(_FONT_REGULAR, 16)
    title_font = _font(_FONT_SERIF_BOLD, 44)
    label_font = _font(_FONT_BOLD, 24)
    desc_font = _font(_FONT_REGULAR, 17)

    # Bigger cards
    card_w = 540
    card_h = 460
    img_max_w = card_w - 40
    img_max_h = card_h - 100
    gap = 30
    total_w = 3 * card_w + 2 * gap
    start_x = (W - total_w) // 2

    # Pre-load ECG as static RGBA
    ecg_pil = load_and_fit(ECG_IMG, img_max_w, img_max_h)

    # Pre-load Echo & CMR video frames
    echo_video = np.load("/tmp/marcus_demo_data/temp_echo_frames.npy")
    cmr_video = np.load("/tmp/marcus_demo_data/temp_cmr_frames.npy")

    def _fit_np(arr: np.ndarray) -> Image.Image:
        pil = Image.fromarray(arr).convert("RGBA")
        ratio = min(img_max_w / pil.width, img_max_h / pil.height)
        return pil.resize((int(pil.width * ratio), int(pil.height * ratio)), Image.LANCZOS)

    echo_frames = [_fit_np(f) for f in echo_video]
    cmr_frames = [_fit_np(f) for f in cmr_video]

    cards_meta = [
        ("ECG", "12-Lead Electrocardiogram", GREEN),
        ("Echo", "Echocardiography", ACCENT),
        ("CMR", "Cardiac MRI", PURPLE),
    ]

    for i in range(n):
        t = i / n
        f = new_frame()
        draw = ImageDraw.Draw(f)

        # Section label
        if t > 0.02:
            la = ease_in_out(min(1.0, (t - 0.02) / 0.15))
            lx = int(80 + 40 * (1 - la))
            lc = tuple(int(c * la) for c in GRAY)
            draw.text((lx, 60), "THE CHALLENGE", font=header_font, fill=lc)

        # Title
        if t > 0.08:
            ta = ease_in_out(min(1.0, (t - 0.08) / 0.2))
            tc = tuple(int(c * ta) for c in WHITE)
            centered_text(draw, 100, "Three Modalities. One Patient.", title_font, fill=tc)

        # Cards slide up sequentially
        for ci, (name, desc, color) in enumerate(cards_meta):
            card_start = 0.18 + ci * 0.1
            if t < card_start:
                continue
            ct = ease_out(min(1.0, (t - card_start) / 0.25))
            cx = start_x + ci * (card_w + gap)
            cy_target = 190
            cy = int(cy_target + 80 * (1 - ct))
            alpha = ct

            # Card background
            draw_rounded_rect(
                draw,
                (cx, cy, cx + card_w, cy + card_h),
                fill=tuple(int(c * alpha) for c in BG_CARD),
                outline=tuple(int(c * alpha) for c in color),
                radius=12, width=3,
            )

            # Left color bar
            bar_x = cx + 8
            draw.rounded_rectangle(
                (bar_x, cy + 12, bar_x + 4, cy + card_h - 12),
                radius=2,
                fill=tuple(int(c * alpha) for c in color),
            )

            # Pick the right image/frame
            if ci == 0:
                cur_img = ecg_pil
            elif ci == 1:
                # Cycle echo video
                anim_t = max(0.0, (t - card_start - 0.25) / (1.0 - card_start - 0.25))
                vf_idx = int((anim_t * 3.0) % 1.0 * len(echo_frames)) % len(echo_frames)
                cur_img = echo_frames[vf_idx] if anim_t > 0 else echo_frames[0]
            else:
                # Cycle CMR video
                anim_t = max(0.0, (t - card_start - 0.25) / (1.0 - card_start - 0.25))
                vf_idx = int((anim_t * 3.0) % 1.0 * len(cmr_frames)) % len(cmr_frames)
                cur_img = cmr_frames[vf_idx] if anim_t > 0 else cmr_frames[0]

            # Paste image with alpha
            temp_img = cur_img.copy()
            if alpha < 1.0:
                r, g, b, a = temp_img.split()
                a = a.point(lambda p: int(p * alpha))
                temp_img = Image.merge("RGBA", (r, g, b, a))
            img_x = cx + (card_w - cur_img.width) // 2
            img_y = cy + 20
            f.paste(temp_img, (img_x, img_y), temp_img)

            # Labels
            label_color = tuple(int(c * alpha) for c in WHITE)
            desc_color = tuple(int(c * alpha) for c in GRAY)
            lbbox = draw.textbbox((0, 0), name, font=label_font)
            lw = lbbox[2] - lbbox[0]
            draw.text((cx + (card_w - lw) // 2, cy + card_h - 70), name, font=label_font, fill=label_color)
            dbbox = draw.textbbox((0, 0), desc, font=desc_font)
            dw = dbbox[2] - dbbox[0]
            draw.text((cx + (card_w - dw) // 2, cy + card_h - 42), desc, font=desc_font, fill=desc_color)

        frames.append(f)
    return frames


def scene_scale(duration_s: float = 4.0) -> list[Image.Image]:
    """Scene 3: Animated counters showing training scale."""
    n = int(duration_s * FPS)
    frames = []

    header_font = _font(_FONT_REGULAR, 16)
    num_font = _font(_FONT_MONO, 72)
    label_font = _font(_FONT_REGULAR, 22)
    plus_font = _font(_FONT_SERIF_BOLD, 36)

    stats = [
        ("13.5M", "Training Images", GREEN),
        ("879K", "Diagnostic Questions", ACCENT),
        ("3", "Expert Models", PURPLE),
    ]
    gap = 100
    col_w = 400
    total = 3 * col_w + 2 * gap
    sx = (W - total) // 2

    for i in range(n):
        t = i / n
        f = new_frame()
        draw = ImageDraw.Draw(f)

        # Section label
        if t > 0.02:
            la = ease_in_out(min(1.0, (t - 0.02) / 0.15))
            lc = tuple(int(c * la) for c in GRAY)
            draw.text((80, 60), "TRAINING SCALE", font=header_font, fill=lc)

        for si, (value, label, color) in enumerate(stats):
            stat_start = 0.15 + si * 0.08
            if t < stat_start:
                continue
            st = ease_out(min(1.0, (t - stat_start) / 0.35))

            cx = sx + si * (col_w + gap) + col_w // 2
            cy = 420

            # Animated counter
            if "M" in value:
                num_val = float(value.replace("M", ""))
                display = f"{num_val * st:.1f}M"
            elif "K" in value:
                num_val = float(value.replace("K", ""))
                display = f"{int(num_val * st):,}K".replace(",", ",")
                if st >= 0.99:
                    display = "879K"
            else:
                display = str(int(float(value) * st)) if st < 0.99 else value

            num_color = tuple(int(c * min(1.0, st * 1.2)) for c in WHITE)
            bbox = draw.textbbox((0, 0), display, font=num_font)
            nw = bbox[2] - bbox[0]
            draw.text((cx - nw // 2, cy - 40), display, font=num_font, fill=num_color)

            # Accent line under number
            line_w = int(120 * st)
            draw.line(
                [(cx - line_w // 2, cy + 50), (cx + line_w // 2, cy + 50)],
                fill=tuple(int(c * st) for c in color),
                width=3,
            )

            # Label
            label_color = tuple(int(c * st) for c in GRAY)
            lbbox = draw.textbbox((0, 0), label, font=label_font)
            lw = lbbox[2] - lbbox[0]
            draw.text((cx - lw // 2, cy + 70), label, font=label_font, fill=label_color)

        frames.append(f)
    return frames


def scene_architecture(duration_s: float = 7.0) -> list[Image.Image]:
    """Scene 4: Expert routing architecture with animated arrows."""
    n = int(duration_s * FPS)
    frames = []

    header_font = _font(_FONT_REGULAR, 16)
    title_font = _font(_FONT_SERIF_BOLD, 38)
    box_font = _font(_FONT_BOLD, 20)
    small_font = _font(_FONT_REGULAR, 15)

    # Layout positions
    col1_x = 160       # Patient Data
    col2_x = 520       # Experts
    col3_x = 960       # Mirage Probe
    col4_x = 1380      # Orchestrator
    box_w = 280
    box_h = 70

    experts = [
        ("ECG Expert", GREEN, 320),
        ("Echo Expert", ACCENT, 440),
        ("CMR Expert", PURPLE, 560),
    ]

    for i in range(n):
        t = i / n
        f = new_frame()
        draw = ImageDraw.Draw(f)

        # Section label + title
        if t > 0.02:
            la = ease_in_out(min(1.0, (t - 0.02) / 0.12))
            lc = tuple(int(c * la) for c in GRAY)
            draw.text((80, 60), "AGENTIC ARCHITECTURE", font=header_font, fill=lc)
        if t > 0.05:
            ta = ease_in_out(min(1.0, (t - 0.05) / 0.15))
            tc = tuple(int(c * ta) for c in WHITE)
            centered_text(draw, 110, "Intelligent Expert Routing", title_font, fill=tc)

        # Patient Data box (appears at 15%)
        if t > 0.15:
            ba = ease_out(min(1.0, (t - 0.15) / 0.15))
            bc = tuple(int(c * ba) for c in BG_CARD)
            oc = tuple(int(c * ba) for c in WARM)
            draw_rounded_rect(draw, (col1_x, 400, col1_x + box_w, 400 + box_h), fill=bc, outline=oc, radius=10)
            tc = tuple(int(c * ba) for c in WHITE)
            bbox = draw.textbbox((0, 0), "Patient Data", font=box_font)
            tw = bbox[2] - bbox[0]
            draw.text((col1_x + (box_w - tw) // 2, 420), "Patient Data", font=box_font, fill=tc)

        # Arrows to experts (25-40%)
        for ei, (ename, ecolor, ey) in enumerate(experts):
            arrow_start = 0.25 + ei * 0.04
            if t > arrow_start:
                at = ease_out(min(1.0, (t - arrow_start) / 0.12))
                ax2 = int(col1_x + box_w + (col2_x - col1_x - box_w) * at)
                ac = tuple(int(c * at) for c in GRAY)
                draw_arrow(draw, col1_x + box_w, 435, min(ax2, col2_x), ey + box_h // 2, color=ac, width=2)

        # Expert boxes (30-50%)
        for ei, (ename, ecolor, ey) in enumerate(experts):
            ebox_start = 0.30 + ei * 0.06
            if t > ebox_start:
                et = ease_out(min(1.0, (t - ebox_start) / 0.15))
                bc = tuple(int(c * et) for c in BG_CARD)
                oc = tuple(int(c * et) for c in ecolor)
                draw_rounded_rect(draw, (col2_x, ey, col2_x + box_w, ey + box_h), fill=bc, outline=oc, radius=10, width=3)
                tc = tuple(int(c * et) for c in WHITE)
                bbox = draw.textbbox((0, 0), ename, font=box_font)
                tw = bbox[2] - bbox[0]
                draw.text((col2_x + (box_w - tw) // 2, ey + 22), ename, font=box_font, fill=tc)

                # Small clinical description beneath
                sub = ["Rhythm  ·  Conduction  ·  Intervals", "Structure  ·  Function  ·  Valves", "Tissue  ·  Perfusion  ·  Scar"][ei]
                sc = tuple(int(c * et * 0.7) for c in GRAY)
                sbbox = draw.textbbox((0, 0), sub, font=small_font)
                sw = sbbox[2] - sbbox[0]
                draw.text((col2_x + (box_w - sw) // 2, ey + box_h + 4), sub, font=small_font, fill=sc)

        # Arrows to Mirage (50-60%)
        if t > 0.50:
            at2 = ease_out(min(1.0, (t - 0.50) / 0.1))
            for _, _, ey in experts:
                ax2 = int(col2_x + box_w + (col3_x - col2_x - box_w) * at2)
                ac = tuple(int(c * at2) for c in GRAY)
                draw_arrow(draw, col2_x + box_w, ey + box_h // 2, min(ax2, col3_x), 435, color=ac, width=2)

        # Mirage Probe box (55%)
        if t > 0.55:
            mt = ease_out(min(1.0, (t - 0.55) / 0.15))
            bc = tuple(int(c * mt) for c in BG_CARD)
            oc = tuple(int(c * mt) for c in AMBER)
            draw_rounded_rect(draw, (col3_x, 400, col3_x + box_w, 400 + box_h), fill=bc, outline=oc, radius=10, width=3)
            tc = tuple(int(c * mt) for c in WHITE)
            txt = "Mirage Probe"
            bbox = draw.textbbox((0, 0), txt, font=box_font)
            tw = bbox[2] - bbox[0]
            draw.text((col3_x + (box_w - tw) // 2, 420), txt, font=box_font, fill=tc)

            # Subtitle
            sub = "Counterfactual Verification"
            sc = tuple(int(c * mt * 0.7) for c in GRAY)
            sbbox = draw.textbbox((0, 0), sub, font=small_font)
            sw = sbbox[2] - sbbox[0]
            draw.text((col3_x + (box_w - sw) // 2, 475), sub, font=small_font, fill=sc)

        # Arrow to Orchestrator (65%)
        if t > 0.65:
            at3 = ease_out(min(1.0, (t - 0.65) / 0.1))
            ax2 = int(col3_x + box_w + (col4_x - col3_x - box_w) * at3)
            ac = tuple(int(c * at3) for c in GRAY)
            draw_arrow(draw, col3_x + box_w, 435, min(ax2, col4_x), 435, color=ac, width=2)

        # Orchestrator box (70%)
        if t > 0.70:
            ot = ease_out(min(1.0, (t - 0.70) / 0.15))
            bc = tuple(int(c * ot) for c in BG_CARD)
            oc = tuple(int(c * ot) for c in ACCENT)
            draw_rounded_rect(draw, (col4_x, 400, col4_x + box_w, 400 + box_h), fill=bc, outline=oc, radius=10, width=3)
            tc = tuple(int(c * ot) for c in WHITE)
            txt = "Orchestrator"
            bbox = draw.textbbox((0, 0), txt, font=box_font)
            tw = bbox[2] - bbox[0]
            draw.text((col4_x + (box_w - tw) // 2, 420), txt, font=box_font, fill=tc)

            sub = "Multi-Expert Synthesis"
            sc = tuple(int(c * ot * 0.7) for c in GRAY)
            sbbox = draw.textbbox((0, 0), sub, font=small_font)
            sw = sbbox[2] - sbbox[0]
            draw.text((col4_x + (box_w - sw) // 2, 475), sub, font=small_font, fill=sc)

        # Final output arrow + label (80%)
        if t > 0.80:
            ft = ease_out(min(1.0, (t - 0.80) / 0.15))
            # Output label below orchestrator
            out_y = 530
            bc = tuple(int(c * ft) for c in BG_CARD)
            oc = tuple(int(c * ft) for c in GREEN)
            out_x = col4_x
            draw_rounded_rect(draw, (out_x, out_y, out_x + box_w, out_y + 60), fill=bc, outline=oc, radius=10, width=2)
            tc = tuple(int(c * ft) for c in WHITE)
            txt = "Clinical Report"
            bbox = draw.textbbox((0, 0), txt, font=box_font)
            tw = bbox[2] - bbox[0]
            draw.text((out_x + (box_w - tw) // 2, out_y + 17), txt, font=box_font, fill=tc)
            # Down arrow
            ac = tuple(int(c * ft) for c in GREEN)
            draw_arrow(draw, col4_x + box_w // 2, 400 + box_h, col4_x + box_w // 2, out_y, color=ac, width=2)

        frames.append(f)
    return frames


def _load_modality_data(mod: str) -> dict:
    """Load pre-rendered attention data for a modality, stripping EOS tokens."""
    path = f"/tmp/marcus_demo_data/{mod}_rendered.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Remove end-of-text / special tokens from the end
    tokens = data["tokens"]
    heatmaps = data["heatmaps"]
    while tokens and tokens[-1].strip() in ("", "<|im_end|>", "<|endoftext|>", "</s>"):
        tokens.pop()
        if heatmaps:
            heatmaps.pop()
    data["tokens"] = tokens
    data["heatmaps"] = heatmaps
    return data


def _render_attention_subscene(
    mod_name: str,
    mod_label: str,
    mod_color: tuple,
    data: dict,
    duration_s: float,
    show_header: bool = True,
) -> list[Image.Image]:
    """Render a sub-scene for one modality showing input + live attention.

    Left: input media playing (video cycles frames, static shows image).
    Right: attention heatmap updating per generated token.
    Bottom: response text appearing token by token.
    """
    n = int(duration_s * FPS)
    frames_out: list[Image.Image] = []

    header_font = _font(_FONT_REGULAR, 16)
    title_font = _font(_FONT_SERIF_BOLD, 38)
    label_font = _font(_FONT_BOLD, 20)
    token_font = _font(_FONT_REGULAR, 17)
    mod_font = _font(_FONT_BOLD, 16)

    heatmaps = data["heatmaps"]       # list of numpy RGB arrays
    video_frames = data["video_frames"]  # list of numpy RGB arrays
    tokens = data["tokens"]
    response = data.get("response", "")
    n_tokens = len(heatmaps)
    n_vframes = len(video_frames)
    is_video = n_vframes > 1

    # Layout
    img_max_w, img_max_h = 560, 400
    left_cx = W // 2 - 340
    right_cx = W // 2 + 340
    img_cy = 430

    # Pre-resize all frames to consistent size
    def fit_frame(arr: np.ndarray) -> Image.Image:
        pil = Image.fromarray(arr).convert("RGBA")
        ratio = min(img_max_w / pil.width, img_max_h / pil.height)
        new_size = (int(pil.width * ratio), int(pil.height * ratio))
        return pil.resize(new_size, Image.LANCZOS)

    resized_video = [fit_frame(vf) for vf in video_frames]
    resized_heatmaps = [fit_frame(hm) for hm in heatmaps]

    # Consistent card size based on resized dimensions
    card_w = resized_video[0].width + 40
    card_h = resized_video[0].height + 70

    # Strip <think>...</think> from response for display
    import re
    clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    for i in range(n):
        t = i / n
        f = new_frame()
        draw = ImageDraw.Draw(f)

        # Header (persistent across sub-scenes if show_header)
        if show_header:
            if t > 0.02:
                la = ease_in_out(min(1.0, (t - 0.02) / 0.1))
                lc = tuple(int(c * la) for c in GRAY)
                draw.text((80, 60), "ATTENTION VISUALIZATION", font=header_font, fill=lc)
            if t > 0.04:
                ta = ease_in_out(min(1.0, (t - 0.04) / 0.12))
                tc = tuple(int(c * ta) for c in WHITE)
                centered_text(draw, 100, "Where the Model Looks", title_font, fill=tc)
        else:
            # Just show header immediately (already faded in during ECG sub-scene)
            draw.text((80, 60), "ATTENTION VISUALIZATION", font=header_font, fill=GRAY)
            centered_text(draw, 100, "Where the Model Looks", title_font, fill=WHITE)

        # Modality badge
        badge_start = 0.08 if show_header else 0.02
        if t > badge_start:
            bt = ease_out(min(1.0, (t - badge_start) / 0.1))
            bc = tuple(int(c * bt) for c in mod_color)
            badge_text = f"  {mod_label}  "
            bbx = draw.textbbox((0, 0), badge_text, font=mod_font)
            bw = bbx[2] - bbx[0]
            bx = (W - bw) // 2
            by = 160
            draw_rounded_rect(draw, (bx - 8, by - 4, bx + bw + 8, by + 24),
                              fill=tuple(int(c * bt * 0.3) for c in mod_color),
                              outline=bc, radius=6, width=2)
            draw.text((bx, by - 2), badge_text, font=mod_font, fill=bc)

        # Cards fade in
        card_start = 0.12 if show_header else 0.05
        if t > card_start:
            ct = ease_out(min(1.0, (t - card_start) / 0.15))

            # Left card: Input
            lcx0 = left_cx - card_w // 2
            lcy0 = img_cy - card_h // 2
            draw_rounded_rect(draw, (lcx0, lcy0, lcx0 + card_w, lcy0 + card_h),
                              fill=tuple(int(c * ct) for c in BG_CARD),
                              outline=tuple(int(c * ct) for c in GRAY),
                              radius=12)
            # Left label
            ll = "Input" + (" (video)" if is_video else "")
            llc = tuple(int(c * ct) for c in WHITE)
            llbx = draw.textbbox((0, 0), ll, font=label_font)
            llw = llbx[2] - llbx[0]
            draw.text((left_cx - llw // 2, lcy0 + card_h - 32), ll, font=label_font, fill=llc)

            # Right card: Attention
            rcx0 = right_cx - card_w // 2
            rcy0 = img_cy - card_h // 2
            draw_rounded_rect(draw, (rcx0, rcy0, rcx0 + card_w, rcy0 + card_h),
                              fill=tuple(int(c * ct) for c in BG_CARD),
                              outline=tuple(int(c * ct) for c in mod_color),
                              radius=12, width=3)
            rl = "Attention Map"
            rlc = tuple(int(c * ct) for c in WHITE)
            rlbx = draw.textbbox((0, 0), rl, font=label_font)
            rlw = rlbx[2] - rlbx[0]
            draw.text((right_cx - rlw // 2, rcy0 + card_h - 32), rl, font=label_font, fill=rlc)

            # Arrow between cards
            mid_x = W // 2
            ac = tuple(int(c * ct) for c in ACCENT)
            draw_arrow(draw, left_cx + card_w // 2 + 10, img_cy,
                       right_cx - card_w // 2 - 10, img_cy,
                       color=ac, width=3, head_size=12)

        # Animated content: video frames + attention heatmaps cycling
        anim_start = 0.25 if show_header else 0.15
        if t > anim_start:
            anim_t = (t - anim_start) / (1.0 - anim_start)  # 0→1 over remaining time

            # Current token index (progress through tokens)
            token_idx = min(int(anim_t * n_tokens), n_tokens - 1)

            # Video frame: cycle through frames (faster than tokens for smooth video feel)
            if is_video:
                # Cycle video frames continuously for smooth playback
                video_cycle_speed = 3.0  # complete cycles
                vf_t = (anim_t * video_cycle_speed) % 1.0
                vf_idx = int(vf_t * n_vframes) % n_vframes
            else:
                vf_idx = 0

            # Paste video frame (left)
            vf_img = resized_video[vf_idx]
            paste_centered(f, vf_img, left_cx, img_cy - 15)

            # Paste heatmap (right) — updates per token
            hm_img = resized_heatmaps[token_idx]
            paste_centered(f, hm_img, right_cx, img_cy - 15)

            # Token text appearing below (typewriter)
            text_y = img_cy + card_h // 2 + 20
            visible_tokens = tokens[:token_idx + 1]
            visible_text = "".join(visible_tokens)
            # Strip think tags
            visible_text = re.sub(r"<think>.*?</think>", "", visible_text, flags=re.DOTALL).strip()
            visible_text = re.sub(r"<think>.*", "", visible_text, flags=re.DOTALL).strip()
            if visible_text:
                # Truncate to fit width
                max_chars = 120
                if len(visible_text) > max_chars:
                    visible_text = "..." + visible_text[-(max_chars - 3):]
                tc = tuple(int(c * 0.9) for c in WARM)
                # Word wrap
                words = visible_text.split(" ")
                lines = []
                current = ""
                for word in words:
                    test = current + " " + word if current else word
                    bbox = draw.textbbox((0, 0), test, font=token_font)
                    if bbox[2] - bbox[0] > W - 300:
                        lines.append(current)
                        current = word
                    else:
                        current = test
                if current:
                    lines.append(current)
                for li, line in enumerate(lines[:3]):
                    lbx = draw.textbbox((0, 0), line, font=token_font)
                    lw = lbx[2] - lbx[0]
                    draw.text(((W - lw) // 2, text_y + li * 24), line, font=token_font, fill=tc)


        frames_out.append(f)
    return frames_out


def scene_attention() -> list[Image.Image]:
    """Scene 5: Attention visualization — all 3 modalities with live attention.

    ECG: static image + attention heatmap evolving per token.
    Echo: video playing (heart beating) + attention updating per token.
    CMR: video playing + attention updating per token.
    """
    all_frames: list[Image.Image] = []

    modalities = [
        ("ecg", "ECG  —  12-Lead Electrocardiogram", GREEN, 5.0, True),
        ("echo", "Echo  —  Echocardiography", ACCENT, 6.0, False),
        ("cmr", "CMR  —  Cardiac MRI", PURPLE, 6.0, False),
    ]

    for mod_key, mod_label, mod_color, dur, show_header in modalities:
        data = _load_modality_data(mod_key)
        sub_frames = _render_attention_subscene(
            mod_key, mod_label, mod_color, data, dur, show_header,
        )

        # Cross-fade between sub-scenes
        if all_frames and CROSSFADE_FRAMES > 0:
            n_cf = min(CROSSFADE_FRAMES, len(all_frames), len(sub_frames))
            for ci in range(n_cf):
                blend_t = (ci + 1) / (n_cf + 1)
                blended = fade_alpha(all_frames[-(n_cf - ci)], sub_frames[ci], blend_t)
                all_frames[-(n_cf - ci)] = blended
            all_frames.extend(sub_frames[n_cf:])
        else:
            all_frames.extend(sub_frames)

    return all_frames


def scene_mirage(duration_s: float = 6.5) -> list[Image.Image]:
    """Scene 6: Mirage detection — hallucination prevention."""
    n = int(duration_s * FPS)
    frames = []

    header_font = _font(_FONT_REGULAR, 16)
    title_font = _font(_FONT_SERIF_BOLD, 38)
    box_font = _font(_FONT_BOLD, 20)
    small_font = _font(_FONT_REGULAR, 17)
    num_font = _font(_FONT_MONO, 56)
    pct_font = _font(_FONT_BOLD, 18)

    for i in range(n):
        t = i / n
        f = new_frame()
        draw = ImageDraw.Draw(f)

        # Header
        if t > 0.02:
            la = ease_in_out(min(1.0, (t - 0.02) / 0.12))
            lc = tuple(int(c * la) for c in GRAY)
            draw.text((80, 60), "MIRAGE DETECTION", font=header_font, fill=lc)
        if t > 0.05:
            ta = ease_in_out(min(1.0, (t - 0.05) / 0.15))
            tc = tuple(int(c * ta) for c in WHITE)
            centered_text(draw, 110, "Eliminating Hallucinations", title_font, fill=tc)

        # Subtitle
        if t > 0.12:
            st = ease_in_out(min(1.0, (t - 0.12) / 0.15))
            sc = tuple(int(c * st) for c in GRAY)
            centered_text(draw, 170, "Counterfactual probing ensures the model truly sees the image", small_font, fill=sc)

        # Left panel: WITH IMAGE (green)
        left_x, panel_y = 140, 240
        panel_w, panel_h = 740, 340
        if t > 0.20:
            pt = ease_out(min(1.0, (t - 0.20) / 0.2))
            bc = tuple(int(c * pt) for c in BG_CARD)
            oc = tuple(int(c * pt) for c in GREEN)
            draw_rounded_rect(draw, (left_x, panel_y, left_x + panel_w, panel_y + panel_h),
                              fill=bc, outline=oc, radius=12, width=2)
            lc = tuple(int(c * pt) for c in GREEN)
            draw.text((left_x + 20, panel_y + 15), "WITH IMAGE", font=box_font, fill=lc)

            # Three query rows
            queries = [
                "Query 1 → Specific, consistent response",
                "Query 2 → Detailed, grounded analysis",
                "Query 3 → Clinically accurate finding",
            ]
            for qi, qtxt in enumerate(queries):
                qs = 0.28 + qi * 0.06
                if t > qs:
                    qt = ease_out(min(1.0, (t - qs) / 0.12))
                    qy = panel_y + 65 + qi * 85
                    qc_bg = tuple(int(c * qt * 0.4) for c in GREEN)
                    draw_rounded_rect(draw, (left_x + 20, qy, left_x + panel_w - 20, qy + 65),
                                      fill=tuple(int(c * qt) for c in (20, 40, 30)),
                                      outline=tuple(int(c * qt * 0.5) for c in GREEN),
                                      radius=8, width=1)
                    tc = tuple(int(c * qt) for c in WARM)
                    draw.text((left_x + 40, qy + 20), qtxt, font=small_font, fill=tc)

        # Right panel: WITHOUT IMAGE (amber)
        right_x = 940
        if t > 0.35:
            pt = ease_out(min(1.0, (t - 0.35) / 0.2))
            bc = tuple(int(c * pt) for c in BG_CARD)
            oc = tuple(int(c * pt) for c in AMBER)
            draw_rounded_rect(draw, (right_x, panel_y, right_x + panel_w, panel_y + panel_h),
                              fill=bc, outline=oc, radius=12, width=2)
            lc = tuple(int(c * pt) for c in AMBER)
            draw.text((right_x + 20, panel_y + 15), "WITHOUT IMAGE", font=box_font, fill=lc)

            # Single generic response
            if t > 0.42:
                qt = ease_out(min(1.0, (t - 0.42) / 0.12))
                qy = panel_y + 65
                draw_rounded_rect(draw, (right_x + 20, qy, right_x + panel_w - 20, qy + 65),
                                  fill=tuple(int(c * qt) for c in (40, 35, 20)),
                                  outline=tuple(int(c * qt * 0.5) for c in AMBER),
                                  radius=8, width=1)
                tc = tuple(int(c * qt) for c in WARM)
                draw.text((right_x + 40, qy + 20), "Same query → Generic, vague response", font=small_font, fill=tc)

            # Divergence detected label
            if t > 0.50:
                dt = ease_in_out(min(1.0, (t - 0.50) / 0.12))
                dy = panel_y + 200
                dc = tuple(int(c * dt) for c in GREEN)
                draw.text((right_x + 40, dy + 20), "Divergence detected → NOT a hallucination", font=small_font, fill=dc)

        # Bottom comparison: MARCUS 0% vs others
        if t > 0.60:
            bt = ease_out(min(1.0, (t - 0.60) / 0.2))
            bar_y = 640
            models = [
                ("MARCUS", "0%", GREEN, 0),
                ("GPT-5 (Thinking)", "38%", RED_ACCENT, 0.38),
                ("Gemini 2.5 Pro (Thinking)", "35%", AMBER, 0.35),
            ]
            bar_total_w = 1400
            bar_sx = (W - bar_total_w) // 2

            for mi, (mname, mpct, mcolor, mval) in enumerate(models):
                mx = bar_sx + mi * 500
                # Model name
                nc = tuple(int(c * bt) for c in WHITE)
                draw.text((mx, bar_y), mname, font=box_font, fill=nc)
                # Percentage
                pc = tuple(int(c * bt) for c in mcolor)
                draw.text((mx, bar_y + 35), mpct, font=num_font, fill=pc)
                # Label
                lbl = "Mirage Rate"
                lc2 = tuple(int(c * bt) for c in GRAY)
                draw.text((mx, bar_y + 100), lbl, font=pct_font, fill=lc2)
                # Bar
                max_bar = 350
                bar_w = int(max_bar * mval * bt)
                bar_x = mx + 150
                if bar_w > 0:
                    draw.rounded_rectangle(
                        (bar_x, bar_y + 50, bar_x + bar_w, bar_y + 75),
                        radius=4,
                        fill=tuple(int(c * bt) for c in mcolor),
                    )

        frames.append(f)
    return frames


def scene_orchestration(duration_s: float = 6.5) -> list[Image.Image]:
    """Scene 7: Multimodal orchestration with typewriter synthesis."""
    n = int(duration_s * FPS)
    frames = []

    header_font = _font(_FONT_REGULAR, 16)
    title_font = _font(_FONT_SERIF_BOLD, 38)
    box_font = _font(_FONT_BOLD, 18)
    finding_font = _font(_FONT_REGULAR, 16)
    synth_font = _font(_FONT_REGULAR, 17)

    findings = [
        ("ECG Expert", GREEN, "Atrial fibrillation with rapid ventricular response,\nleft axis deviation, and ST-segment depression in V4-V6"),
        ("Echo Expert", ACCENT, "Reduced LV systolic function (EF 35%),\nmoderate mitral regurgitation, LA dilatation"),
        ("CMR Expert", PURPLE, "Late gadolinium enhancement in inferolateral wall,\nconsistent with prior myocardial infarction"),
    ]

    synth_text = (
        "Integrated Assessment: This patient presents with atrial fibrillation "
        "and reduced ejection fraction (35%) in the setting of prior inferolateral "
        "myocardial infarction with scar. The combination of AF, reduced EF, and "
        "structural abnormalities suggests an ischemic cardiomyopathy with secondary "
        "atrial arrhythmia. Recommend guideline-directed medical therapy."
    )

    card_w = 520
    card_h = 110
    card_x = (W - card_w) // 2

    for i in range(n):
        t = i / n
        f = new_frame()
        draw = ImageDraw.Draw(f)

        # Header
        if t > 0.02:
            la = ease_in_out(min(1.0, (t - 0.02) / 0.12))
            lc = tuple(int(c * la) for c in GRAY)
            draw.text((80, 60), "MULTIMODAL SYNTHESIS", font=header_font, fill=lc)
        if t > 0.05:
            ta = ease_in_out(min(1.0, (t - 0.05) / 0.15))
            tc = tuple(int(c * ta) for c in WHITE)
            centered_text(draw, 110, "Expert Findings → Unified Report", title_font, fill=tc)

        # Expert finding cards
        for fi, (fname, fcolor, ftext) in enumerate(findings):
            fs = 0.12 + fi * 0.08
            if t < fs:
                continue
            ft2 = ease_out(min(1.0, (t - fs) / 0.18))
            cy = 200 + fi * (card_h + 20)

            # Slide in from left
            x_offset = int(-200 * (1 - ft2))
            cx0 = card_x + x_offset

            bc = tuple(int(c * ft2) for c in BG_CARD)
            oc = tuple(int(c * ft2) for c in fcolor)
            draw_rounded_rect(draw, (cx0, cy, cx0 + card_w, cy + card_h),
                              fill=bc, outline=oc, radius=10, width=2)

            # Left color bar
            draw.rounded_rectangle(
                (cx0 + 8, cy + 10, cx0 + 12, cy + card_h - 10),
                radius=2,
                fill=tuple(int(c * ft2) for c in fcolor),
            )

            # Name and finding
            nc = tuple(int(c * ft2) for c in fcolor)
            draw.text((cx0 + 25, cy + 10), fname, font=box_font, fill=nc)
            fc = tuple(int(c * ft2) for c in WARM)
            draw.text((cx0 + 25, cy + 38), ftext, font=finding_font, fill=fc)

        # Down arrows
        if t > 0.40:
            at = ease_out(min(1.0, (t - 0.40) / 0.1))
            arrow_y1 = 200 + 3 * (card_h + 20) - 10
            arrow_y2 = arrow_y1 + 50
            ac = tuple(int(c * at) for c in ACCENT)
            draw_arrow(draw, W // 2, arrow_y1, W // 2, int(arrow_y1 + 50 * at), color=ac, width=3, head_size=14)

        # Synthesis box with typewriter effect
        if t > 0.48:
            st = ease_out(min(1.0, (t - 0.48) / 0.12))
            synth_y = 610
            synth_h = 180
            synth_w = 800
            sx = (W - synth_w) // 2

            bc = tuple(int(c * st) for c in BG_CARD)
            oc = tuple(int(c * st) for c in ACCENT)
            draw_rounded_rect(draw, (sx, synth_y, sx + synth_w, synth_y + synth_h),
                              fill=bc, outline=oc, radius=12, width=3)

            # Label
            lc = tuple(int(c * st) for c in ACCENT)
            draw.text((sx + 20, synth_y + 12), "Orchestrator Synthesis", font=box_font, fill=lc)

            # Typewriter text
            if t > 0.55:
                type_t = min(1.0, (t - 0.55) / 0.40)
                n_chars = int(len(synth_text) * type_t)
                visible = synth_text[:n_chars]
                tc = tuple(int(c * st) for c in WARM)
                # Word wrap manually
                words = visible.split(" ")
                lines = []
                current = ""
                for word in words:
                    test = current + " " + word if current else word
                    bbox = draw.textbbox((0, 0), test, font=synth_font)
                    if bbox[2] - bbox[0] > synth_w - 50:
                        lines.append(current)
                        current = word
                    else:
                        current = test
                if current:
                    lines.append(current)
                for li, line in enumerate(lines[:6]):
                    draw.text((sx + 20, synth_y + 42 + li * 22), line, font=synth_font, fill=tc)

        frames.append(f)
    return frames


def scene_performance(duration_s: float = 5.5) -> list[Image.Image]:
    """Scene 8: Performance results with paper figure."""
    n = int(duration_s * FPS)
    frames = []

    header_font = _font(_FONT_REGULAR, 16)
    title_font = _font(_FONT_SERIF_BOLD, 38)
    callout_font = _font(_FONT_BOLD, 24)
    desc_font = _font(_FONT_REGULAR, 18)

    perf_img = load_and_fit(FIG_PERF, 1400, 650)

    for i in range(n):
        t = i / n
        f = new_frame()
        draw = ImageDraw.Draw(f)

        # Header
        if t > 0.02:
            la = ease_in_out(min(1.0, (t - 0.02) / 0.12))
            lc = tuple(int(c * la) for c in GRAY)
            draw.text((80, 60), "PERFORMANCE", font=header_font, fill=lc)
        if t > 0.05:
            ta = ease_in_out(min(1.0, (t - 0.05) / 0.15))
            tc = tuple(int(c * ta) for c in WHITE)
            centered_text(draw, 100, "Benchmark Results", title_font, fill=tc)

        # Performance figure
        if t > 0.15:
            ft = ease_out(min(1.0, (t - 0.15) / 0.25))
            temp = perf_img.copy()
            if ft < 1.0:
                r, g, b, a = temp.split()
                a = a.point(lambda p: int(p * ft))
                temp = Image.merge("RGBA", (r, g, b, a))
            paste_centered(f, temp, W // 2, 470)

        # Callout text
        if t > 0.55:
            ct = ease_in_out(min(1.0, (t - 0.55) / 0.2))
            cc = tuple(int(c * ct) for c in GREEN)
            centered_text(draw, 830, "Outperforms GPT-5 (Thinking) and Gemini 2.5 Pro (Thinking)", callout_font, fill=cc)

        if t > 0.70:
            dt = ease_in_out(min(1.0, (t - 0.70) / 0.2))
            dc = tuple(int(c * dt) for c in GRAY)
            centered_text(draw, 870, "Evaluated on 2,041 Stanford + 536 UCSF studies", desc_font, fill=dc)

        frames.append(f)
    return frames


def scene_closing(duration_s: float = 4.5) -> list[Image.Image]:
    """Scene 9: Closing card — mirror of title with additions."""
    n = int(duration_s * FPS)
    frames = []

    title_font = _font(_FONT_SERIF_BOLD, 88)
    sub_font = _font(_FONT_ITALIC, 24)
    affil_font = _font(_FONT_REGULAR, 22)
    link_font = _font(_FONT_MONO, 20)

    for i in range(n):
        t = i / n
        f = new_frame()
        draw = ImageDraw.Draw(f)

        # Fade to black at the end (last 25%)
        fade_out = 1.0
        if t > 0.75:
            fade_out = 1.0 - ease_in_out((t - 0.75) / 0.25)

        # Title
        ta = ease_in_out(min(1.0, t / 0.25)) * fade_out
        tc = tuple(int(c * ta) for c in WHITE)
        centered_text(draw, 300, "MARCUS", title_font, fill=tc)

        # Accent line
        if t > 0.1:
            la = ease_out(min(1.0, (t - 0.1) / 0.2)) * fade_out
            line_w = int(360 * la)
            cx = W // 2
            draw.line(
                [(cx - line_w // 2, 405), (cx + line_w // 2, 405)],
                fill=tuple(int(c * la) for c in ACCENT),
                width=3,
            )

        # Tagline
        if t > 0.2:
            sa = ease_in_out(min(1.0, (t - 0.2) / 0.2)) * fade_out
            sc = tuple(int(c * sa) for c in GRAY)
            centered_text(draw, 430,
                          "An agentic multimodal vision-language model",
                          sub_font, fill=sc)
            centered_text(draw, 460,
                          "for cardiac diagnosis and management",
                          sub_font, fill=sc)

        # Affiliation
        if t > 0.35:
            aa = ease_in_out(min(1.0, (t - 0.35) / 0.2)) * fade_out
            ac = tuple(int(c * aa) for c in GRAY)
            centered_text(draw, 530, "Stanford University  |  UCSF", affil_font, fill=ac)

        # Link
        if t > 0.45:
            ga = ease_in_out(min(1.0, (t - 0.45) / 0.2)) * fade_out
            gc = tuple(int(c * ga) for c in ACCENT)
            centered_text(draw, 590, "github.com/masadi-99/MARCUS", link_font, fill=gc)

        frames.append(f)
    return frames


# ═══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def generate_video():
    print("Generating MARCUS demo video...")
    print(f"  Resolution: {W}x{H} @ {FPS} fps")

    # Generate all scenes
    scenes = [
        ("Title", scene_title),
        ("Problem", scene_problem),
        ("Scale", scene_scale),
        ("Architecture", scene_architecture),
        ("Attention", scene_attention),
        ("Mirage", scene_mirage),
        ("Orchestration", scene_orchestration),
        ("Performance", scene_performance),
        ("Closing", scene_closing),
    ]

    all_frames: list[Image.Image] = []

    for idx, (name, func) in enumerate(scenes):
        print(f"  Rendering scene {idx + 1}/{len(scenes)}: {name}...", end=" ", flush=True)
        scene_frames = func()
        print(f"{len(scene_frames)} frames")

        if all_frames and CROSSFADE_FRAMES > 0:
            # Cross-fade transition
            n_cf = min(CROSSFADE_FRAMES, len(all_frames), len(scene_frames))
            for ci in range(n_cf):
                t = (ci + 1) / (n_cf + 1)
                blended = fade_alpha(all_frames[-(n_cf - ci)], scene_frames[ci], t)
                all_frames[-(n_cf - ci)] = blended
            all_frames.extend(scene_frames[n_cf:])
        else:
            all_frames.extend(scene_frames)

    total_duration = len(all_frames) / FPS
    print(f"\n  Total frames: {len(all_frames)} ({total_duration:.1f} seconds)")

    # Write raw MP4 with OpenCV
    print(f"  Writing raw video to {OUTPUT_RAW}...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_RAW, fourcc, FPS, (W, H))

    for fi, frame in enumerate(all_frames):
        # PIL (RGB) → OpenCV (BGR)
        frame_np = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
        if (fi + 1) % 100 == 0:
            print(f"    {fi + 1}/{len(all_frames)} frames written", flush=True)

    writer.release()
    print(f"  Raw video written: {os.path.getsize(OUTPUT_RAW) / 1e6:.1f} MB")

    # Re-encode with H.264 for smaller size and browser compatibility
    print(f"  Re-encoding to H.264...")
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_bin = "ffmpeg"

    cmd = [
        ffmpeg_bin,
        "-i", OUTPUT_RAW,
        "-c:v", "libx264",
        "-crf", "20",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-y",
        OUTPUT_FINAL,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FFmpeg error: {result.stderr[-500:]}")
        print(f"  Falling back to raw MP4")
        os.rename(OUTPUT_RAW, OUTPUT_FINAL)
    else:
        final_size = os.path.getsize(OUTPUT_FINAL) / 1e6
        print(f"  Final video: {OUTPUT_FINAL} ({final_size:.1f} MB)")
        os.remove(OUTPUT_RAW)

    print(f"\nDone! Video saved to: {OUTPUT_FINAL}")
    print(f"Duration: {total_duration:.1f}s | Size: {os.path.getsize(OUTPUT_FINAL) / 1e6:.1f} MB")


if __name__ == "__main__":
    generate_video()
