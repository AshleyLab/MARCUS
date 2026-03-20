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
/* ── Force light mode (override system dark preference) ──────────────── */
:root { color-scheme: light !important; }
body.dark { background: var(--background-fill-primary) !important; }

/* ── Global ──────────────────────────────────────────────────────────── */
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Header ──────────────────────────────────────────────────────────── */
.marcus-header {
    text-align: center;
    padding: 32px 20px 20px;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 8px;
}
.marcus-header h1 {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #1e293b;
    letter-spacing: 0.12em;
    margin: 0 0 4px;
}
.marcus-header .subtitle {
    font-size: 0.95rem;
    color: #64748b;
    font-weight: 400;
    margin: 0 0 10px;
    letter-spacing: 0.01em;
}
.marcus-header .authors {
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.6;
}
.marcus-header .authors a {
    color: #3b82f6;
    text-decoration: none;
}

/* ── Tab styling ─────────────────────────────────────────────────── */
.tab-nav button {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em;
}

/* ── Section headings ────────────────────────────────────────────── */
h2 {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #1e293b;
    margin-top: 1.0em;
    padding-bottom: 6px;
    border-bottom: 1px solid #e2e8f0;
}
h3 {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 1.0rem;
    font-weight: 600;
    color: #334155;
}

/* ── Status badges ───────────────────────────────────────────────── */
.status-row {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 12px 0 8px;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'Inter', system-ui, sans-serif;
}
.status-online {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}
.status-offline {
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.status-online .status-dot { background: #059669; }
.status-offline .status-dot { background: #dc2626; }

/* ── Result boxes ────────────────────────────────────────────────── */
.result-box textarea {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
}

/* ── Score cards ──────────────────────────────────────────────────── */
.score-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}
.score-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
}
.score-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
}

/* ── Verdict banners ─────────────────────────────────────────────── */
.verdict-pass {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 8px;
    padding: 12px 18px;
    color: #065f46;
    font-weight: 600;
    font-size: 0.92rem;
    text-align: center;
}
.verdict-fail {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 12px 18px;
    color: #92400e;
    font-weight: 600;
    font-size: 0.92rem;
    text-align: center;
}

/* ── Timing pill ─────────────────────────────────────────────────── */
.timing-pill {
    display: inline-block;
    background: #f1f5f9;
    color: #475569;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 3px 12px;
    border-radius: 12px;
    margin-top: 6px;
}

/* ── Figure captions ─────────────────────────────────────────────── */
.figure-caption {
    font-size: 0.78rem;
    color: #94a3b8;
    text-align: center;
    margin-top: 4px;
    font-style: italic;
}

/* ── Table refinements ───────────────────────────────────────────── */
table {
    font-size: 0.85rem !important;
}
table th {
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* ── Hide Gradio footer ──────────────────────────────────────────── */
footer { display: none !important; }

/* ── Card-style group containers ─────────────────────────────────── */
.card-group {
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
}

/* ── Pipeline step indicator ─────────────────────────────────────── */
.pipeline-steps {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 6px;
    margin: 14px 0 6px;
    flex-wrap: wrap;
}
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0;
}
.step-box {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #475569;
    white-space: nowrap;
}
.step-arrow {
    color: #94a3b8;
    font-size: 0.9rem;
    padding: 0 6px;
}

/* ── Disclaimer ──────────────────────────────────────────────────── */
.disclaimer {
    text-align: center;
    font-size: 0.76rem;
    color: #94a3b8;
    padding: 12px 20px;
    border-top: 1px solid #e2e8f0;
    margin-top: 16px;
}

/* ── Mirage example cards ────────────────────────────────────────── */
.mirage-example {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
}
.mirage-example-pass { border-left: 4px solid #059669; }
.mirage-example-fail { border-left: 4px solid #d97706; }
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
    """Send a chat query to the expert's FastAPI UI and collect the streamed response."""
    base = _ui_url(modality)
    payload = {
        "video_id": media_id,
        "media_kind": media_kind,
        "messages": [{"role": "user", "content": question}],
    }
    async with httpx.AsyncClient(timeout=180) as c:
        resp = await c.post(f"{base}/chat", json=payload)
        resp.raise_for_status()
        raw = resp.text
    # Parse SSE data lines
    answer_parts = []
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                continue
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    answer_parts.append(delta["content"])
            except (json.JSONDecodeError, IndexError, KeyError):
                answer_parts.append(data_str)
    return "".join(answer_parts) if answer_parts else raw


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    import re
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Single-modality handler
# ---------------------------------------------------------------------------


async def run_single_analysis(
    file_obj: Any,
    modality: str,
    question: str,
    enable_mirage: bool,
    template: str | None = None,
) -> tuple[str, str, str]:
    """Run single-modality analysis. Returns (answer, confidence_info, timing)."""
    if file_obj is None:
        return "Please upload a file or select an example.", "", ""
    # Fall back to template dropdown if textbox is empty
    if not question.strip() and template:
        question = template
    if not question.strip():
        return "Please enter a clinical question.", "", ""

    modality = modality.lower()
    t0 = time.time()

    # Get file path
    file_path = file_obj if isinstance(file_obj, str) else file_obj.name if hasattr(file_obj, "name") else str(file_obj)

    try:
        alive = await _check_expert(modality)
        if not alive:
            return f"The {EXPERTS[modality]['label']} expert is not available. Start the model server first.", "", ""

        media_id, media_kind = await _upload_file(file_path, modality)
        answer = await _chat_query(modality, media_id, media_kind, question)
        answer = _strip_think_tags(answer)

        elapsed = time.time() - t0
        timing = f'<span class="timing-pill">Completed in {elapsed:.1f}s</span>'

        confidence_info = ""
        if enable_mirage:
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

        return answer, confidence_info, timing

    except Exception as e:
        return f"Error: {e}", "", ""


# ---------------------------------------------------------------------------
# Multimodal handler
# ---------------------------------------------------------------------------


async def run_multimodal_analysis(
    ecg_file: Any,
    echo_file: Any,
    cmr_file: Any,
    question: str,
    template: str | None = None,
) -> tuple[str, str, str, str, str, str]:
    """Run multimodal analysis. Returns (synth_answer, ecg_resp, echo_resp, cmr_resp, confidence_md, timing)."""
    # Fall back to template dropdown if textbox is empty
    if not question.strip() and template:
        question = template
    if not question.strip():
        return "Please enter a clinical question.", "", "", "", "", ""

    files = {"ecg": ecg_file, "echo": echo_file, "cmr": cmr_file}
    provided = {m: f for m, f in files.items() if f is not None}
    if len(provided) < 2:
        return "Please provide at least two modalities for multimodal analysis.", "", "", "", "", ""

    t0 = time.time()

    # Check health of needed experts
    for mod in provided:
        alive = await _check_expert(mod)
        if not alive:
            return f"The {EXPERTS[mod]['label']} expert is not available.", "", "", "", "", ""

    try:
        # Upload all files
        media_ids: dict[str, str] = {}
        for mod, f in provided.items():
            fp = f if isinstance(f, str) else f.name if hasattr(f, "name") else str(f)
            mid, _ = await _upload_file(fp, mod)
            media_ids[mod] = mid

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

        elapsed = time.time() - t0
        timing = f'<span class="timing-pill">Multimodal synthesis in {elapsed:.1f}s</span>'

        synth_answer = _strip_think_tags(result.answer)
        return synth_answer, ecg_resp, echo_resp, cmr_resp, confidence_md, timing

    except Exception as e:
        return f"Error: {e}", "", "", "", "", ""


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
            verdict = '<div class="verdict-fail">⚠ MIRAGE DETECTED — Responses are similar with and without visual input, suggesting the model may not be genuinely referencing the provided data.</div>'
        else:
            verdict = '<div class="verdict-pass">✓ NO MIRAGE — Responses are grounded in the provided visual data. The model\'s analysis changes substantively when the image is removed.</div>'

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
                O'Sullivan JW*, Asadi M*, Elbe L, Chaudhari A, Nedaee T,
                Haddad F, Salerno M, Fei-Fei L, Adeli E, Arnaout R, Ashley EA<br>
                Stanford University · UCSF
            </p>
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
                    "and **cardiac MRI** — both independently and as multimodal input. "
                    "Trained on **13.5 million images** from **270,000 clinical studies** "
                    "with physician-verified ground truth."
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
                        mirage_cb = gr.Checkbox(
                            label="Enable mirage probing",
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
                    [single_file, modality_radio, question_input, mirage_cb, template_dd],
                    [answer_box, confidence_md, timing_md],
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
                    <div class="pipeline-step" style="margin-left:6px;">
                        <span class="step-arrow">→</span>
                        <span class="step-box" style="background:#dbeafe; border-color:#93c5fd;">Orchestrator</span>
                        <span class="step-arrow">→</span>
                        <span class="step-box" style="background:#2563eb; color:#fff; border-color:#1d4ed8;">Report</span>
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
                    [multi_ecg, multi_echo, multi_cmr, multi_question, multi_template],
                    [multi_answer, ecg_resp_box, echo_resp_box, cmr_resp_box,
                     multi_confidence, multi_timing],
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
            "MARCUS is intended for research use and clinical decision support. "
            "It is not a substitute for professional medical judgment. "
            "For research purposes only."
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
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
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
        share=False,
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
