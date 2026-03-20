"""FastAPI app for video chat UI."""
import asyncio
import json
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from video_chat_ui import config

app = FastAPI(title="Video Chat UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path(config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

PREPROCESS_TIMEOUT_SEC = float(os.environ.get("PREPROCESS_TIMEOUT_SEC", "900"))
_preprocess_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="preprocess")


def get_path_for_id(file_id: str) -> Path | None:
    path = Path(config.UPLOAD_DIR) / file_id
    return path if path.is_file() else None


def _container_safe_video_path(p: Path) -> str:
    """Return a video path accessible inside the Singularity container.

    The default upload dir may live under a symlink (e.g. ~/.cache ->
    /projects/…) that is not bind-mounted inside the container.  /tmp/
    is always mounted, so we hardlink or copy there.
    """
    tmp_dir = Path("/tmp/video-chat-ui-media")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / p.name
    if not dest.exists():
        try:
            os.link(str(p.resolve()), str(dest))
        except OSError:
            shutil.copy2(str(p), str(dest))
    return str(dest)


@app.get("/health")
def health():
    return {"status": "ok"}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    video_id: str | None = None
    messages: list[ChatMessage]
    media_kind: Literal["video", "image"] = "video"


@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    ext = Path(video.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail=f"Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}")
    content = await video.read()
    if len(content) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(400, detail=f"Max {config.MAX_UPLOAD_MB}MB")
    file_id = f"{uuid.uuid4().hex}{ext}"
    path = Path(config.UPLOAD_DIR) / file_id
    path.write_bytes(content)
    return {"id": file_id, "filename": video.filename or file_id, "path": str(path.resolve())}


def build_api_messages(
    media_path: str | None,
    messages: list[ChatMessage],
    media_kind: Literal["video", "image"] = "video",
) -> list[dict]:
    api_messages = []
    first_user = True
    for m in messages:
        if m.role == "user":
            if first_user and media_path:
                if media_kind == "image":
                    content = [
                        {"type": "image_url", "image_url": {"url": media_path}},
                        {"type": "text", "text": m.content},
                    ]
                else:
                    content = [
                        {"type": "video_url", "video_url": {"url": media_path}},
                        {"type": "text", "text": m.content},
                    ]
                first_user = False
            else:
                content = m.content
            api_messages.append({"role": "user", "content": content})
        else:
            api_messages.append({"role": "assistant", "content": m.content})
    return api_messages


def _preprocess_sync(expert: str, content: bytes, filename: str) -> tuple[Path, Path, str]:
    expert = expert.lower().strip()
    fn = (filename or "").lower()
    if expert == "ecg":
        if fn.endswith(".xml"):
            from video_chat_ui.preprocessing.pipeline import preprocess_ecg_xml

            out, scratch = preprocess_ecg_xml(None, content)
            return out, scratch, "image"
        elif fn.endswith(".npy"):
            from video_chat_ui.preprocessing.pipeline import preprocess_ecg_npy

            out, scratch = preprocess_ecg_npy(None, content)
            return out, scratch, "image"
        else:
            raise ValueError("ECG mode requires a .npy file (12 x N array) or a .xml institutional ECG file.")
    if not (fn.endswith(".tgz") or fn.endswith(".tar.gz")):
        raise ValueError("CMR/Echo mode requires a .tgz (or .tar.gz) archive.")
    import tempfile

    tgz_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tf:
            tf.write(content)
            tgz_path = Path(tf.name)
        from video_chat_ui.preprocessing.pipeline import preprocess_tgz_for_expert

        assert expert in ("cmr", "echo")
        out, scratch = preprocess_tgz_for_expert(tgz_path, expert)  # type: ignore[arg-type]
        return out, scratch, "video"
    finally:
        if tgz_path is not None:
            tgz_path.unlink(missing_ok=True)


@app.get("/media/{file_id}")
def serve_media(file_id: str):
    """Serve uploaded/processed file for in-browser preview (opaque id)."""
    p = get_path_for_id(file_id)
    if not p:
        raise HTTPException(404, detail="Not found")
    return FileResponse(p)


@app.post("/preprocess")
async def preprocess_endpoint(file: UploadFile = File(...), expert: str = Form(...)):
    exp = expert.lower().strip()
    if exp not in ("cmr", "echo", "ecg"):
        raise HTTPException(400, detail="expert must be cmr, echo, or ecg")
    content = await file.read()
    if len(content) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(400, detail=f"Max {config.MAX_UPLOAD_MB}MB")
    loop = asyncio.get_event_loop()
    try:
        out_path, scratch, kind = await asyncio.wait_for(
            loop.run_in_executor(
                _preprocess_executor,
                lambda: _preprocess_sync(exp, content, file.filename or ""),
            ),
            timeout=PREPROCESS_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, detail="Preprocessing timed out. Try a smaller archive.")
    except Exception as e:
        msg = str(e) or type(e).__name__
        raise HTTPException(400, detail=msg)
    suffix = out_path.suffix.lower() or ".bin"
    file_id = f"{uuid.uuid4().hex}{suffix}"
    dest = Path(config.UPLOAD_DIR) / file_id
    try:
        shutil.copy2(out_path, dest)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {"id": file_id, "kind": kind, "expert": exp}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(400, detail="messages required")
    media_ref = None
    if req.video_id:
        p = get_path_for_id(req.video_id)
        if not p or not p.is_file():
            raise HTTPException(400, detail="Invalid or expired media id")
        if req.media_kind == "video":
            media_ref = _container_safe_video_path(p)
        else:
            ui_port = config.PORT
            media_ref = f"http://127.0.0.1:{ui_port}/media/{req.video_id}"
    api_messages = build_api_messages(media_ref, req.messages, req.media_kind)
    body = {
        "model": "gpt-3.5-turbo",
        "messages": api_messages,
        "stream": True,
        "max_tokens": 1024,
    }
    url = f"{config.API_BASE_URL.rstrip('/')}/v1/chat/completions"
    try:
        async def _stream_from_api():
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, json=body) as resp:
                    if resp.status_code != 200:
                        text = await resp.aread()
                        yield f"data: {json.dumps({'error': text.decode('utf-8', errors='replace')})}\n\n"
                        return
                    try:
                        async for chunk in resp.aiter_text():
                            if chunk.strip():
                                yield chunk
                    except (httpx.RemoteProtocolError, httpx.ReadError):
                        pass

        return StreamingResponse(
            _stream_from_api(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except httpx.ConnectError:
        raise HTTPException(
            503,
            detail="Cannot connect to model API. Start it first (e.g. llamafactory-cli api ...).",
        )
    except httpx.TimeoutException:
        raise HTTPException(504, detail="API request timed out.")


@app.post("/chat_attention")
async def chat_attention(req: ChatRequest):
    """Forward attention extraction request to the LLaMA-Factory API."""
    if not req.messages:
        raise HTTPException(400, detail="messages required")
    media_ref = None
    if req.video_id:
        p = get_path_for_id(req.video_id)
        if not p or not p.is_file():
            raise HTTPException(400, detail="Invalid or expired media id")
        if req.media_kind == "video":
            media_ref = _container_safe_video_path(p)
        else:
            ui_port = config.PORT
            media_ref = f"http://127.0.0.1:{ui_port}/media/{req.video_id}"
    api_messages = build_api_messages(media_ref, req.messages, req.media_kind)
    body = {
        "model": "gpt-3.5-turbo",
        "messages": api_messages,
        "stream": False,
        "max_tokens": 1024,
    }
    url = f"{config.API_BASE_URL.rstrip('/')}/v1/chat/attention"
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=body)
            if resp.status_code != 200:
                raise HTTPException(resp.status_code, detail=resp.text)
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(503, detail="Cannot connect to model API.")
    except httpx.TimeoutException:
        raise HTTPException(504, detail="API request timed out (attention extraction is slower).")
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(500, detail=str(e))


static_dir = Path(__file__).resolve().parent / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    _tpl = static_dir / "index.template.html"

    @app.get("/")
    def index():
        doc = os.environ.get("EXPERT_DOC_TITLE", "CMR expert model")
        heading = os.environ.get("EXPERT_PAGE_HEADING", "CMR expert model")
        tagline = os.environ.get(
            "EXPERT_TAGLINE",
            "Upload a video and chat with the CMR expert model.",
        )
        html = _tpl.read_text(encoding="utf-8")
        html = (
            html.replace("{{DOCUMENT_TITLE}}", doc)
            .replace("{{PAGE_HEADING}}", heading)
            .replace("{{TAGLINE}}", tagline)
        )
        return HTMLResponse(html)
