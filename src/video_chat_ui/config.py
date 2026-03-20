"""Runtime configuration."""
import os
from pathlib import Path

API_BASE_URL = os.environ.get("LLAMA_API_URL", "http://localhost:8000")

_xdg = os.environ.get("XDG_CACHE_HOME")
_default_upload = (Path(_xdg) if _xdg else Path.home() / ".cache") / "video-chat-ui" / "uploads"
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", str(_default_upload))

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".png", ".jpg", ".jpeg", ".npy", ".xml", ".tgz"}
PORT = int(os.environ.get("PORT", "8765"))
