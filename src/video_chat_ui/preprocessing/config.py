"""Preprocessing defaults (overridable via env)."""
import os
import tempfile
from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    workdir: str | None = None
    jpg_quality: int = 95
    mp4_fps: int = 30

    def resolved_workdir(self) -> str:
        return self.workdir or os.environ.get(
            "VIDEO_CHAT_WORKDIR", os.path.join(tempfile.gettempdir(), "video_chat_ui_preprocess")
        )
