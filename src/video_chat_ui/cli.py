"""
CLI: start model API + web UI per expert, or UI-only when API already runs.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import uvicorn

EXPERTS = {
    "cmr": {
        "api_port": 8000,
        "ui_port": 8765,
        "model_relpath": "saves/Qwen2.5-VL-3B-Instruct/full/cmr_grpo",
        "EXPERT_DOC_TITLE": "CMR expert model",
        "EXPERT_PAGE_HEADING": "CMR expert model",
        "EXPERT_TAGLINE": "Upload a video and chat with the CMR expert model.",
    },
    "echo": {
        "api_port": 8010,
        "ui_port": 8770,
        "model_relpath": "saves/Qwen2.5-VL-3B-Instruct/full/echo_grpo",
        "EXPERT_DOC_TITLE": "Echo expert model",
        "EXPERT_PAGE_HEADING": "Echo expert model",
        "EXPERT_TAGLINE": "Upload a video and chat with the Echo expert model.",
    },
    "ecg": {
        "api_port": 8020,
        "ui_port": 8775,
        "model_relpath": "saves/Qwen2.5-VL-3B-Instruct/full/ecg_sft",
        "EXPERT_DOC_TITLE": "ECG expert model",
        "EXPERT_PAGE_HEADING": "ECG expert model",
        "EXPERT_TAGLINE": "Upload a video and chat with the ECG expert model.",
    },
}


def _wait_api(port: int, timeout_sec: int = 240) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    print("Model API did not become ready in time.", file=sys.stderr)
    sys.exit(1)


def _start_api(llama_dir: str, sif: str, api_port: int, model_relpath: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["API_HOST"] = "0.0.0.0"
    env["API_PORT"] = str(api_port)
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    yaml_arg = "examples/inference/qwen2_5vl.yaml"
    model_arg = f"model_name_or_path={model_relpath}"
    if os.path.isfile(sif):
        cmd = [
            "singularity",
            "exec",
            "--nv",
            sif,
            "llamafactory-cli",
            "api",
            yaml_arg,
            model_arg,
        ]
    else:
        cmd = ["llamafactory-cli", "api", yaml_arg, model_arg]
    return subprocess.Popen(
        cmd,
        cwd=llama_dir,
        env=env,
        start_new_session=True,
    )


def _stop_api(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def run_expert(name: str) -> None:
    cfg = EXPERTS[name]
    llama_dir = os.environ.get("LLAMA_FACTORY_DIR", os.path.expanduser("~/LLaMA-Factory"))
    sif = os.path.expanduser(os.environ.get("LLAMA_FACTORY_SIF", "~/llamafactory_latest.sif"))
    model_relpath = os.environ.get("MODEL_RELPATH", cfg["model_relpath"])
    api_port = int(os.environ.get("API_PORT", cfg["api_port"]))
    ui_port = int(os.environ.get("PORT", cfg["ui_port"]))

    for key in ("EXPERT_DOC_TITLE", "EXPERT_PAGE_HEADING", "EXPERT_TAGLINE"):
        os.environ[key] = cfg[key]
    os.environ["LLAMA_API_URL"] = f"http://127.0.0.1:{api_port}"
    os.environ["PORT"] = str(ui_port)

    if not os.path.isdir(llama_dir):
        print(f"LLAMA_FACTORY_DIR not found: {llama_dir}", file=sys.stderr)
        sys.exit(1)

    proc = _start_api(llama_dir, sif, api_port, model_relpath)
    print(f"Loading {name.upper()} expert (API :{api_port})...")
    try:
        _wait_api(api_port)
        print(f"Open http://localhost:{ui_port}  (Ctrl+C stops UI and API)")
        uvicorn.run("video_chat_ui.app:app", host="0.0.0.0", port=ui_port)
    except KeyboardInterrupt:
        pass
    finally:
        _stop_api(proc)
        print("Stopped.")


def main_ui_only() -> None:
    os.environ.setdefault("EXPERT_DOC_TITLE", "CMR expert model")
    os.environ.setdefault("EXPERT_PAGE_HEADING", "CMR expert model")
    os.environ.setdefault(
        "EXPERT_TAGLINE",
        "Upload a video and chat with the CMR expert model.",
    )
    port = int(os.environ.get("PORT", "8765"))
    print(f"UI http://localhost:{port}  API {os.environ.get('LLAMA_API_URL', 'http://localhost:8000')}")
    uvicorn.run("video_chat_ui.app:app", host="0.0.0.0", port=port)


def main_cmr() -> None:
    run_expert("cmr")


def main_echo() -> None:
    run_expert("echo")


def main_ecg() -> None:
    run_expert("ecg")
