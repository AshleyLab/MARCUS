"""Tests for CLI validation in run_expert function."""
from unittest.mock import patch, MagicMock
import pytest
import sys

from video_chat_ui.cli import run_expert


def test_run_expert_exits_when_checkpoint_not_found(capsys):
    """When checkpoint path doesn't exist, run_expert should exit with error message."""
    call_count = [0]

    def mock_isdir(path):
        # First call (llama_dir check): return True
        # Second call (checkpoint check): return False
        call_count[0] += 1
        if call_count[0] == 1:
            return True  # llama_dir exists
        return False  # checkpoint doesn't exist

    with patch("video_chat_ui.cli.os.path.isdir", side_effect=mock_isdir), \
         patch("video_chat_ui.cli.os.environ.get", side_effect=lambda key, default=None: default), \
         pytest.raises(SystemExit) as exc_info:
        run_expert("ecg")

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Checkpoint not found:" in captured.err
    assert "download_checkpoints.py" in captured.err
    assert "--model ecg" in captured.err


def test_run_expert_exits_when_llama_dir_not_found(capsys):
    """When LLAMA_FACTORY_DIR doesn't exist, run_expert should exit with error message."""
    with patch("video_chat_ui.cli.os.path.isdir", return_value=False), \
         patch("video_chat_ui.cli.os.environ.get", side_effect=lambda key, default=None: default), \
         pytest.raises(SystemExit) as exc_info:
        run_expert("ecg")

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "LLAMA_FACTORY_DIR not found:" in captured.err


def test_run_expert_continues_when_both_paths_exist():
    """When both llama_dir and checkpoint exist, run_expert should continue to _start_api."""
    mock_proc = MagicMock()

    with patch("video_chat_ui.cli.os.path.isdir", return_value=True), \
         patch("video_chat_ui.cli.os.environ.get", side_effect=lambda key, default=None: default), \
         patch("video_chat_ui.cli.os.environ.__setitem__") as mock_setitem, \
         patch("video_chat_ui.cli._start_api", return_value=mock_proc) as mock_start_api, \
         patch("video_chat_ui.cli._wait_api"), \
         patch("video_chat_ui.cli.uvicorn.run", side_effect=KeyboardInterrupt()), \
         patch("video_chat_ui.cli._stop_api"):

        # Should not raise any exception, but KeyboardInterrupt is caught
        run_expert("ecg")

        # Verify _start_api was called (meaning checkpoint validation passed)
        mock_start_api.assert_called_once()
