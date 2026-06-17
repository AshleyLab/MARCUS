"""Tests for eval CLI OpenAI API key preflight check."""
from unittest.mock import patch
import pytest

from video_chat_ui.eval.cli import main


def test_main_exits_when_openai_api_key_not_set(capsys):
    """When OPENAI_API_KEY is not set, main should return 1 and print error to stderr."""
    with patch.dict("os.environ", {}, clear=False) as mock_env:
        # Remove OPENAI_API_KEY if it exists
        mock_env.pop("OPENAI_API_KEY", None)
        result = main([])

    assert result == 1
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY is not set" in captured.err
    assert "gpt-4o-mini" in captured.err
    assert "export OPENAI_API_KEY=sk-" in captured.err


def test_main_continues_when_openai_api_key_is_set(capsys):
    """When OPENAI_API_KEY is set, main should proceed past the check."""
    from pathlib import Path as PathLib

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}), \
         patch("video_chat_ui.eval.cli.run_batch") as mock_run_batch:

        # Create a temporary input file to avoid mocking Path
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            temp_file = f.name

        try:
            # Call main with proper arguments
            result = main([
                "--input", temp_file,
                "--task", "vqa"
            ])

            # Should have passed the key check and executed run_batch
            mock_run_batch.assert_called_once()
            # Should return 0 (success)
            assert result == 0
        finally:
            # Clean up temp file
            import os
            os.unlink(temp_file)
