"""Tests for the /api-status endpoint."""
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from video_chat_ui.app import app

client = TestClient(app)


def test_api_status_returns_200_when_model_unreachable():
    """When httpx raises a ConnectError the endpoint must return 200 with connected=False."""
    with patch("video_chat_ui.app.httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_instance.get.side_effect = httpx.ConnectError("refused")

        response = client.get("/api-status")

    assert response.status_code == 200
    assert response.json() == {"connected": False}


def test_api_status_returns_connected_true_on_200():
    """When the model API returns 200, connected should be True."""
    mock_response = AsyncMock()
    mock_response.status_code = 200

    with patch("video_chat_ui.app.httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_instance.get.return_value = mock_response

        response = client.get("/api-status")

    assert response.status_code == 200
    assert response.json() == {"connected": True}


def test_api_status_returns_connected_false_on_non_200():
    """When the model API returns a non-200 status, connected should be False."""
    mock_response = AsyncMock()
    mock_response.status_code = 503

    with patch("video_chat_ui.app.httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_instance.get.return_value = mock_response

        response = client.get("/api-status")

    assert response.status_code == 200
    assert response.json() == {"connected": False}


def test_api_status_returns_connected_false_on_timeout():
    """When httpx raises a TimeoutException the endpoint must still return 200."""
    with patch("video_chat_ui.app.httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_instance.get.side_effect = httpx.TimeoutException("timeout")

        response = client.get("/api-status")

    assert response.status_code == 200
    assert response.json() == {"connected": False}
