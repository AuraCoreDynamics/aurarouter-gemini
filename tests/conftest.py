"""Shared fixtures for aurarouter-gemini tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_usage_metadata(prompt_tokens: int = 10, candidates_tokens: int = 25):
    """Create a mock usage_metadata object."""
    meta = SimpleNamespace()
    meta.prompt_token_count = prompt_tokens
    meta.candidates_token_count = candidates_tokens
    return meta


def _make_generate_response(
    text: str = "Hello from Gemini!",
    prompt_tokens: int = 10,
    candidates_tokens: int = 25,
):
    """Create a mock generate_content response."""
    resp = SimpleNamespace()
    resp.text = text
    resp.usage_metadata = _make_usage_metadata(prompt_tokens, candidates_tokens)
    return resp


@pytest.fixture
def mock_genai_client():
    """Patch google.genai.Client and return the mock client instance."""
    with patch("aurarouter_gemini.provider.genai.Client") as mock_cls:
        client_instance = MagicMock()
        mock_cls.return_value = client_instance

        # Set up models.generate_content to return a proper response
        client_instance.models.generate_content.return_value = _make_generate_response()

        yield client_instance


@pytest.fixture
def provider(mock_genai_client):
    """Create a GeminiProvider with a mocked client."""
    from aurarouter_gemini.provider import GeminiProvider

    p = GeminiProvider(api_key="test-key-123")
    return p


@pytest.fixture
def provider_custom_model(mock_genai_client):
    """Create a GeminiProvider with a custom default model."""
    from aurarouter_gemini.provider import GeminiProvider

    p = GeminiProvider(api_key="test-key-123", default_model="gemini-2.5-pro")
    return p
