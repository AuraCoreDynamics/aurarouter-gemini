"""Tests for GeminiProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aurarouter_gemini.models import GEMINI_MODELS


# -------------------------------------------------------------------------
# Construction
# -------------------------------------------------------------------------


class TestProviderInit:
    """Tests for GeminiProvider construction and API key resolution."""

    def test_create_with_explicit_key(self, mock_genai_client):
        from aurarouter_gemini.provider import GeminiProvider

        p = GeminiProvider(api_key="explicit-key")
        assert p._api_key == "explicit-key"

    def test_create_with_gemini_api_key_env(self, mock_genai_client, monkeypatch):
        from aurarouter_gemini.provider import GeminiProvider

        monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
        p = GeminiProvider()
        assert p._api_key == "env-gemini-key"

    def test_create_with_google_api_key_env(self, mock_genai_client, monkeypatch):
        from aurarouter_gemini.provider import GeminiProvider

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "env-google-key")
        p = GeminiProvider()
        assert p._api_key == "env-google-key"

    def test_gemini_key_takes_precedence_over_google_key(self, mock_genai_client, monkeypatch):
        from aurarouter_gemini.provider import GeminiProvider

        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        p = GeminiProvider()
        assert p._api_key == "gemini-key"

    def test_missing_api_key_raises(self, mock_genai_client, monkeypatch):
        from aurarouter_gemini.provider import GeminiProvider

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key is required"):
            GeminiProvider()

    def test_default_model(self, provider):
        assert provider._default_model == "gemini-2.5-flash"

    def test_custom_default_model(self, provider_custom_model):
        assert provider_custom_model._default_model == "gemini-2.5-pro"


# -------------------------------------------------------------------------
# generate()
# -------------------------------------------------------------------------


class TestGenerate:
    """Tests for the generate() method."""

    def test_generate_returns_text(self, provider, mock_genai_client):
        resp = SimpleNamespace()
        resp.text = "Generated text"
        resp.usage_metadata = SimpleNamespace(
            prompt_token_count=5,
            candidates_token_count=15,
        )
        mock_genai_client.models.generate_content.return_value = resp

        result = provider.generate("Hello")
        assert result["text"] == "Generated text"

    def test_generate_returns_model_id(self, provider, mock_genai_client):
        result = provider.generate("Hello")
        assert result["model_id"] == "gemini-2.5-flash"

    def test_generate_with_explicit_model(self, provider, mock_genai_client):
        result = provider.generate("Hello", model="gemini-2.5-pro")
        assert result["model_id"] == "gemini-2.5-pro"
        call_kwargs = mock_genai_client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-2.5-pro"

    def test_generate_extracts_input_tokens(self, provider, mock_genai_client):
        resp = SimpleNamespace()
        resp.text = "Response"
        resp.usage_metadata = SimpleNamespace(
            prompt_token_count=42,
            candidates_token_count=99,
        )
        mock_genai_client.models.generate_content.return_value = resp

        result = provider.generate("Test")
        assert result["input_tokens"] == 42

    def test_generate_extracts_output_tokens(self, provider, mock_genai_client):
        resp = SimpleNamespace()
        resp.text = "Response"
        resp.usage_metadata = SimpleNamespace(
            prompt_token_count=42,
            candidates_token_count=99,
        )
        mock_genai_client.models.generate_content.return_value = resp

        result = provider.generate("Test")
        assert result["output_tokens"] == 99

    def test_generate_context_limit_from_catalog(self, provider, mock_genai_client):
        result = provider.generate("Hello", model="gemini-2.5-flash")
        assert result["context_limit"] == 1048576

    def test_generate_unknown_model_context_limit(self, provider, mock_genai_client):
        result = provider.generate("Hello", model="gemini-unknown")
        assert result["context_limit"] == 0

    def test_generate_passes_prompt_as_contents(self, provider, mock_genai_client):
        provider.generate("My prompt")
        call_kwargs = mock_genai_client.models.generate_content.call_args
        assert call_kwargs.kwargs["contents"] == "My prompt"

    def test_generate_json_mode(self, provider, mock_genai_client):
        provider.generate("Give JSON", json_mode=True)
        call_kwargs = mock_genai_client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config["response_mime_type"] == "application/json"

    def test_generate_no_json_mode(self, provider, mock_genai_client):
        provider.generate("Hello", json_mode=False)
        call_kwargs = mock_genai_client.models.generate_content.call_args
        assert call_kwargs.kwargs["config"] is None

    def test_generate_handles_none_text(self, provider, mock_genai_client):
        resp = SimpleNamespace()
        resp.text = None
        resp.usage_metadata = SimpleNamespace(
            prompt_token_count=1,
            candidates_token_count=0,
        )
        mock_genai_client.models.generate_content.return_value = resp

        result = provider.generate("Hello")
        assert result["text"] == ""

    def test_generate_handles_missing_usage_metadata(self, provider, mock_genai_client):
        resp = SimpleNamespace()
        resp.text = "Response"
        resp.usage_metadata = None
        mock_genai_client.models.generate_content.return_value = resp

        result = provider.generate("Hello")
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0


# -------------------------------------------------------------------------
# generate_with_history()
# -------------------------------------------------------------------------


class TestGenerateWithHistory:
    """Tests for the generate_with_history() method."""

    def test_history_returns_text(self, provider, mock_genai_client):
        resp = SimpleNamespace()
        resp.text = "History response"
        resp.usage_metadata = SimpleNamespace(
            prompt_token_count=20,
            candidates_token_count=30,
        )
        mock_genai_client.models.generate_content.return_value = resp

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = provider.generate_with_history(messages)
        assert result["text"] == "History response"

    def test_history_maps_assistant_to_model(self, provider, mock_genai_client):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        provider.generate_with_history(messages)
        call_kwargs = mock_genai_client.models.generate_content.call_args
        contents = call_kwargs.kwargs["contents"]
        # The assistant message should be mapped to "model"
        assert contents[1]["role"] == "model"

    def test_history_with_system_prompt(self, provider, mock_genai_client):
        messages = [{"role": "user", "content": "Hi"}]
        provider.generate_with_history(messages, system_prompt="Be helpful")
        call_kwargs = mock_genai_client.models.generate_content.call_args
        contents = call_kwargs.kwargs["contents"]
        # System prompt becomes a user+model exchange prepended
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == "Be helpful"
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"][0]["text"] == "Understood."

    def test_history_token_extraction(self, provider, mock_genai_client):
        resp = SimpleNamespace()
        resp.text = "Response"
        resp.usage_metadata = SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
        )
        mock_genai_client.models.generate_content.return_value = resp

        result = provider.generate_with_history([{"role": "user", "content": "Test"}])
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    def test_history_with_explicit_model(self, provider, mock_genai_client):
        messages = [{"role": "user", "content": "Hi"}]
        result = provider.generate_with_history(messages, model="gemini-2.0-flash")
        assert result["model_id"] == "gemini-2.0-flash"

    def test_history_json_mode(self, provider, mock_genai_client):
        messages = [{"role": "user", "content": "Give JSON"}]
        provider.generate_with_history(messages, json_mode=True)
        call_kwargs = mock_genai_client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config["response_mime_type"] == "application/json"


# -------------------------------------------------------------------------
# list_models()
# -------------------------------------------------------------------------


class TestListModels:
    """Tests for the list_models() method."""

    def test_list_models_returns_all(self, provider):
        models = provider.list_models()
        assert len(models) == len(GEMINI_MODELS)

    def test_list_models_contains_flash(self, provider):
        models = provider.list_models()
        ids = [m["model_id"] for m in models]
        assert "gemini-2.5-flash" in ids

    def test_list_models_contains_pro(self, provider):
        models = provider.list_models()
        ids = [m["model_id"] for m in models]
        assert "gemini-2.5-pro" in ids

    def test_list_models_returns_copies(self, provider):
        """Ensure returned models are copies, not references to the catalog."""
        models = provider.list_models()
        models[0]["model_id"] = "mutated"
        fresh = provider.list_models()
        assert fresh[0]["model_id"] != "mutated"

    def test_model_has_required_fields(self, provider):
        models = provider.list_models()
        for m in models:
            assert "model_id" in m
            assert "display_name" in m
            assert "context_window" in m
            assert "cost_per_1m_input" in m
            assert "cost_per_1m_output" in m
            assert "capabilities" in m
