"""Tests for the MCP server factory."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _make_response(text="Server response", prompt_tokens=5, candidates_tokens=15):
    resp = SimpleNamespace()
    resp.text = text
    resp.usage_metadata = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=candidates_tokens,
    )
    return resp


@pytest.fixture(autouse=True)
def _reset_provider_singleton():
    """Reset the module-level provider singleton between tests."""
    import aurarouter_gemini.server as srv
    srv._provider = None
    yield
    srv._provider = None


@pytest.fixture
def mock_client():
    """Patch google.genai.Client for server tests."""
    with patch("aurarouter_gemini.provider.genai.Client") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        client.models.generate_content.return_value = _make_response()
        yield client


@pytest.fixture
def server(mock_client):
    """Create a FastMCP server instance."""
    import os
    os.environ.setdefault("GEMINI_API_KEY", "test-server-key")
    from aurarouter_gemini.server import create_server
    return create_server()


# -------------------------------------------------------------------------
# Tool registration
# -------------------------------------------------------------------------


class TestToolRegistration:
    """Verify that all four required tools are registered."""

    def test_server_has_generate_tool(self, server):
        tools = server._tool_manager.list_tools()
        names = [t.name for t in tools]
        assert "provider.generate" in names, f"Tools: {names}"

    def test_server_has_list_models_tool(self, server):
        tools = server._tool_manager.list_tools()
        names = [t.name for t in tools]
        assert "provider.list_models" in names, f"Tools: {names}"

    def test_server_has_generate_with_history_tool(self, server):
        tools = server._tool_manager.list_tools()
        names = [t.name for t in tools]
        assert "provider.generate_with_history" in names, f"Tools: {names}"

    def test_server_has_capabilities_tool(self, server):
        tools = server._tool_manager.list_tools()
        names = [t.name for t in tools]
        assert "provider.capabilities" in names, f"Tools: {names}"

    def test_exactly_four_tools(self, server):
        tools = server._tool_manager.list_tools()
        assert len(tools) == 4

    def test_no_unrecognized_provider_tools(self, server):
        """All provider.* tools should be in the protocol's ALL_TOOLS set."""
        known = {
            "provider.generate",
            "provider.list_models",
            "provider.generate_with_history",
            "provider.health_check",
            "provider.capabilities",
        }
        tools = server._tool_manager.list_tools()
        for t in tools:
            if t.name.startswith("provider."):
                assert t.name in known, f"Unrecognised tool: {t.name}"


# -------------------------------------------------------------------------
# Tool execution
# -------------------------------------------------------------------------


class TestToolExecution:
    """Test that tools produce valid output."""

    @pytest.mark.asyncio
    async def test_generate_returns_json(self, server, mock_client):
        mock_client.models.generate_content.return_value = _make_response("Test output")
        result = await server.call_tool("provider.generate", {"prompt": "Hello"})
        # FastMCP call_tool returns (content_list, metadata_dict)
        content_list = result[0]
        text = content_list[0].text
        data = json.loads(text)
        assert "text" in data
        assert data["text"] == "Test output"
        assert "model_id" in data
        assert "input_tokens" in data
        assert "output_tokens" in data

    @pytest.mark.asyncio
    async def test_list_models_returns_json(self, server, mock_client):
        result = await server.call_tool("provider.list_models", {})
        content_list = result[0]
        text = content_list[0].text
        data = json.loads(text)
        assert isinstance(data, list)
        assert len(data) == 3
        ids = [m["model_id"] for m in data]
        assert "gemini-2.5-pro" in ids

    @pytest.mark.asyncio
    async def test_capabilities_returns_json(self, server, mock_client):
        result = await server.call_tool("provider.capabilities", {})
        content_list = result[0]
        text = content_list[0].text
        data = json.loads(text)
        assert data["provider"] == "gemini"
        assert "provider.generate" in data["tools"]
        assert data["features"]["json_mode"] is True

    @pytest.mark.asyncio
    async def test_generate_with_history_returns_json(self, server, mock_client):
        mock_client.models.generate_content.return_value = _make_response("Multi-turn")
        messages = [{"role": "user", "content": "Hi"}]
        result = await server.call_tool(
            "provider.generate_with_history",
            {"messages": messages},
        )
        content_list = result[0]
        text = content_list[0].text
        data = json.loads(text)
        assert data["text"] == "Multi-turn"

    @pytest.mark.asyncio
    async def test_generate_passes_model_parameter(self, server, mock_client):
        await server.call_tool(
            "provider.generate",
            {"prompt": "Hello", "model": "gemini-2.0-flash"},
        )
        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-2.0-flash"

    @pytest.mark.asyncio
    async def test_generate_json_mode(self, server, mock_client):
        await server.call_tool(
            "provider.generate",
            {"prompt": "JSON please", "json_mode": True},
        )
        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["config"]["response_mime_type"] == "application/json"
