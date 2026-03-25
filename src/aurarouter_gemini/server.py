"""FastMCP server factory for the Gemini provider.

Exposes four MCP tools that implement the AuraRouter Provider Protocol:

- ``provider.generate`` -- Single-shot text generation
- ``provider.list_models`` -- Enumerate available Gemini models
- ``provider.generate_with_history`` -- Multi-turn generation
- ``provider.capabilities`` -- Advertise provider features
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from aurarouter_gemini.provider import GeminiProvider

# Lazy-initialised singleton -- created on first tool call so that
# the server can start without an API key (useful for --help).
_provider: GeminiProvider | None = None


def _get_provider() -> GeminiProvider:
    """Return (and lazily create) the singleton GeminiProvider."""
    global _provider
    if _provider is None:
        _provider = GeminiProvider()
    return _provider


def create_server() -> FastMCP:
    """Build and return a configured FastMCP server instance."""
    mcp = FastMCP("aurarouter-gemini")

    # ------------------------------------------------------------------
    # provider.generate
    # ------------------------------------------------------------------
    @mcp.tool(name="provider.generate")
    def provider_generate(
        prompt: str,
        model: str = "",
        json_mode: bool = False,
    ) -> str:
        """Generate text from a prompt using a Gemini model.

        Args:
            prompt: The input text prompt.
            model: Model ID to use (default: gemini-2.5-flash).
            json_mode: If true, request JSON-formatted output.

        Returns:
            JSON string with text, model_id, token counts, and context_limit.
        """
        provider = _get_provider()
        result = provider.generate(prompt=prompt, model=model, json_mode=json_mode)
        return json.dumps(result)

    # ------------------------------------------------------------------
    # provider.list_models
    # ------------------------------------------------------------------
    @mcp.tool(name="provider.list_models")
    def provider_list_models() -> str:
        """List available Gemini models.

        Returns:
            JSON string with a list of model metadata dicts.
        """
        provider = _get_provider()
        models = provider.list_models()
        return json.dumps(models)

    # ------------------------------------------------------------------
    # provider.generate_with_history
    # ------------------------------------------------------------------
    @mcp.tool(name="provider.generate_with_history")
    def provider_generate_with_history(
        messages: list[dict[str, Any]],
        system_prompt: str = "",
        model: str = "",
        json_mode: bool = False,
    ) -> str:
        """Multi-turn generation with conversation history.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}.
            system_prompt: Optional system instruction.
            model: Model ID to use (default: gemini-2.5-flash).
            json_mode: If true, request JSON-formatted output.

        Returns:
            JSON string with text, model_id, token counts, and context_limit.
        """
        provider = _get_provider()
        result = provider.generate_with_history(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            json_mode=json_mode,
        )
        return json.dumps(result)

    # ------------------------------------------------------------------
    # provider.capabilities
    # ------------------------------------------------------------------
    @mcp.tool(name="provider.capabilities")
    def provider_capabilities() -> str:
        """Advertise provider capabilities.

        Returns:
            JSON string describing supported tools and features.
        """
        return json.dumps({
            "provider": "gemini",
            "version": "0.5.1",
            "tools": [
                "provider.generate",
                "provider.list_models",
                "provider.generate_with_history",
                "provider.capabilities",
            ],
            "features": {
                "json_mode": True,
                "multi_turn": True,
                "streaming": False,
                "token_counting": True,
            },
        })

    return mcp
