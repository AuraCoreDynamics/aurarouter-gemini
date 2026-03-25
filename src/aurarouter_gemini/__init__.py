"""aurarouter-gemini -- Google Gemini MCP provider for AuraRouter."""

__version__ = "0.5.1"
__package_name__ = "aurarouter-gemini"


def get_provider_metadata():
    """Entry point callable for AuraRouter provider discovery.

    Returns a ProviderMetadata-compatible dict (or the dataclass itself
    if aurarouter is installed).  The catalog calls this function when
    it loads the ``aurarouter.providers`` entry point.
    """
    try:
        from aurarouter.providers.protocol import ProviderMetadata

        return ProviderMetadata(
            name="gemini",
            provider_type="mcp",
            version=__version__,
            description="Google Gemini models via MCP (2.5 Pro, 2.5 Flash, 2.0 Flash)",
            command=["python", "-m", "aurarouter_gemini"],
            requires_config=["api_key"],
            homepage="https://github.com/AuraCore-Dynamics/aurarouter-gemini",
        )
    except ImportError:
        # aurarouter is not installed -- return a plain dict so the
        # package can still be introspected without the core.
        return {
            "name": "gemini",
            "provider_type": "mcp",
            "version": __version__,
            "description": "Google Gemini models via MCP (2.5 Pro, 2.5 Flash, 2.0 Flash)",
            "command": ["python", "-m", "aurarouter_gemini"],
            "requires_config": ["api_key"],
            "homepage": "https://github.com/AuraCore-Dynamics/aurarouter-gemini",
        }
