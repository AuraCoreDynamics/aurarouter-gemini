"""Entry point for ``python -m aurarouter_gemini``."""

from aurarouter_gemini.server import create_server


def main():
    """Create and run the Gemini MCP provider server."""
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
