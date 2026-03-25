# aurarouter-gemini

Google Gemini MCP provider server for [AuraRouter](https://github.com/AuraCore-Dynamics/aurarouter).

Exposes Google Gemini models (2.5 Pro, 2.5 Flash, 2.0 Flash) as an MCP provider that AuraRouter can discover and route to automatically.

## Installation

```bash
pip install aurarouter-gemini
```

## Configuration

Set your API key via environment variable:

```bash
export GEMINI_API_KEY="your-api-key"
# or
export GOOGLE_API_KEY="your-api-key"
```

## Usage

### As an MCP server

```bash
python -m aurarouter_gemini
```

### Auto-discovery

When installed alongside AuraRouter, the package registers itself via the `aurarouter.providers` entry point group. AuraRouter's `ProviderCatalog` will discover it automatically.

## License

Apache-2.0
