"""Gemini model catalog -- known models and their metadata."""

GEMINI_MODELS: list[dict] = [
    {
        "model_id": "gemini-2.5-pro",
        "display_name": "Gemini 2.5 Pro",
        "context_window": 1048576,
        "cost_per_1m_input": 1.25,
        "cost_per_1m_output": 10.0,
        "capabilities": ["code", "reasoning", "chat"],
    },
    {
        "model_id": "gemini-2.5-flash",
        "display_name": "Gemini 2.5 Flash",
        "context_window": 1048576,
        "cost_per_1m_input": 0.15,
        "cost_per_1m_output": 0.60,
        "capabilities": ["code", "reasoning", "chat"],
    },
    {
        "model_id": "gemini-2.0-flash",
        "display_name": "Gemini 2.0 Flash",
        "context_window": 1048576,
        "cost_per_1m_input": 0.10,
        "cost_per_1m_output": 0.40,
        "capabilities": ["code", "chat"],
    },
]


def get_model_info(model_id: str) -> dict | None:
    """Look up a model by its ID. Returns None if not found."""
    for model in GEMINI_MODELS:
        if model["model_id"] == model_id:
            return model
    return None


def get_default_model() -> str:
    """Return the default model ID."""
    return "gemini-2.5-flash"
