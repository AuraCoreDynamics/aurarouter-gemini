"""GeminiProvider -- wraps the google-genai SDK for text generation."""

from __future__ import annotations

import os
from typing import Any

from google import genai

from aurarouter_gemini.models import GEMINI_MODELS, get_model_info


class GeminiProvider:
    """Thin wrapper around the Google GenAI SDK.

    Reads the API key from the ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``
    environment variable (constructor parameter takes precedence).
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gemini-2.5-flash",
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Gemini API key is required. Set the GEMINI_API_KEY or "
                "GOOGLE_API_KEY environment variable, or pass api_key= "
                "to GeminiProvider()."
            )
        self._api_key = resolved_key
        self._default_model = default_model
        self._client = genai.Client(api_key=self._api_key)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: str = "",
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Single-shot text generation.

        Returns a dict with keys: text, model_id, input_tokens,
        output_tokens, context_limit.
        """
        model_id = model or self._default_model
        model_info = get_model_info(model_id)
        context_limit = model_info["context_window"] if model_info else 0

        config: dict[str, Any] = {}
        if json_mode:
            config["response_mime_type"] = "application/json"

        generation_config = config if config else None

        response = self._client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=generation_config,
        )

        text = response.text or ""
        input_tokens = 0
        output_tokens = 0

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        return {
            "text": text,
            "model_id": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "context_limit": context_limit,
        }

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        model: str = "",
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Multi-turn generation using a message history.

        Messages are dicts with ``role`` (``"user"`` or ``"model"``) and
        ``content`` keys.  The ``system_prompt`` is prepended as a user
        message if provided.

        Returns the same dict shape as :meth:`generate`.
        """
        model_id = model or self._default_model
        model_info = get_model_info(model_id)
        context_limit = model_info["context_window"] if model_info else 0

        # Build contents list for the SDK
        contents: list[dict[str, Any]] = []

        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": system_prompt}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})

        for msg in messages:
            role = msg.get("role", "user")
            # Map common role names to Gemini roles
            if role in ("assistant", "system"):
                role = "model"
            content_text = msg.get("content", "")
            contents.append({"role": role, "parts": [{"text": content_text}]})

        config: dict[str, Any] = {}
        if json_mode:
            config["response_mime_type"] = "application/json"

        generation_config = config if config else None

        response = self._client.models.generate_content(
            model=model_id,
            contents=contents,
            config=generation_config,
        )

        text = response.text or ""
        input_tokens = 0
        output_tokens = 0

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        return {
            "text": text,
            "model_id": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "context_limit": context_limit,
        }

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """Return the catalog of available Gemini models."""
        return [dict(m) for m in GEMINI_MODELS]
