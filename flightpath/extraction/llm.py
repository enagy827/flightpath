"""Provider-agnostic LLM client (OpenAI or Anthropic), chosen in config/settings.yaml.

Longaeva provides OpenAI credit ($300): use it for HIGH-VOLUME classification
(reviews, forum posts, news, job postings) with a small model + the Batch API
(discounted, async). Reserve larger models for earnings calls and event analysis.

Track spend: log tokens per call to data/interim/llm_usage.csv and stop at a budget cap.
"""
from __future__ import annotations

from typing import Any


class LLMClient:
    def __init__(self, provider: str, model: str, budget_usd: float | None = None):
        # TODO: init openai.OpenAI() or anthropic.Anthropic() from .env keys
        raise NotImplementedError

    def complete_json(self, system: str, user: str, schema: dict) -> dict[str, Any]:
        """Single call returning JSON that validates against schema (structured outputs). TODO."""
        raise NotImplementedError

    def submit_batch(self, requests: list[dict], job_name: str) -> str:
        """Submit an async batch job (OpenAI Batch API). Returns batch id. TODO."""
        raise NotImplementedError

    def collect_batch(self, batch_id: str) -> list[dict]:
        """Poll + download batch results. TODO."""
        raise NotImplementedError

    def log_usage(self, response: Any) -> None:
        """Append tokens + estimated cost; raise if budget exceeded. TODO."""
        raise NotImplementedError
