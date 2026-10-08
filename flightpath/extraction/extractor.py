"""Evidence extractor for long documents (earnings calls, 10-Q MD&A, investor decks)."""
from __future__ import annotations

from .documents import Document
from .llm import LLMClient
from .schema import Evidence

SYSTEM_PROMPT = """\
You extract structured evidence about an airline's business drivers from documents.
Rules:
- Every item must include a verbatim quote from the text.
- Only fill numeric_value if the number appears verbatim in the quote.
- Never estimate, compute, or infer numbers.
- Return JSON matching the Evidence schema.
"""  # TODO: refine; add few-shot examples


class EvidenceExtractor:
    def __init__(self, llm: LLMClient, min_confidence: float = 0.6):
        raise NotImplementedError

    def extract(self, doc: Document) -> list[Evidence]:
        """TODO: chunk -> llm.complete_json -> validate -> verify -> filter by confidence."""
        raise NotImplementedError

    def verify(self, ev: Evidence, doc: Document) -> bool:
        """Hallucination guard: quote must appear in doc text; numeric_value must appear in quote.
        TODO: implement (fuzzy match on whitespace)."""
        raise NotImplementedError


def evidence_to_priors(evidence: list[Evidence]) -> dict:
    """Map evidence to parameter beliefs, e.g. widen/shift the yield prior when management
    flags pricing pressure. Keep this mapping explicit and documented (interpretability).
    TODO: design mapping rules.
    """
    raise NotImplementedError
