"""Load and chunk source documents (earnings call transcripts, 10-Q MD&A, investor decks)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class Document:
    doc_id: str
    kind: str            # "transcript" | "10q" | "10k" | "8k" | "investor_day"
    published: date      # becomes Evidence.available_at
    text: str


def load_documents(kind: str | None = None) -> list[Document]:
    """TODO: read from data/raw/documents/; decide transcript source (free/licensed - cite it)."""
    raise NotImplementedError


def chunk(doc: Document, max_tokens: int = 3000) -> list[str]:
    """TODO: split by speaker turn for transcripts, by section for filings."""
    raise NotImplementedError
