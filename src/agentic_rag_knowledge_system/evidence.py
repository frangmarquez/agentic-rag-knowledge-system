"""Citation-ready evidence records."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_rag_knowledge_system.ingestion.models import Metadata
from agentic_rag_knowledge_system.retrieval.schemas import (
    RetrievalHit,
    RetrievalResult,
)


class EvidenceItem(BaseModel):
    """One retrieved chunk prepared for citation-grounded generation."""

    model_config = ConfigDict(frozen=True)

    citation_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    rank: int = Field(ge=1)
    retriever: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    score: float
    quote: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("retriever", "quote")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @classmethod
    def from_hit(
        cls,
        hit: RetrievalHit,
        *,
        rank: int,
        retriever: str,
    ) -> "EvidenceItem":
        return cls(
            citation_id=f"E{rank}",
            rank=rank,
            retriever=retriever,
            chunk_id=hit.chunk_id,
            source_id=hit.source_id,
            source_uri=hit.source_uri,
            chunk_index=hit.chunk_index,
            score=hit.score,
            quote=hit.text,
            metadata=dict(hit.metadata),
        )


def build_evidence_items(
    result: RetrievalResult,
    *,
    retriever: str,
) -> tuple[EvidenceItem, ...]:
    """Convert ranked retrieval hits into citation-ready evidence."""

    return tuple(
        EvidenceItem.from_hit(hit, rank=rank, retriever=retriever)
        for rank, hit in enumerate(result.hits, start=1)
    )


class EvidenceContext(BaseModel):
    """Rendered evidence block plus included/omitted citation IDs."""

    model_config = ConfigDict(frozen=True)

    text: str
    evidence: tuple[EvidenceItem, ...] = Field(default_factory=tuple)
    omitted_citation_ids: tuple[str, ...] = Field(default_factory=tuple)
    max_chars: int = Field(ge=1)


def build_evidence_context(
    evidence: tuple[EvidenceItem, ...] | list[EvidenceItem],
    *,
    max_chars: int,
) -> EvidenceContext:
    """Render evidence into a bounded context block."""

    # ponytail: character budget; switch to model tokenizer after generator choice.
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")

    items = tuple(evidence)
    blocks: list[str] = []
    included: list[EvidenceItem] = []
    omitted: list[str] = []
    used_chars = 0

    for index, item in enumerate(items):
        separator = "\n\n" if blocks else ""
        remaining = max_chars - used_chars - len(separator)
        if remaining <= 0:
            omitted.extend(other.citation_id for other in items[index:])
            break

        block = _render_evidence_block(item)
        if len(block) > remaining:
            block = _render_evidence_block(item, max_chars=remaining)
            if block is None:
                omitted.extend(other.citation_id for other in items[index:])
                break
            omitted.extend(other.citation_id for other in items[index + 1 :])

        blocks.append(separator + block)
        included.append(item)
        used_chars += len(separator) + len(block)

        if omitted:
            break

    return EvidenceContext(
        text="".join(blocks),
        evidence=tuple(included),
        omitted_citation_ids=tuple(omitted),
        max_chars=max_chars,
    )


def _render_evidence_block(
    item: EvidenceItem,
    *,
    max_chars: int | None = None,
) -> str | None:
    header = (
        f"[{item.citation_id}] source={item.source_uri} "
        f"chunk={item.chunk_index} retriever={item.retriever} "
        f"score={item.score:.6g}"
    )
    quote = item.quote.strip()
    if max_chars is None:
        return f"{header}\n{quote}"

    quote_budget = max_chars - len(header) - 1
    if quote_budget < 1:
        return None
    return f"{header}\n{_truncate(quote, quote_budget)}"


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3].rstrip() + "..."
