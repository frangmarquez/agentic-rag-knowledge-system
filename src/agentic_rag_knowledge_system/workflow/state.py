"""Typed state for the future agent workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from agentic_rag_knowledge_system.evidence import EvidenceContext, EvidenceItem
from agentic_rag_knowledge_system.retrieval.schemas import RetrievalResult


class QueryAnalysis(BaseModel):
    """Result of analyzing the user query before retrieval."""

    model_config = ConfigDict(frozen=True)

    normalized_query: str = Field(min_length=1)
    needs_retrieval: bool
    reason: str = Field(min_length=1)

    @field_validator("normalized_query", "reason")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


class ReformulatedQuery(BaseModel):
    """One retrieval query derived from the original user query."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("query", "reason")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


class GroundingVerdict(BaseModel):
    """Grounding check result for a generated answer."""

    model_config = ConfigDict(frozen=True)

    status: Literal["supported", "unsupported", "insufficient_evidence"]
    reason: str = Field(min_length=1)
    unsupported_claims: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must not be blank")
        return stripped

    @field_validator("unsupported_claims")
    @classmethod
    def strip_claims(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(value.strip() for value in values if value.strip())


class AgentState(BaseModel):
    """Explicit state passed between future LangGraph nodes."""

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(default="ad_hoc", min_length=1)
    user_query: str = Field(min_length=1)
    analysis: QueryAnalysis | None = None
    reformulated_queries: tuple[ReformulatedQuery, ...] = Field(default_factory=tuple)
    retrieval_result: RetrievalResult | None = None
    evidence: tuple[EvidenceItem, ...] = Field(default_factory=tuple)
    evidence_context: EvidenceContext | None = None
    answer: str | None = None
    grounding: GroundingVerdict | None = None
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=1, ge=0)

    @field_validator("query_id", "user_query")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("answer")
    @classmethod
    def strip_optional_answer(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("answer must not be blank")
        return stripped

    @model_validator(mode="after")
    def validate_state(self) -> "AgentState":
        if self.retry_count > self.max_retries:
            raise ValueError("retry_count must be <= max_retries")

        if self.evidence_context is not None:
            known_ids = {item.citation_id for item in self.evidence}
            context_ids = {item.citation_id for item in self.evidence_context.evidence}
            if not context_ids.issubset(known_ids):
                raise ValueError("evidence_context must use state evidence")

        return self

    @computed_field
    @property
    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries
