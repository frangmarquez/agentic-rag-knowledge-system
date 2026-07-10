"""Typed ingestion models and stable identifier helpers."""

from __future__ import annotations

import hashlib
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MetadataValue = str | int | float | bool | None
Metadata = dict[str, MetadataValue]

_SHA256_PATTERN = r"^[a-f0-9]{64}$"


def normalize_text(content: str) -> str:
    """Normalize text before hashing and chunking."""

    return content.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def compute_content_hash(content: str) -> str:
    """Return a SHA-256 hash for normalized UTF-8 text."""

    normalized = normalize_text(content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_source_id(source_uri: str) -> str:
    """Return a stable source identifier from a normalized source URI."""

    digest = hashlib.sha256(source_uri.encode("utf-8")).hexdigest()
    return f"src_{digest[:16]}"


def stable_chunk_id(
    *,
    source_id: str,
    chunk_index: int,
    start_char: int,
    end_char: int,
    content_hash: str,
) -> str:
    """Return a stable chunk identifier from lineage and content."""

    identity = f"{source_id}:{chunk_index}:{start_char}:{end_char}:{content_hash}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"chk_{digest[:20]}"


class SourceDocument(BaseModel):
    """Normalized document ready for deterministic chunking."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    content_hash: str = Field(pattern=_SHA256_PATTERN)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        normalized = normalize_text(str(value))
        if not normalized.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return normalized

    @model_validator(mode="after")
    def validate_content_hash(self) -> Self:
        expected_hash = compute_content_hash(self.content)
        if self.content_hash != expected_hash:
            raise ValueError("content_hash must match normalized content")
        return self


class DocumentChunk(BaseModel):
    """A citation-addressable chunk derived from a source document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    content_hash: str = Field(pattern=_SHA256_PATTERN)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("text", mode="before")
    @classmethod
    def normalize_chunk_text(cls, value: str) -> str:
        normalized = normalize_text(str(value))
        if not normalized.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return normalized

    @model_validator(mode="after")
    def validate_offsets_and_hash(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")

        expected_hash = compute_content_hash(self.text)
        if self.content_hash != expected_hash:
            raise ValueError("content_hash must match normalized text")

        return self
