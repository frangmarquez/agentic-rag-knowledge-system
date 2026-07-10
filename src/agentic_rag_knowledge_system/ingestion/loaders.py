"""Local text and Markdown document loaders."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from agentic_rag_knowledge_system.ingestion.models import (
    Metadata,
    MetadataValue,
    SourceDocument,
    compute_content_hash,
    normalize_text,
    stable_source_id,
)

SUPPORTED_TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})


def load_text_file(
    path: str | Path,
    *,
    corpus_root: str | Path | None = None,
    metadata: Mapping[str, MetadataValue] | None = None,
    encoding: str = "utf-8",
) -> SourceDocument:
    """Load one UTF-8 text/Markdown file into a normalized document."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    if file_path.suffix.lower() not in SUPPORTED_TEXT_SUFFIXES:
        raise ValueError(f"Unsupported text document suffix: {file_path.suffix}")

    raw_content = file_path.read_text(encoding=encoding)
    content = normalize_text(raw_content)
    if not content.strip():
        raise ValueError(f"Refusing to ingest empty document: {file_path}")

    source_uri = _source_uri_for(file_path, corpus_root=corpus_root)
    content_hash = compute_content_hash(content)
    document_metadata = _default_metadata_for(file_path)
    document_metadata.update(dict(metadata or {}))

    return SourceDocument(
        source_id=stable_source_id(source_uri),
        source_uri=source_uri,
        title=_infer_title(content, fallback=file_path.stem),
        content=content,
        content_hash=content_hash,
        metadata=document_metadata,
    )


def load_text_directory(
    directory: str | Path,
    *,
    recursive: bool = True,
    metadata: Mapping[str, MetadataValue] | None = None,
    encoding: str = "utf-8",
) -> list[SourceDocument]:
    """Load supported text documents from a directory in deterministic order."""

    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(root)

    pattern = "**/*" if recursive else "*"
    paths = sorted(
        (
            path
            for path in root.glob(pattern)
            if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )

    return [
        load_text_file(path, corpus_root=root, metadata=metadata, encoding=encoding)
        for path in paths
    ]


def _source_uri_for(path: Path, *, corpus_root: str | Path | None) -> str:
    if corpus_root is None:
        return path.resolve().as_posix()

    return path.resolve().relative_to(Path(corpus_root).resolve()).as_posix()


def _default_metadata_for(path: Path) -> Metadata:
    return {
        "file_name": path.name,
        "file_extension": path.suffix.lower(),
    }


def _infer_title(content: str, *, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("<!--", "<", "---")):
            continue
        if stripped.startswith("# "):
            return stripped.removeprefix("# ").strip() or fallback

    return fallback
