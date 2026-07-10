"""Agentic RAG Knowledge System package."""

from importlib.metadata import PackageNotFoundError, version

_DISTRIBUTION_NAME = "agentic-rag-knowledge-system"

try:
    __version__ = version(_DISTRIBUTION_NAME)
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
