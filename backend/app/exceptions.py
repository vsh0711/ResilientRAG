"""
Typed exceptions for failure modes that are operational (a dependency is
down or unreachable) rather than bugs, so the API layer can return a
clean, specific error to the client instead of a raw 500 with an internal
stack trace leaked into the response body.
"""
from __future__ import annotations


class RetrievalBackendError(Exception):
    """Raised when the vector store or embedding backend cannot complete
    an operation — e.g. Qdrant is unreachable, or the embedding model
    can't be loaded (no network to pull it, out of disk, etc). This is an
    infrastructure failure, not a bad document or a bad query, so it maps
    to HTTP 503 rather than 400/500."""


class PayloadTooLargeError(Exception):
    """Raised when an uploaded file exceeds the configured size limit."""
