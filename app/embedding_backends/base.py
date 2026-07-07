"""Abstract backend interface used by all embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.common.errors import BackendNotLoadedError
from app.embedding_backends.models import EmbeddingRequest, EmbeddingResult


class EmbeddingBackend(ABC):
    """Unified backend interface for ECAPA-style embedding extractors."""

    backend_name: str
    backend_version: str
    feature_version: str
    embedding_dim: int

    def __init__(self) -> None:
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Initialize the backend and any model resources it needs."""

    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def extract_embedding(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Extract one embedding from a standardized request."""

    def extract_embeddings(
        self,
        requests: list[EmbeddingRequest],
    ) -> list[EmbeddingResult]:
        """Fallback batch implementation using repeated single-item extraction."""
        self.ensure_loaded()
        return [self.extract_embedding(request) for request in requests]

    def unload(self) -> None:
        """Release backend resources if necessary."""
        self._loaded = False

    def ensure_loaded(self) -> None:
        if not self._loaded:
            raise BackendNotLoadedError(
                f"Backend `{self.backend_name}` has not been loaded yet."
            )
