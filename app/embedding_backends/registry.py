"""Simple registry for embedding backend instances."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.common.errors import BackendAlreadyRegisteredError, BackendNotFoundError
from app.embedding_backends.base import EmbeddingBackend


@dataclass(slots=True)
class EmbeddingBackendRegistry:
    """Keeps backend instances discoverable by a stable backend name."""

    _items: dict[str, EmbeddingBackend] = field(default_factory=dict)

    def register(self, backend: EmbeddingBackend) -> None:
        if backend.backend_name in self._items:
            raise BackendAlreadyRegisteredError(
                f"Backend `{backend.backend_name}` is already registered."
            )
        self._items[backend.backend_name] = backend

    def get(self, backend_name: str) -> EmbeddingBackend:
        try:
            return self._items[backend_name]
        except KeyError as exc:
            raise BackendNotFoundError(
                f"Backend `{backend_name}` is not registered."
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._items.keys())
