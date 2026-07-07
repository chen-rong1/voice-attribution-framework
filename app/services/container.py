"""A thin container used to wire framework-level services together."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.embedding_backends.registry import EmbeddingBackendRegistry


@dataclass(slots=True)
class FrameworkContainer:
    """The first project-level assembly point for shared registries."""

    backend_registry: EmbeddingBackendRegistry = field(default_factory=EmbeddingBackendRegistry)
