from app.common.errors import BackendAlreadyRegisteredError, BackendNotFoundError
from app.embedding_backends.base import EmbeddingBackend
from app.embedding_backends.models import EmbeddingRequest, EmbeddingResult
from app.embedding_backends.registry import EmbeddingBackendRegistry


class DummyBackend(EmbeddingBackend):
    backend_name = "dummy-backend"
    backend_version = "0.1.0"
    feature_version = "fbank80_v1"
    embedding_dim = 3

    def load(self) -> None:
        self._loaded = True

    def extract_embedding(self, request: EmbeddingRequest) -> EmbeddingResult:
        raise NotImplementedError


def test_registry_register_and_list_names() -> None:
    registry = EmbeddingBackendRegistry()
    registry.register(DummyBackend())
    assert registry.names() == ["dummy-backend"]


def test_registry_rejects_duplicate_names() -> None:
    registry = EmbeddingBackendRegistry()
    registry.register(DummyBackend())
    try:
        registry.register(DummyBackend())
    except BackendAlreadyRegisteredError:
        pass
    else:
        raise AssertionError("expected duplicate backend registration to fail")


def test_registry_raises_for_unknown_backend() -> None:
    registry = EmbeddingBackendRegistry()
    try:
        registry.get("missing")
    except BackendNotFoundError:
        pass
    else:
        raise AssertionError("expected unknown backend lookup to fail")
