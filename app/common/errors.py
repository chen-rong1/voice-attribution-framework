"""Framework-specific exception types."""


class VoiceAttributionError(Exception):
    """Base exception for the framework."""


class BackendAlreadyRegisteredError(VoiceAttributionError):
    """Raised when the same backend name is registered twice."""


class BackendNotFoundError(VoiceAttributionError):
    """Raised when a backend cannot be resolved from the registry."""


class BackendNotLoadedError(VoiceAttributionError):
    """Raised when extraction is requested before a backend is loaded."""


class InvalidAudioInputError(VoiceAttributionError):
    """Raised when an audio chunk does not meet the normalization contract."""


class EmbeddingExtractionError(VoiceAttributionError):
    """Raised when a backend fails to return a valid embedding."""
