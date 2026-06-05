"""Exception hierarchy for PyNeuraLogic.

All library-raised exceptions inherit from :class:`NeuraLogicError` so that
callers can write a single ``except NeuraLogicError`` block instead of
catching the generic ``Exception``.
"""


class NeuraLogicError(Exception):
    """Base class for all errors raised by PyNeuraLogic."""


class ModelError(NeuraLogicError):
    """Raised when the model definition or build process fails."""


class DatasetError(NeuraLogicError):
    """Raised when dataset preparation or grounding fails."""


class BackendError(NeuraLogicError):
    """Raised when the Java backend encounters an internal error."""


class ConfigurationError(NeuraLogicError):
    """Raised when settings, JVM, or environment configuration is invalid."""
