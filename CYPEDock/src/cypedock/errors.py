"""Domain-specific exceptions."""


class CYPEdockError(RuntimeError):
    """Base exception for expected workflow failures."""


class ConfigurationError(CYPEdockError):
    """The configuration is invalid or incomplete."""


class InputError(CYPEdockError):
    """An input file or molecular record is invalid."""


class AlignmentError(CYPEdockError):
    """A reference cannot be aligned safely to a conformation."""


class ExternalToolError(CYPEdockError):
    """An external command failed or produced invalid output."""
