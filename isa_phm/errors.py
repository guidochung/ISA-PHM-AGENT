"""
Custom exception hierarchy for the ISA-PHM wrapper.

All public-facing errors subclass ISAPHMError so callers can catch
everything with a single except clause when needed.
"""


class ISAPHMError(Exception):
    """Base exception for all ISA-PHM wrapper errors."""


class ParseError(ISAPHMError):
    """Raised when the ISA-JSON cannot be parsed (malformed JSON)."""


class ValidationError(ISAPHMError):
    """Raised when validation fails (schema and/or strict semantic checks)."""


class PreprocessingError(ISAPHMError):
    """Raised when ISA-JSON has an unfixable structural problem."""


class ExtractionError(ISAPHMError):
    """Raised when valid ISA-JSON cannot be mapped to ISA-PHM domain models.

    Typical causes:
    - Unresolvable @id reference after repair
    - Study has no assays
    - Missing required fields in the PHM subset
    """


class DataFileError(ISAPHMError):
    """Raised when a data file cannot be loaded.

    Includes: file not found, encoding failure, malformed CSV,
    unsupported file format, empty path.
    """


class StudyNotFoundError(ISAPHMError):
    """Raised when a study_id does not match any study in the dataset."""


class AssayNotFoundError(ISAPHMError):
    """Raised when an assay_id does not match any assay in a study."""


class RunNotFoundError(ISAPHMError):
    """Raised when a run_id does not match any run in an assay."""


class AmbiguousRunError(ISAPHMError):
    """Raised when load_dataframe() is called on a multi-run assay without a run_id."""


class PlotError(ISAPHMError):
    """Raised when a plot cannot be produced.

    Typical causes:
    - Empty or all-NaN DataFrame column
    - Non-numeric column where numeric is required
    - Fewer than 2 rows
    - Missing required parameter (e.g., fs for FFT)
    """


class ExportError(ISAPHMError):
    """Raised when exporting a cleaned DataFrame fails (permission, path)."""
