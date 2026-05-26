"""
isa_phm — Python wrapper for ISA-PHM datasets.

Primary entry point: :class:`ISAWrapper`.

Quick start::

    from isa_phm import ISAWrapper

    wrapper = ISAWrapper("i_investigation.json", data_root="data/")
    df = wrapper.study("Case 01").assay("a_st01_se01").load_dataframe()
"""

from .wrapper import ISAWrapper
from .plotter import ISAPlotter, PlotConfig
from .proxy import AssayGroup
from .semantic import SemanticNormalizer
from .tools import ToolRegistry
from .schemas import (
    DataLoadMetadata,
    DatasetValidationReport,
    PublicationModel,
    SemanticDiagnostics,
    SemanticField,
    SemanticManifest,
    ValidationIssue,
)
from .errors import (
    ISAPHMError,
    ParseError,
    ValidationError,
    PreprocessingError,
    ExtractionError,
    DataFileError,
    StudyNotFoundError,
    AssayNotFoundError,
    RunNotFoundError,
    AmbiguousRunError,
    PlotError,
    ExportError,
)

__all__ = [
    "ISAWrapper",
    "ISAPlotter",
    "PlotConfig",
    "AssayGroup",
    "SemanticNormalizer",
    "ToolRegistry",
    "DataLoadMetadata",
    "DatasetValidationReport",
    "PublicationModel",
    "SemanticField",
    "SemanticDiagnostics",
    "SemanticManifest",
    "ValidationIssue",
    "ISAPHMError",
    "ParseError",
    "ValidationError",
    "PreprocessingError",
    "ExtractionError",
    "DataFileError",
    "StudyNotFoundError",
    "AssayNotFoundError",
    "RunNotFoundError",
    "AmbiguousRunError",
    "PlotError",
    "ExportError",
]
__version__ = "0.2.0"
