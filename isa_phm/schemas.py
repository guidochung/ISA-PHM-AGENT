"""
Pydantic domain models for the ISA-PHM wrapper.

Three tiers:
- Domain models  : InvestigationModel, StudyModel, AssayModel, RunRecord, ...
                   Built by MetadataExtractor; represent the parsed ISA-PHM dataset.
- Summary models : StudySummary, AssayOverview, ...
                   Returned by proxy methods for inspection without loading data.
- Report models  : MissingValuesReport, RepairAction, RepairLog
                   Returned by analysis and preprocessing methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


@dataclass
class RepairAction:
    """One auto-fix action applied by ISAPreprocessor."""

    severity: Literal["INFO", "WARNING"]
    entity_type: str  # "DataFile", "Study", "@id", etc.
    entity_id: str
    field_name: str
    old_value: Any
    new_value: Any
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.entity_type}({self.entity_id}).{self.field_name}: {self.message}"


@dataclass
class RepairLog:
    """Collects all repair actions applied during preprocessing."""

    actions: list[RepairAction] = field(default_factory=list)
    fatal_error: str | None = None

    def add(self, action: RepairAction) -> None:
        self.actions.append(action)

    def warnings(self) -> list[RepairAction]:
        return [a for a in self.actions if a.severity == "WARNING"]

    def summary(self) -> str:
        if not self.actions:
            return "No repairs applied."
        lines = [str(a) for a in self.actions]
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.actions)


# ---------------------------------------------------------------------------
# ISA-PHM domain models (built by MetadataExtractor)
# ---------------------------------------------------------------------------


class ParameterValue(BaseModel):
    """A single resolved parameter value from a process sequence."""

    parameter_name: str
    value: str | int | float | bool | None
    unit: str | None = None

    def __str__(self) -> str:
        if self.unit:
            return f"{self.parameter_name}: {self.value} {self.unit}"
        return f"{self.parameter_name}: {self.value}"


class DataFile(BaseModel):
    """A data file entry from an ISA assay."""

    id: str
    path: str  # Resolved absolute path string (may be empty for unnamed raw files)
    file_type: str  # "Raw Data File" or "Processed Data File"
    exists_at_extract: bool = False  # Snapshot captured during extraction

    @property
    def is_raw(self) -> bool:
        return "Raw" in self.file_type

    @property
    def is_processed(self) -> bool:
        return "Processed" in self.file_type

    @property
    def exists(self) -> bool:
        """
        Live file existence check.

        Unlike ``exists_at_extract``, this reflects current filesystem state.
        """
        if not self.path:
            return False
        try:
            return Path(self.path).exists()
        except OSError:
            return False

    @property
    def as_path(self) -> Path | None:
        return Path(self.path) if self.path else None


class ProtocolParameter(BaseModel):
    """A single parameter definition from a protocol."""

    id: str
    parameter_name: str


class ProtocolModel(BaseModel):
    """A measurement or processing protocol from a study."""

    id: str
    name: str
    protocol_type: str  # annotationValue of protocolType
    parameters: list[ProtocolParameter] = Field(default_factory=list)
    sensor_id: str | None = None  # Extracted from protocol comments


class SensorInfo(BaseModel):
    """Sensor metadata derived from the assay and its measurement protocol."""

    sensor_id: str | None = None  # UUID from protocol comment
    alias: str  # Assay filename used as sensor identifier (e.g., "a_st01_se01")
    technology_type: str  # e.g., "Accelerometer"
    technology_platform: str  # e.g., "Wilcoxon 786B-10"
    measurement_type: str  # e.g., "Vibration"


class FactorModel(BaseModel):
    """A study factor (experiment variable)."""

    factor_id: str  # @id string
    factor_name: str
    factor_type: str  # annotationValue (e.g., "Operating condition")
    unit: str | None = None
    description: str | None = None


class RunRecord(BaseModel):
    """One run: a pair of (measurement process, processing process) outputs."""

    run_id: str  # e.g., "run_01"
    run_number: int  # 1-based
    raw_file: DataFile | None = None
    processed_file: DataFile | None = None
    measurement_params: list[ParameterValue] = Field(default_factory=list)
    processing_params: list[ParameterValue] = Field(default_factory=list)
    factor_values: dict[str, Any] = Field(default_factory=dict)  # factor_name -> value


class AssayModel(BaseModel):
    """One assay = one sensor channel in ISA-PHM."""

    assay_id: str  # filename field, e.g. "a_st01_se01"
    sensor: SensorInfo
    measurement_protocol: ProtocolModel | None = None
    processing_protocol: ProtocolModel | None = None
    runs: list[RunRecord] = Field(default_factory=list)

    @property
    def run_count(self) -> int:
        return len(self.runs)

    @property
    def has_runs(self) -> bool:
        return len(self.runs) > 1

    def get_raw_files(self) -> list[DataFile]:
        return [r.raw_file for r in self.runs if r.raw_file is not None]

    def get_processed_files(self) -> list[DataFile]:
        return [r.processed_file for r in self.runs if r.processed_file is not None]

    def get_run(self, run_id: str) -> RunRecord | None:
        for r in self.runs:
            if r.run_id == run_id:
                return r
        return None


class ContactModel(BaseModel):
    """A person/contact from the investigation."""

    contact_id: str | None = None
    first_name: str
    last_name: str
    email: str
    affiliation: str
    roles: list[str] = Field(default_factory=list)
    orcid: str | None = None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


class PublicationModel(BaseModel):
    """A publication entry from the investigation-level ISA block."""

    title: str
    doi: str | None = None
    pubmed_id: str | None = None
    status: str | None = None
    author_tokens: list[str] = Field(default_factory=list)
    corresponding_author: str | None = None
    resolved_author_names: list[str] = Field(default_factory=list)
    resolved_author_emails: list[str] = Field(default_factory=list)
    unresolved_author_tokens: list[str] = Field(default_factory=list)


class StudyModel(BaseModel):
    """One study (experiment) in the ISA investigation."""

    study_id: str  # identifier UUID
    title: str
    description: str
    factors: list[FactorModel] = Field(default_factory=list)
    assays: list[AssayModel] = Field(default_factory=list)

    @property
    def run_count(self) -> int:
        if not self.assays:
            return 0
        return max(a.run_count for a in self.assays)

    @property
    def has_runs(self) -> bool:
        return self.run_count > 1

    def get_assay(self, assay_id: str) -> AssayModel | None:
        for a in self.assays:
            if a.assay_id == assay_id:
                return a
        return None


class InvestigationModel(BaseModel):
    """Root ISA investigation."""

    title: str
    description: str
    identifier: str
    experiment_type: str  # raw value from ISA comments, e.g. "diagnostic-single"
    studies: list[StudyModel] = Field(default_factory=list)
    contacts: list[ContactModel] = Field(default_factory=list)
    publications: list[PublicationModel] = Field(default_factory=list)

    def contacts_df(self) -> "pd.DataFrame":
        """Return investigation contacts as a DataFrame."""
        import pandas as pd

        rows = [
            {
                "contact_id": c.contact_id or "",
                "full_name": c.full_name,
                "first_name": c.first_name,
                "last_name": c.last_name,
                "email": c.email,
                "affiliation": c.affiliation,
                "roles": ", ".join(c.roles),
                "orcid": c.orcid or "",
            }
            for c in self.contacts
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "contact_id",
                "full_name",
                "first_name",
                "last_name",
                "email",
                "affiliation",
                "roles",
                "orcid",
            ],
        )

    def publications_df(self) -> "pd.DataFrame":
        """Return investigation publications as a DataFrame."""
        import pandas as pd

        rows = [
            {
                "title": p.title,
                "doi": p.doi or "",
                "pubmed_id": p.pubmed_id or "",
                "status": p.status or "",
                "author_tokens": "; ".join(p.author_tokens),
                "corresponding_author": p.corresponding_author or "",
                "resolved_author_names": "; ".join(p.resolved_author_names),
                "resolved_author_emails": "; ".join(p.resolved_author_emails),
                "unresolved_author_tokens": "; ".join(p.unresolved_author_tokens),
            }
            for p in self.publications
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "title",
                "doi",
                "pubmed_id",
                "status",
                "author_tokens",
                "corresponding_author",
                "resolved_author_names",
                "resolved_author_emails",
                "unresolved_author_tokens",
            ],
        )


# ---------------------------------------------------------------------------
# Summary / overview models (returned by proxy inspection methods)
# ---------------------------------------------------------------------------


class FactorSummary(BaseModel):
    factor_id: str
    factor_name: str
    factor_type: str
    unit: str | None = None


class RunSummary(BaseModel):
    run_id: str
    run_number: int
    factor_values: dict[str, Any] = Field(default_factory=dict)
    raw_file_path: str | None = None
    processed_file_path: str | None = None
    n_measurement_params: int = 0
    n_processing_params: int = 0


class AssaySummary(BaseModel):
    assay_id: str
    sensor_id: str | None = None
    sensor_alias: str
    technology_type: str
    measurement_type: str
    n_runs: int
    n_raw_files: int = 0
    n_processed_files: int = 0


class AssayOverview(BaseModel):
    assay_id: str
    sensor_id: str | None = None
    sensor_alias: str
    technology_type: str
    technology_platform: str
    measurement_type: str
    n_runs: int
    runs: list[RunSummary] = Field(default_factory=list)


class StudySummary(BaseModel):
    study_id: str
    title: str
    n_assays: int
    n_runs: int
    n_factors: int


class StudyOverview(BaseModel):
    study_id: str
    title: str
    description: str
    experiment_type: str
    n_assays: int
    run_count: int
    assays: list[AssaySummary] = Field(default_factory=list)
    factors: list[FactorSummary] = Field(default_factory=list)


class InvestigationOverview(BaseModel):
    title: str
    description: str
    identifier: str
    experiment_type: str
    n_studies: int
    n_contacts: int
    studies: list[StudySummary] = Field(default_factory=list)


class RunOverview(BaseModel):
    run_id: str
    run_number: int
    assay_id: str
    study_id: str
    sensor_alias: str
    measurement_type: str
    technology_type: str
    raw_file_path: str | None = None
    processed_file_path: str | None = None
    factor_values: dict[str, Any] = Field(default_factory=dict)
    measurement_params: list[dict[str, Any]] = Field(default_factory=list)
    processing_params: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# File-load metadata
# ---------------------------------------------------------------------------


class DataLoadMetadata(BaseModel):
    assay_id: str
    run_id: str
    requested_file_type: Literal["raw", "processed", "auto"]
    resolved_file_type: Literal["raw", "processed"]
    file_path: str
    from_cache: bool = False
    csv_engine: Literal["c", "python"] | None = None
    csv_sep: str | None = None
    csv_encoding: str | None = None
    csv_detection_source: str | None = None
    csv_bad_lines: Literal["error", "warn", "skip"] | None = None


# ---------------------------------------------------------------------------
# Semantic normalization models
# ---------------------------------------------------------------------------


class SemanticField(BaseModel):
    source_name: str
    source_kind: Literal["factor", "measurement_parameter", "processing_parameter"]
    semantic_key: str
    status: Literal["mapped", "unknown", "ambiguous"]
    confidence: float
    provenance: str
    candidates: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class SemanticDiagnostics(BaseModel):
    total_fields: int = 0
    mapped_fields: int = 0
    unknown_fields: int = 0
    ambiguous_fields: int = 0
    unknown_ratio: float = 0.0
    ambiguous_ratio: float = 0.0
    missing_override_fields: list[str] = Field(default_factory=list)
    strict_violations: list[str] = Field(default_factory=list)


class SemanticManifest(BaseModel):
    investigation_id: str
    study_factors: dict[str, list[SemanticField]] = Field(default_factory=dict)
    assay_measurement_params: dict[str, list[SemanticField]] = Field(default_factory=dict)
    assay_processing_params: dict[str, list[SemanticField]] = Field(default_factory=dict)
    diagnostics: SemanticDiagnostics = Field(default_factory=SemanticDiagnostics)


# ---------------------------------------------------------------------------
# Dataset validation report models
# ---------------------------------------------------------------------------


class ValidationIssue(BaseModel):
    code: str
    level: Literal["error", "warning", "info"]
    scope: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


class DatasetValidationReport(BaseModel):
    ok: bool
    n_errors: int = 0
    n_warnings: int = 0
    n_info: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Analysis report models
# ---------------------------------------------------------------------------


class MissingValuesReport(BaseModel):
    n_rows: int
    n_cols: int
    n_missing: int
    pct_missing: float
    by_column: dict[str, int] = Field(default_factory=dict)  # column_name -> n_missing


class OutlierReport(BaseModel):
    n_outliers: int
    pct_outliers: float
    method: str
    threshold: float
    by_column: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def to_dataframe(self) -> "pd.DataFrame":
        """Return outlier statistics as a display-ready DataFrame."""
        import pandas as pd

        columns = [
            "column",
            "n_outliers",
            "pct_outliers",
            "lower_bound",
            "upper_bound",
            "method",
            "threshold",
        ]
        rows = [
            {
                "column": col,
                "n_outliers": info.get("n_outliers", self.n_outliers),
                "pct_outliers": self.pct_outliers,
                "lower_bound": round(info.get("lower_bound", 0.0), 6),
                "upper_bound": round(info.get("upper_bound", 0.0), 6),
                "method": self.method,
                "threshold": self.threshold,
            }
            for col, info in self.by_column.items()
        ]
        if not rows:
            rows = [
                {
                    "column": "__all__",
                    "n_outliers": self.n_outliers,
                    "pct_outliers": self.pct_outliers,
                    "lower_bound": None,
                    "upper_bound": None,
                    "method": self.method,
                    "threshold": self.threshold,
                }
            ]
        return pd.DataFrame(rows, columns=columns)
