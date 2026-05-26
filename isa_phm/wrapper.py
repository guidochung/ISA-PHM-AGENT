"""
ISAWrapper — the primary entry point for the ISA-PHM Python wrapper.

Chains the four core components:
    ISAParser  →  ISAPreprocessor  →  MetadataExtractor  →  DataIntegrator

Exposes the fluent proxy API via ``QueryNavigator`` and convenience shortcuts.

Example usage::

    from isa_phm import ISAWrapper

    wrapper = ISAWrapper(
        "path/to/i_investigation.json",
        data_root="path/to/data/",
    )
    print(wrapper.investigation_overview())
    df = wrapper.study("Case 01").assay("a_st01_se01").load_dataframe()
    fig = wrapper.study("Case 01").assay("a_st01_se01").plot_lifecycle()
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd
    from .tools import ToolRegistry

from .extractor import MetadataExtractor
from .integrator import DataIntegrator
from .parser import ISAParser
from .plotter import ISAPlotter, PlotConfig
from .preprocessor import ISAPreprocessor
from .proxy import QueryNavigator, StudyProxy
from .semantic import SemanticNormalizer
from .schemas import (
    DatasetValidationReport,
    InvestigationModel,
    InvestigationOverview,
    RepairLog,
    SemanticManifest,
    StudySummary,
    ValidationIssue,
)

logger = logging.getLogger("isa_phm")


class ISAWrapper:
    """
    All-in-one facade that parses, preprocesses, and extracts an ISA-PHM dataset.

    Parameters
    ----------
    path : str | Path
        Path to the ISA-JSON file (``i_...json``).
    data_root : str | Path | None
        Directory that contains the actual CSV measurement files.
        If None, defaults to the directory containing the ISA-JSON file.
    auto_fix : bool
        Enable the five ISAPreprocessor auto-fix rules (default True).
    strict_validation : bool
        Use isatools for schema validation (default True).
        Set to False to skip isatools check (e.g., when isatools is not installed).
    cache_maxsize : int
        Maximum DataFrames held in the DataIntegrator's FIFO cache (default 100).
    enable_chunked_large_file_mode : bool
        Enable chunked CSV reading for lifecycle feature extraction on large files.
    large_file_threshold_mb : float
        Minimum file size (MB) that triggers chunked large-file mode.
    chunk_rows : int
        Number of rows per chunk when large-file mode is active.
    csv_bad_lines : "error" | "warn" | "skip"
        CSV malformed-line behavior. Default ``"error"`` (strict).
    plot_config : PlotConfig | None
        Optional style overrides passed to ISAPlotter.
    semantic_config_path : str | Path | None
        Optional JSON path with semantic alias overrides.
    """

    def __init__(
        self,
        path: str | Path,
        data_root: str | Path | None = None,
        auto_fix: bool = True,
        strict_validation: bool = True,
        cache_maxsize: int = 100,
        enable_chunked_large_file_mode: bool = True,
        large_file_threshold_mb: float = 64.0,
        chunk_rows: int = 250_000,
        csv_bad_lines: Literal["error", "warn", "skip"] = "error",
        plot_config: PlotConfig | None = None,
        semantic_config_path: str | Path | None = None,
    ) -> None:
        path = Path(path)
        if not path.exists():
            from .errors import ParseError

            raise ParseError(f"ISA-JSON file not found: '{path}'.")

        if data_root is None:
            data_root = path.parent
        data_root = Path(data_root)

        logger.info("ISAWrapper: loading '%s', data_root='%s'.", path, data_root)

        # --- Step 1: Parse ---
        parser = ISAParser(strict=strict_validation)
        raw = parser.load(path)

        # --- Step 2: Preprocess ---
        preprocessor = ISAPreprocessor(data_root=data_root, auto_fix=auto_fix)
        repaired, repair_log = preprocessor.preprocess(raw)

        # --- Step 3: Extract domain model ---
        extractor = MetadataExtractor()
        investigation = extractor.extract(repaired)

        # --- Step 4: Wire integrator + plotter + proxy layer ---
        integrator = DataIntegrator(
            data_root=data_root,
            cache_maxsize=cache_maxsize,
            enable_chunked_large_file_mode=enable_chunked_large_file_mode,
            large_file_threshold_mb=large_file_threshold_mb,
            chunk_rows=chunk_rows,
            csv_bad_lines=csv_bad_lines,
        )
        plotter = ISAPlotter(config=plot_config)
        semantic = SemanticNormalizer(override_config_path=semantic_config_path)
        navigator = QueryNavigator(investigation, integrator, plotter, semantic=semantic)

        # Store as instance attributes.
        self._investigation: InvestigationModel = investigation
        self._integrator: DataIntegrator = integrator
        self._plotter: ISAPlotter = plotter
        self._navigator: QueryNavigator = navigator
        self._semantic: SemanticNormalizer = semantic
        self._repair_log: RepairLog = repair_log
        self._source_path: Path = path
        self._tool_registry: ToolRegistry | None = None

        n_repairs = len(repair_log)
        if n_repairs:
            logger.info("ISAWrapper: %d preprocessor repair(s) applied.", n_repairs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def investigation(self) -> InvestigationModel:
        """The fully-extracted domain model (read-only)."""
        return self._investigation

    @property
    def repair_log(self) -> RepairLog:
        """Preprocessing repair actions.  Empty if no repairs were needed."""
        return self._repair_log

    @property
    def source_path(self) -> Path:
        """Absolute path to the ISA-JSON source file."""
        return self._source_path

    # ------------------------------------------------------------------
    # AI tool interface
    # ------------------------------------------------------------------

    def tool_registry(self) -> "ToolRegistry":
        """Return a JSON-only tool registry for AI agent integrations."""
        if self._tool_registry is None:
            from .tools import ToolRegistry

            self._tool_registry = ToolRegistry(self)
        return self._tool_registry

    def list_tools(self) -> list[dict[str, Any]]:
        """List available AI-facing tools and their input schemas."""
        return self.tool_registry().list_tools()

    def call_tool(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke an AI-facing tool by name using a JSON payload."""
        return self.tool_registry().invoke(name=name, args=args)

    # ------------------------------------------------------------------
    # High-level inspection
    # ------------------------------------------------------------------

    def investigation_overview(self) -> InvestigationOverview:
        """Return a structured summary of the investigation."""
        return self._navigator.investigation_overview()

    def list_studies(self) -> list[StudySummary]:
        """Return a summary row for each study."""
        return self._navigator.list_studies()

    def summary(self) -> "pd.DataFrame":
        """
        Return a compact one-row investigation summary as a DataFrame.

        This is a notebook-friendly alternative to raw model repr output.
        """
        import pandas as pd

        ov = self.investigation_overview()
        desc = " ".join((ov.description or "").split())
        if len(desc) > 180:
            desc = f"{desc[:177]}..."
        return pd.DataFrame(
            [
                {
                    "title": ov.title,
                    "identifier": ov.identifier,
                    "experiment_type": ov.experiment_type,
                    "n_studies": ov.n_studies,
                    "n_contacts": ov.n_contacts,
                    "n_publications": len(self._investigation.publications),
                    "description_short": desc,
                    "source_path": str(self._source_path),
                }
            ]
        )

    def contacts(self) -> "pd.DataFrame":
        """
        Return investigation contacts as a notebook-friendly DataFrame.

        This avoids manual traversal of ``wrapper.investigation.contacts``.
        """
        return self._investigation.contacts_df()

    def publications(self) -> "pd.DataFrame":
        """
        Return investigation publications as a notebook-friendly DataFrame.

        Notes
        -----
        ``author_tokens`` is kept as a semicolon-joined string because ISA
        commonly stores contact IDs there, not resolved names.
        """
        return self._investigation.publications_df()

    def extensive_summary(self) -> "dict[str, pd.DataFrame]":
        """
        Return a full investigation summary split into notebook-ready tables.

        Returns a dict with keys:
        - ``investigation``: one-row top-level metadata
        - ``studies``: one row per study
        - ``assays``: one row per assay (sensor channel)
        - ``factors``: one row per factor
        - ``contacts``: one row per contact
        - ``publications``: one row per publication
        """
        import pandas as pd

        ov = self.investigation_overview()
        inv_df = self.summary()

        study_rows: list[dict] = []
        assay_rows: list[dict] = []
        factor_rows: list[dict] = []

        for s in self._investigation.studies:
            study_rows.append(
                {
                    "study_id": s.study_id,
                    "title": s.title,
                    "n_assays": len(s.assays),
                    "n_runs": s.run_count,
                    "n_factors": len(s.factors),
                }
            )

            for a in s.assays:
                n_raw_files = sum(
                    1 for r in a.runs if r.raw_file is not None and bool(r.raw_file.path)
                )
                n_processed_files = sum(
                    1
                    for r in a.runs
                    if r.processed_file is not None and bool(r.processed_file.path)
                )
                assay_rows.append(
                    {
                        "study_id": s.study_id,
                        "study_title": s.title,
                        "assay_id": a.assay_id,
                        "sensor_alias": a.sensor.alias,
                        "sensor_id": a.sensor.sensor_id,
                        "measurement_type": a.sensor.measurement_type,
                        "technology_type": a.sensor.technology_type,
                        "technology_platform": a.sensor.technology_platform,
                        "n_runs": len(a.runs),
                        "n_raw_files": n_raw_files,
                        "n_processed_files": n_processed_files,
                    }
                )

            for f in s.factors:
                factor_rows.append(
                    {
                        "study_id": s.study_id,
                        "study_title": s.title,
                        "factor_name": f.factor_name,
                        "factor_type": f.factor_type,
                        "unit": f.unit or "",
                        "description": f.description or "",
                    }
                )

        studies_df = pd.DataFrame(
            study_rows,
            columns=["study_id", "title", "n_assays", "n_runs", "n_factors"],
        )
        assays_df = pd.DataFrame(
            assay_rows,
            columns=[
                "study_id",
                "study_title",
                "assay_id",
                "sensor_alias",
                "sensor_id",
                "measurement_type",
                "technology_type",
                "technology_platform",
                "n_runs",
                "n_raw_files",
                "n_processed_files",
            ],
        )
        factors_df = pd.DataFrame(
            factor_rows,
            columns=[
                "study_id",
                "study_title",
                "factor_name",
                "factor_type",
                "unit",
                "description",
            ],
        )
        contacts_df = self.contacts().loc[
            :, ["contact_id", "full_name", "email", "affiliation", "roles", "orcid"]
        ]
        publications_df = self.publications()

        # Keep stable ordering for interactive notebooks.
        if not studies_df.empty:
            studies_df = studies_df.sort_values(["title", "study_id"]).reset_index(
                drop=True
            )
        if not assays_df.empty:
            assays_df = assays_df.sort_values(
                ["study_title", "assay_id"]
            ).reset_index(drop=True)
        if not factors_df.empty:
            factors_df = factors_df.sort_values(
                ["study_title", "factor_name"]
            ).reset_index(drop=True)

        # Keep top-level title in investigation table for easy context checks.
        inv_df.loc[:, "n_studies"] = ov.n_studies

        return {
            "investigation": inv_df,
            "studies": studies_df,
            "assays": assays_df,
            "factors": factors_df,
            "contacts": contacts_df,
            "publications": publications_df,
        }

    def semantic_manifest(
        self,
        *,
        strict: bool = False,
        max_unknown_ratio: float = 0.05,
        max_ambiguous_ratio: float = 0.0,
        require_override_config: bool = False,
    ) -> SemanticManifest:
        """
        Return normalized semantic labels for factors and protocol parameters.

        Parameters
        ----------
        strict : bool
            When True, raise :class:`ValidationError` if strict thresholds fail.
        max_unknown_ratio : float
            Maximum allowed ratio of unknown semantic fields in strict mode.
        max_ambiguous_ratio : float
            Maximum allowed ratio of ambiguous semantic fields in strict mode.
        require_override_config : bool
            Require a user semantic override config for strict validation.
        """
        return self._semantic.build_manifest_with_controls(
            self._investigation,
            strict=strict,
            max_unknown_ratio=max_unknown_ratio,
            max_ambiguous_ratio=max_ambiguous_ratio,
            require_override_config=require_override_config,
        )

    def ai_context(
        self,
        include_semantics: bool = True,
        include_validation: bool = False,
    ) -> dict[str, Any]:
        """
        Return a deterministic JSON-safe metadata export for AI pipelines.

        The payload is metadata-only and never includes DataFrames or model objects.
        """
        investigation = self._investigation
        generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        source_path = str(self._source_path.resolve())

        studies_rows: list[dict[str, Any]] = []
        assays_rows: list[dict[str, Any]] = []
        factors_rows: list[dict[str, Any]] = []
        for study in sorted(investigation.studies, key=lambda s: (s.title.lower(), s.study_id)):
            studies_rows.append(
                {
                    "study_id": study.study_id,
                    "title": study.title,
                    "description": study.description,
                    "n_assays": len(study.assays),
                    "n_runs": study.run_count,
                    "n_factors": len(study.factors),
                }
            )

            for assay in sorted(study.assays, key=lambda a: a.assay_id):
                n_raw_files = sum(
                    1 for run in assay.runs if run.raw_file is not None and bool(run.raw_file.path)
                )
                n_processed_files = sum(
                    1
                    for run in assay.runs
                    if run.processed_file is not None and bool(run.processed_file.path)
                )
                assays_rows.append(
                    {
                        "study_id": study.study_id,
                        "study_title": study.title,
                        "assay_id": assay.assay_id,
                        "sensor_alias": assay.sensor.alias,
                        "sensor_id": assay.sensor.sensor_id,
                        "measurement_type": assay.sensor.measurement_type,
                        "technology_type": assay.sensor.technology_type,
                        "technology_platform": assay.sensor.technology_platform,
                        "n_runs": len(assay.runs),
                        "n_raw_files": n_raw_files,
                        "n_processed_files": n_processed_files,
                    }
                )

            for factor in sorted(study.factors, key=lambda f: f.factor_name.lower()):
                factors_rows.append(
                    {
                        "study_id": study.study_id,
                        "study_title": study.title,
                        "factor_id": factor.factor_id,
                        "factor_name": factor.factor_name,
                        "factor_type": factor.factor_type,
                        "unit": factor.unit,
                        "description": factor.description,
                    }
                )

        contact_rows = [
            {
                "contact_id": contact.contact_id,
                "first_name": contact.first_name,
                "last_name": contact.last_name,
                "full_name": contact.full_name,
                "email": contact.email,
                "affiliation": contact.affiliation,
                "roles": list(contact.roles),
                "orcid": contact.orcid,
            }
            for contact in sorted(
                investigation.contacts,
                key=lambda c: (c.full_name.lower(), c.email.lower(), c.contact_id or ""),
            )
        ]
        publication_rows = [
            {
                "title": publication.title,
                "doi": publication.doi,
                "pubmed_id": publication.pubmed_id,
                "status": publication.status,
                "author_tokens": list(publication.author_tokens),
                "corresponding_author": publication.corresponding_author,
                "resolved_author_names": list(publication.resolved_author_names),
                "resolved_author_emails": list(publication.resolved_author_emails),
                "unresolved_author_tokens": list(publication.unresolved_author_tokens),
            }
            for publication in sorted(
                investigation.publications,
                key=lambda p: (p.title.lower(), p.doi or "", p.pubmed_id or ""),
            )
        ]

        payload: dict[str, Any] = {
            "schema_version": "isa_phm.ai_context.v1",
            "generated_at_utc": generated_at,
            "source_path": source_path,
            "investigation": {
                "title": investigation.title,
                "description": investigation.description,
                "identifier": investigation.identifier,
                "experiment_type": investigation.experiment_type,
                "n_studies": len(investigation.studies),
                "n_contacts": len(investigation.contacts),
                "n_publications": len(investigation.publications),
            },
            "contacts": contact_rows,
            "publications": publication_rows,
            "studies": studies_rows,
            "assays": assays_rows,
            "factors": factors_rows,
        }

        if include_semantics:
            manifest = self.semantic_manifest(strict=False)
            payload["semantic_manifest"] = {
                "investigation_id": manifest.investigation_id,
                "study_factors": {
                    k: [f.model_dump() for f in sorted(v, key=lambda x: x.source_name.lower())]
                    for k, v in sorted(manifest.study_factors.items(), key=lambda x: x[0])
                },
                "assay_measurement_params": {
                    k: [f.model_dump() for f in sorted(v, key=lambda x: x.source_name.lower())]
                    for k, v in sorted(
                        manifest.assay_measurement_params.items(),
                        key=lambda x: x[0],
                    )
                },
                "assay_processing_params": {
                    k: [f.model_dump() for f in sorted(v, key=lambda x: x.source_name.lower())]
                    for k, v in sorted(
                        manifest.assay_processing_params.items(),
                        key=lambda x: x[0],
                    )
                },
                "diagnostics": manifest.diagnostics.model_dump(),
            }

        if include_validation:
            payload["validation_report"] = self.validate_dataset().model_dump()

        return payload

    def validate_dataset(
        self,
        *,
        check_files: bool = True,
        semantic_strict: bool = False,
        max_unknown_ratio: float = 0.05,
        max_ambiguous_ratio: float = 0.0,
        require_override_config: bool = False,
    ) -> DatasetValidationReport:
        """
        Return a structured dataset validation report.

        Checks include structure completeness, file references, metadata quality,
        and semantic coverage.
        """
        issue_buckets: dict[tuple[str, str, str], dict[str, Any]] = {}

        def add_issue(
            *,
            code: str,
            level: str,
            scope: str,
            message: str,
            context: dict[str, Any] | None = None,
        ) -> None:
            payload = context or {}
            payload_key = json.dumps(payload, sort_keys=True, default=str)
            bucket_key = (code, level, message)
            bucket = issue_buckets.get(bucket_key)
            if bucket is None:
                issue_buckets[bucket_key] = {
                    "code": code,
                    "level": level,
                    "message": message,
                    "context": dict(payload),
                    "context_keys": {payload_key},
                    "contexts_sample": [dict(payload)],
                    "scopes": [scope],
                    "scope_set": {scope},
                    "n_occurrences": 1,
                }
                return

            bucket["n_occurrences"] += 1
            if scope not in bucket["scope_set"]:
                bucket["scope_set"].add(scope)
                bucket["scopes"].append(scope)
            if payload_key not in bucket["context_keys"]:
                bucket["context_keys"].add(payload_key)
                if len(bucket["contexts_sample"]) < 10:
                    bucket["contexts_sample"].append(dict(payload))

        inv = self._investigation
        if not inv.title.strip():
            add_issue(
                code="INV_MISSING_TITLE",
                level="warning",
                scope="investigation",
                message="Investigation title is empty.",
            )
        if not inv.identifier.strip():
            add_issue(
                code="INV_MISSING_IDENTIFIER",
                level="warning",
                scope="investigation",
                message="Investigation identifier is empty.",
            )
        if not inv.studies:
            add_issue(
                code="INV_NO_STUDIES",
                level="error",
                scope="investigation",
                message="No studies found in investigation.",
            )

        for contact in inv.contacts:
            scope = f"contact:{contact.contact_id or contact.full_name}"
            if not contact.email:
                add_issue(
                    code="CONTACT_MISSING_EMAIL",
                    level="info",
                    scope=scope,
                    message="Contact has no email address.",
                )
            if not contact.full_name:
                add_issue(
                    code="CONTACT_MISSING_NAME",
                    level="warning",
                    scope=scope,
                    message="Contact has no name fields.",
                )

        for publication in inv.publications:
            scope = f"publication:{publication.title or 'untitled'}"
            if not publication.title:
                add_issue(
                    code="PUBLICATION_MISSING_TITLE",
                    level="warning",
                    scope=scope,
                    message="Publication title is empty.",
                )
            if publication.author_tokens and not publication.resolved_author_names:
                add_issue(
                    code="PUBLICATION_AUTHORS_UNRESOLVED",
                    level="warning",
                    scope=scope,
                    message="Publication author tokens could not be resolved to contacts.",
                    context={"author_tokens": publication.author_tokens},
                )
            elif publication.unresolved_author_tokens:
                add_issue(
                    code="PUBLICATION_AUTHORS_PARTIAL",
                    level="info",
                    scope=scope,
                    message="Publication contains unresolved author tokens.",
                    context={"unresolved_author_tokens": publication.unresolved_author_tokens},
                )

        for study in inv.studies:
            study_scope = f"study:{study.study_id or study.title}"
            if not study.assays:
                add_issue(
                    code="STUDY_NO_ASSAYS",
                    level="error",
                    scope=study_scope,
                    message="Study has no assays.",
                )
            if not study.factors:
                add_issue(
                    code="STUDY_NO_FACTORS",
                    level="warning",
                    scope=study_scope,
                    message="Study has no factors.",
                )
            if not study.title:
                add_issue(
                    code="STUDY_MISSING_TITLE",
                    level="warning",
                    scope=study_scope,
                    message="Study title is empty.",
                )

            for assay in study.assays:
                assay_scope = f"{study_scope}/assay:{assay.assay_id}"
                if not assay.runs:
                    add_issue(
                        code="ASSAY_NO_RUNS",
                        level="error",
                        scope=assay_scope,
                        message="Assay has no runs.",
                    )
                    continue

                for run in assay.runs:
                    run_scope = f"{assay_scope}/run:{run.run_id}"
                    raw_path = run.raw_file.path if run.raw_file is not None else ""
                    processed_path = (
                        run.processed_file.path if run.processed_file is not None else ""
                    )

                    if not raw_path and not processed_path:
                        add_issue(
                            code="RUN_NO_DATA_FILE",
                            level="error",
                            scope=run_scope,
                            message="Run has neither raw nor processed data file path.",
                        )

                    if run.raw_file is not None and not run.raw_file.path:
                        add_issue(
                            code="RAW_FILE_PATH_EMPTY",
                            level="info",
                            scope=run_scope,
                            message="Raw data file record exists but path is empty.",
                        )
                    if run.processed_file is not None and not run.processed_file.path:
                        add_issue(
                            code="PROCESSED_FILE_PATH_EMPTY",
                            level="info",
                            scope=run_scope,
                            message="Processed data file record exists but path is empty.",
                        )

                    if check_files:
                        for kind, data_file in (
                            ("raw", run.raw_file),
                            ("processed", run.processed_file),
                        ):
                            if data_file is None or not data_file.path:
                                continue
                            if not data_file.exists:
                                add_issue(
                                    code="FILE_NOT_FOUND",
                                    level="warning",
                                    scope=run_scope,
                                    message=f"{kind} data file path does not exist on disk.",
                                    context={
                                        "file_type": kind,
                                        "path": data_file.path,
                                        "exists_at_extract": data_file.exists_at_extract,
                                    },
                                )

        manifest = self.semantic_manifest(
            strict=False,
            max_unknown_ratio=max_unknown_ratio,
            max_ambiguous_ratio=max_ambiguous_ratio,
            require_override_config=require_override_config,
        )
        diagnostics = manifest.diagnostics

        if diagnostics.unknown_fields > 0:
            add_issue(
                code="SEM_UNKNOWN_FIELDS",
                level="warning",
                scope="semantic",
                message="Unknown semantic fields detected.",
                context={
                    "unknown_fields": diagnostics.unknown_fields,
                    "unknown_ratio": diagnostics.unknown_ratio,
                },
            )
        if diagnostics.ambiguous_fields > 0:
            add_issue(
                code="SEM_AMBIGUOUS_FIELDS",
                level="warning",
                scope="semantic",
                message="Ambiguous semantic fields detected.",
                context={
                    "ambiguous_fields": diagnostics.ambiguous_fields,
                    "ambiguous_ratio": diagnostics.ambiguous_ratio,
                },
            )

        violations = self._semantic.evaluate_strict_violations(
            manifest,
            max_unknown_ratio=max_unknown_ratio,
            max_ambiguous_ratio=max_ambiguous_ratio,
            require_override_config=require_override_config,
        )
        if semantic_strict and violations:
            for code in violations:
                add_issue(
                    code=code,
                    level="error",
                    scope="semantic",
                    message="Semantic strict validation violation.",
                    context={
                        "max_unknown_ratio": max_unknown_ratio,
                        "max_ambiguous_ratio": max_ambiguous_ratio,
                        "require_override_config": require_override_config,
                    },
                )
        elif violations:
            for code in violations:
                add_issue(
                    code=code,
                    level="warning",
                    scope="semantic",
                    message="Semantic strict threshold would fail.",
                    context={
                        "max_unknown_ratio": max_unknown_ratio,
                        "max_ambiguous_ratio": max_ambiguous_ratio,
                        "require_override_config": require_override_config,
                    },
                )

        issues: list[ValidationIssue] = []
        for bucket in issue_buckets.values():
            scopes: list[str] = bucket["scopes"]
            context_payload: dict[str, Any] = dict(bucket["context"])
            n_occurrences = int(bucket["n_occurrences"])

            if n_occurrences > 1:
                context_payload["n_occurrences"] = n_occurrences
                context_payload["scopes_sample"] = scopes[:10]
                context_payload["n_unique_scopes"] = len(scopes)
                if len(bucket["contexts_sample"]) > 1:
                    context_payload["contexts_sample"] = bucket["contexts_sample"]
                scope_value = "multiple"
            else:
                scope_value = scopes[0]

            issues.append(
                ValidationIssue(
                    code=bucket["code"],
                    level=bucket["level"],
                    scope=scope_value,
                    message=bucket["message"],
                    context=context_payload,
                )
            )

        n_errors = sum(1 for issue in issues if issue.level == "error")
        n_warnings = sum(1 for issue in issues if issue.level == "warning")
        n_info = sum(1 for issue in issues if issue.level == "info")
        return DatasetValidationReport(
            ok=(n_errors == 0),
            n_errors=n_errors,
            n_warnings=n_warnings,
            n_info=n_info,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # Fluent proxy navigation
    # ------------------------------------------------------------------

    def study(self, study_id: str | int) -> StudyProxy:
        """
        Navigate into a study by 1-based index, UUID, or title.

        Parameters
        ----------
        study_id : str | int
            1-based study index, study UUID, or title (case-insensitive).

        Returns
        -------
        StudyProxy

        Raises
        ------
        StudyNotFoundError

        Example
        -------
        >>> wrapper.study("BPFO Fault Severity 1 100%").assay("a_st01_se01").load_dataframe()
        """
        return self._navigator.study(study_id)

    def compare_studies(
        self,
        study_ids: "list[str | int]",
        assay_id: "str | int | None" = None,
        assay_group: "AssayGroup | None" = None,
        file_type: "Literal['raw', 'processed', 'auto']" = "raw",
        n_workers: "int | None" = None,
    ) -> "dict[str, pd.DataFrame]":
        """
        Load lifecycle features for the same sensor across multiple studies.

        Delegates to :py:meth:`QueryNavigator.compare_studies`.
        Only studies listed in ``study_ids`` are included.
        """
        return self._navigator.compare_studies(
            study_ids,
            assay_id=assay_id,
            assay_group=assay_group,
            file_type=file_type,
            n_workers=n_workers,
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Evict all DataFrames from the integrator's FIFO cache."""
        self._integrator.clear_cache()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_studies = len(self._investigation.studies)
        return (
            f"ISAWrapper("
            f"title={self._investigation.title!r}, "
            f"n_studies={n_studies}, "
            f"experiment_type={self._investigation.experiment_type!r})"
        )
