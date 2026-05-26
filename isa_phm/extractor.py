"""
MetadataExtractor — converts a preprocessed ISA-JSON dict into typed
ISA-PHM domain models (InvestigationModel → StudyModel → AssayModel → RunRecord).

Extraction flow
---------------
1. Build a ReferenceResolver index from the full dict.
2. Extract investigation-level fields (title, experiment_type, contacts).
3. For each study:
   a. Extract factors from study.factors[].
   b. Extract study-level sample factorValues (used on all runs).
   c. For each assay:
      i.  Extract sensor identity from technologyType / technologyPlatform.
      ii. Build a data file index (dataFile @id → DataFile model).
      iii. Order the processSequence chain via nextProcess/previousProcess links.
      iv. Walk the ordered chain grouping (measurement, processing) pairs → RunRecords.
      v.  Identify measurement and processing protocols from first pair.

Run identification
------------------
The processSequence is a flat list.  We reconstruct the ordered chain by
following nextProcess links from the root process (the one with no previousProcess).

A measurement process has inputs starting with "#sample/" and outputs "#data_file/".
A processing process has inputs starting with "#data_file/" and outputs "#data_file/".

Each consecutive (measurement, processing) pair in the chain = one run.
run_id format: "run_01" (2-digit pad) or "run_0001" (4-digit pad for >99 runs).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from .errors import ExtractionError
from .schemas import (
    AssayModel,
    ContactModel,
    DataFile,
    FactorModel,
    InvestigationModel,
    ParameterValue,
    PublicationModel,
    ProtocolModel,
    ProtocolParameter,
    RunRecord,
    SensorInfo,
    StudyModel,
)
from .utils import ReferenceResolver

logger = logging.getLogger("isa_phm")


class MetadataExtractor:
    """
    Extract typed ISA-PHM domain models from a preprocessed ISA-JSON dict.

    Usage
    -----
    extractor = MetadataExtractor()
    investigation = extractor.extract(repaired_dict)
    """

    def extract(self, repaired: dict) -> InvestigationModel:
        """
        Convert a preprocessed ISA-JSON dict to an InvestigationModel.

        Parameters
        ----------
        repaired : dict
            Output of ISAPreprocessor.preprocess().

        Returns
        -------
        InvestigationModel

        Raises
        ------
        ExtractionError
            On unresolvable @id references or structurally invalid PHM data.
        """
        resolver = ReferenceResolver()
        resolver.build(repaired)
        logger.debug("ReferenceResolver ready with %d objects.", len(resolver))

        experiment_type = self._extract_comment(repaired, "experiment_type", "unknown")
        contacts = [self._extract_contact(p) for p in repaired.get("people", [])]
        publications = [
            self._extract_publication(p) for p in repaired.get("publications", [])
        ]
        self._link_publications_to_contacts(publications, contacts)
        studies = [
            self._extract_study(s, resolver)
            for s in repaired.get("studies", [])
        ]

        return InvestigationModel(
            title=repaired.get("title", ""),
            description=repaired.get("description", ""),
            identifier=repaired.get("identifier", ""),
            experiment_type=experiment_type,
            studies=studies,
            contacts=contacts,
            publications=publications,
        )

    # ------------------------------------------------------------------
    # Study extraction
    # ------------------------------------------------------------------

    def _extract_study(self, study: dict, resolver: ReferenceResolver) -> StudyModel:
        study_id = study.get("identifier", "")
        title = study.get("title", "")
        description = study.get("description", "")

        # Factors
        factors = [self._extract_factor(f) for f in study.get("factors", [])]

        # Per-sample factor value index (sample_@id → {factor_name: value}).
        # Key "" holds the study-level fallback (first non-empty sample).
        sample_factor_index = self._build_sample_factor_index(study, factors, resolver)
        study_factor_values = sample_factor_index.get("", {})

        # Protocols lookup (study.protocols[] indexed by @id).
        protocol_index = self._build_protocol_index(study, resolver)

        # Assays
        assays = [
            self._extract_assay(a, study_factor_values, sample_factor_index, protocol_index, resolver)
            for a in study.get("assays", [])
        ]

        return StudyModel(
            study_id=study_id,
            title=title,
            description=description,
            factors=factors,
            assays=assays,
        )

    def _extract_factor(self, f: dict) -> FactorModel:
        factor_type_obj = f.get("factorType") or {}
        factor_type = factor_type_obj.get("annotationValue", "")

        # Unit and description stored in factor comments.
        unit = None
        description = None
        for comment in f.get("comments", []):
            name = comment.get("name", "")
            value = comment.get("value")
            if name == "unit" and value:
                unit = str(value)
            if name == "description" and value:
                description = str(value)

        return FactorModel(
            factor_id=f.get("@id", ""),
            factor_name=f.get("factorName", ""),
            factor_type=factor_type,
            unit=unit or None,
            description=description,
        )

    def _extract_sample_factor_values(
        self,
        sample: dict,
        factor_by_id: dict[str, FactorModel],
        resolver: ReferenceResolver,
    ) -> dict[str, Any]:
        """Return {factor_name: value_str} for a single sample object."""
        result: dict[str, Any] = {}
        for fv in sample.get("factorValues", []):
            cat_ref = fv.get("category", {})
            factor_obj = resolver.try_resolve(cat_ref)
            if factor_obj is None:
                continue
            fid = factor_obj.get("@id", "")
            factor_model = factor_by_id.get(fid)
            factor_name = factor_model.factor_name if factor_model else factor_obj.get("factorName", fid)

            raw_value = fv.get("value")
            unit_str = resolver.get_unit_str(fv.get("unit"))
            if unit_str:
                result[factor_name] = f"{raw_value} {unit_str}"
            else:
                result[factor_name] = raw_value
        return result

    def _build_sample_factor_index(
        self,
        study: dict,
        factors: list[FactorModel],
        resolver: ReferenceResolver,
    ) -> dict[str, dict[str, Any]]:
        """
        Return {sample_@id: {factor_name: value}} for every sample in the study.

        Also returns the first non-empty entry as the study-level fallback
        (accessible via the empty-string key "").
        """
        factor_by_id = {f.factor_id: f for f in factors}
        index: dict[str, dict[str, Any]] = {}
        fallback: dict[str, Any] = {}

        materials = study.get("materials", {})
        samples = materials.get("samples", [])

        seen_ids: set[str] = set()
        for sample_ref in samples:
            sample = resolver.try_resolve(sample_ref) if len(sample_ref) == 1 else sample_ref
            if sample is None:
                continue
            sid = sample.get("@id", "")
            if sid in seen_ids:
                continue
            seen_ids.add(sid)

            fv_map = self._extract_sample_factor_values(sample, factor_by_id, resolver)
            index[sid] = fv_map
            if not fallback and fv_map:
                fallback = fv_map

        index[""] = fallback  # empty-string key = study-level fallback
        return index

    def _build_protocol_index(
        self, study: dict, resolver: ReferenceResolver
    ) -> dict[str, ProtocolModel]:
        """Build a @id → ProtocolModel index from study.protocols[]."""
        index: dict[str, ProtocolModel] = {}
        for proto in study.get("protocols", []):
            pid = proto.get("@id", "")
            if not pid:
                continue
            index[pid] = self._extract_protocol(proto, resolver)
        return index

    def _extract_protocol(
        self, proto: dict, resolver: ReferenceResolver
    ) -> ProtocolModel:
        proto_type_obj = proto.get("protocolType") or {}
        proto_type = proto_type_obj.get("annotationValue", "")

        # Sensor ID may be stored in protocol comments.
        sensor_id: str | None = None
        for comment in proto.get("comments", []):
            if comment.get("name") == "Sensor id":
                sensor_id = comment.get("value") or None

        # Extract parameter definitions.
        parameters: list[ProtocolParameter] = []
        for param in proto.get("parameters", []):
            param_id = param.get("@id", "")
            pname_obj = param.get("parameterName") or {}
            pname = pname_obj.get("annotationValue", param.get("name", ""))
            parameters.append(ProtocolParameter(id=param_id, parameter_name=pname))

        return ProtocolModel(
            id=proto.get("@id", ""),
            name=proto.get("name", ""),
            protocol_type=proto_type,
            parameters=parameters,
            sensor_id=sensor_id,
        )

    # ------------------------------------------------------------------
    # Assay extraction
    # ------------------------------------------------------------------

    def _extract_assay(
        self,
        assay: dict,
        study_factor_values: dict[str, Any],
        sample_factor_index: dict[str, dict[str, Any]],
        protocol_index: dict[str, ProtocolModel],
        resolver: ReferenceResolver,
    ) -> AssayModel:
        assay_id = assay.get("filename", "")

        tech_type = resolver.get_annotation_value(
            assay.get("technologyType"), default=""
        )
        measurement_type = resolver.get_annotation_value(
            assay.get("measurementType"), default=""
        )
        tech_platform = assay.get("technologyPlatform", "")

        # Build data file index: @id → DataFile.
        data_file_index = self._build_data_file_index(assay)

        # Order process chain.
        raw_processes = assay.get("processSequence", [])
        ordered_procs = _order_process_chain(raw_processes)
        logger.debug(
            "Assay '%s': %d processes → %d in ordered chain.",
            assay_id, len(raw_processes), len(ordered_procs),
        )

        # Identify measurement and processing protocols from chain.
        meas_protocol: ProtocolModel | None = None
        proc_protocol: ProtocolModel | None = None

        for proc in ordered_procs:
            proto_ref = proc.get("executesProtocol", {})
            proto_id = proto_ref.get("@id", "") if isinstance(proto_ref, dict) else ""
            inputs = proc.get("inputs", [])
            is_meas = bool(inputs) and inputs[0].get("@id", "").startswith("#sample/")

            if is_meas and meas_protocol is None:
                meas_protocol = protocol_index.get(proto_id)
                if meas_protocol is None:
                    # Fall back: resolve the protocol object directly.
                    proto_obj = resolver.try_resolve(proto_ref)
                    if proto_obj:
                        meas_protocol = self._extract_protocol(proto_obj, resolver)
            elif not is_meas and proc_protocol is None:
                proc_protocol = protocol_index.get(proto_id)
                if proc_protocol is None:
                    proto_obj = resolver.try_resolve(proto_ref)
                    if proto_obj:
                        proc_protocol = self._extract_protocol(proto_obj, resolver)

            if meas_protocol and proc_protocol:
                break

        # Sensor identity.
        sensor_id = meas_protocol.sensor_id if meas_protocol else None

        # Prefer the "sensor alias" comment on the assay over the filename-based ID.
        sensor_alias = assay_id
        for comment in assay.get("comments", []):
            if comment.get("name", "").lower().strip() == "sensor alias":
                value = comment.get("value", "").strip()
                if value:
                    sensor_alias = value
                    break

        sensor = SensorInfo(
            sensor_id=sensor_id,
            alias=sensor_alias,
            technology_type=tech_type,
            technology_platform=tech_platform,
            measurement_type=measurement_type,
        )

        # Build parameter definition lookup (id → name) for this assay.
        param_def_index = _build_param_def_index(meas_protocol, proc_protocol)

        # Extract runs from ordered process chain.
        runs = _extract_runs(
            ordered_procs, data_file_index, param_def_index,
            resolver, study_factor_values, sample_factor_index,
        )

        return AssayModel(
            assay_id=assay_id,
            sensor=sensor,
            measurement_protocol=meas_protocol,
            processing_protocol=proc_protocol,
            runs=runs,
        )

    def _build_data_file_index(self, assay: dict) -> dict[str, DataFile]:
        """Build @id → DataFile index from assay.dataFiles[]."""
        index: dict[str, DataFile] = {}
        for df in assay.get("dataFiles", []):
            df_id = df.get("@id", "")
            name = df.get("name", "")
            file_type = df.get("type", "")
            exists = Path(name).exists() if name else False
            index[df_id] = DataFile(
                id=df_id,
                path=name,
                file_type=file_type,
                exists_at_extract=exists,
            )
        return index

    # ------------------------------------------------------------------
    # Contact extraction
    # ------------------------------------------------------------------

    def _extract_contact(self, person: dict) -> ContactModel:
        roles = [
            r.get("annotationValue", "")
            for r in person.get("roles", [])
            if r.get("annotationValue")
        ]
        orcid: str | None = None
        for comment in person.get("comments", []):
            if comment.get("name") == "orcid":
                orcid = comment.get("value") or None

        return ContactModel(
            contact_id=self._normalize_contact_token(person.get("@id")),
            first_name=person.get("firstName", ""),
            last_name=person.get("lastName", ""),
            email=person.get("email", ""),
            affiliation=person.get("affiliation", ""),
            roles=roles,
            orcid=orcid,
        )

    def _extract_publication(self, publication: dict) -> PublicationModel:
        """
        Extract an investigation-level publication record.

        ISA authorList is typically a semicolon-separated string of author tokens
        (often contact IDs). Tokens are preserved as-is after trimming.
        """
        author_raw = publication.get("authorList", "")
        if isinstance(author_raw, str):
            author_tokens = [
                token.strip().lstrip("#")
                for token in author_raw.split(";")
                if token.strip()
            ]
        else:
            author_tokens = []

        corresponding_author: str | None = None
        for comment in publication.get("comments", []):
            if (comment.get("name") or "").strip().lower() == "corresponding author id":
                raw_value = comment.get("value")
                if raw_value:
                    corresponding_author = str(raw_value).strip().lstrip("#")
                break

        status_obj = publication.get("status") or {}
        status = (
            status_obj.get("annotationValue", "")
            if isinstance(status_obj, dict)
            else str(status_obj)
        )

        doi = publication.get("doi")
        pubmed_id = publication.get("pubMedID")
        return PublicationModel(
            title=publication.get("title", ""),
            doi=str(doi).strip() if doi else None,
            pubmed_id=str(pubmed_id).strip() if pubmed_id else None,
            status=str(status).strip() if status else None,
            author_tokens=author_tokens,
            corresponding_author=corresponding_author,
        )

    @staticmethod
    def _normalize_contact_token(raw: Any) -> str | None:
        if raw is None:
            return None
        token = str(raw).strip().lstrip("#")
        return token or None

    def _link_publications_to_contacts(
        self,
        publications: list[PublicationModel],
        contacts: list[ContactModel],
    ) -> None:
        """
        Resolve publication author tokens to investigation contacts by contact_id.

        Mapping is intentionally conservative: unresolved tokens are preserved and
        never guessed.
        """
        contacts_by_id = {
            c.contact_id: c
            for c in contacts
            if c.contact_id
        }
        for pub in publications:
            resolved_names: list[str] = []
            resolved_emails: list[str] = []
            unresolved: list[str] = []
            seen_names: set[str] = set()
            seen_emails: set[str] = set()

            for token in pub.author_tokens:
                normalized = self._normalize_contact_token(token)
                contact = contacts_by_id.get(normalized) if normalized else None
                if contact is None:
                    unresolved.append(token)
                    continue

                full_name = contact.full_name
                if full_name and full_name not in seen_names:
                    resolved_names.append(full_name)
                    seen_names.add(full_name)
                if contact.email and contact.email not in seen_emails:
                    resolved_emails.append(contact.email)
                    seen_emails.add(contact.email)

            pub.resolved_author_names = resolved_names
            pub.resolved_author_emails = resolved_emails
            pub.unresolved_author_tokens = unresolved

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_comment(obj: dict, name: str, default: str = "") -> str:
        for c in obj.get("comments", []):
            if c.get("name") == name:
                return c.get("value", default) or default
        return default


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no class state needed)
# ---------------------------------------------------------------------------


def _order_process_chain(processes: list[dict]) -> list[dict]:
    """
    Return the process list in chain order by following nextProcess links.

    Starts at the root process (no previousProcess field) and walks
    forward via nextProcess until the chain ends.

    Raises ExtractionError when cycles or unreachable processes are found.
    """
    if not processes:
        return []

    if any("@id" not in p for p in processes):
        raise ExtractionError("Process chain contains process entries without '@id'.")

    by_id = {p["@id"]: p for p in processes if "@id" in p}

    # Find root: process(es) with no previousProcess field.
    roots = [p for p in processes if "previousProcess" not in p or not p.get("previousProcess")]

    if not roots:
        raise ExtractionError(
            "Process chain has no clear root (all processes have previousProcess)."
        )

    if len(roots) > 1:
        logger.debug(
            "%d root processes found (multi-run dataset). Walking all chains.",
            len(roots),
        )

    # Walk every root chain and concatenate (handles multi-run datasets where
    # each run has its own independent measurement → processing chain).
    ordered: list[dict] = []
    seen: set[str] = set()

    for root in roots:
        current: dict | None = root
        while current is not None:
            cid = current.get("@id", "")
            if cid in seen:
                raise ExtractionError(
                    f"Cycle detected in process chain at process '{cid}'."
                )
            seen.add(cid)
            ordered.append(current)

            next_ref = current.get("nextProcess")
            if next_ref:
                next_id = next_ref.get("@id", "") if isinstance(next_ref, dict) else str(next_ref)
                current = by_id.get(next_id)
            else:
                current = None

    # Hard-fail if the chain walk missed any processes (e.g. unreachable cycles).
    if len(ordered) < len(processes):
        missing = sorted(set(by_id.keys()) - {p.get("@id", "") for p in ordered})
        raise ExtractionError(
            "Process chain walk did not cover all processes. "
            f"Captured {len(ordered)} of {len(processes)}. "
            f"Unreachable process IDs: {missing}"
        )

    return ordered


def _build_param_def_index(
    meas: ProtocolModel | None,
    proc: ProtocolModel | None,
) -> dict[str, str]:
    """Return {protocol_parameter @id → parameter_name} mapping."""
    index: dict[str, str] = {}
    for protocol in (meas, proc):
        if protocol:
            for param in protocol.parameters:
                index[param.id] = param.parameter_name
    return index


def _resolve_param_value(
    pv: dict,
    param_def_index: dict[str, str],
    resolver: ReferenceResolver,
) -> ParameterValue:
    """
    Convert a raw parameterValue dict into a ParameterValue model.

    The category field is a bare @id reference to a protocol_parameter.
    We look up the human-readable name from param_def_index first;
    fall back to resolving the full object from the tree.
    """
    cat_ref = pv.get("category", {})
    cat_id = cat_ref.get("@id", "") if isinstance(cat_ref, dict) else str(cat_ref)

    param_name = param_def_index.get(cat_id, "")
    if not param_name:
        # Try resolving from the tree.
        cat_obj = resolver.try_resolve(cat_ref)
        if cat_obj:
            pname_obj = cat_obj.get("parameterName") or {}
            param_name = pname_obj.get("annotationValue", cat_id)
        else:
            param_name = cat_id  # Fall back to @id string.

    value = pv.get("value")
    unit = resolver.get_unit_str(pv.get("unit"))

    return ParameterValue(parameter_name=param_name, value=value, unit=unit)


def _make_run_id(run_number: int, max_runs: int) -> str:
    """Return a zero-padded run ID string."""
    n_digits = max(2, len(str(max_runs)))
    return f"run_{run_number:0{n_digits}d}"


def _extract_runs(
    ordered_procs: list[dict],
    data_file_index: dict[str, DataFile],
    param_def_index: dict[str, str],
    resolver: ReferenceResolver,
    study_factor_values: dict[str, Any],
    sample_factor_index: dict[str, dict[str, Any]] | None = None,
) -> list[RunRecord]:
    """
    Walk the ordered process chain and extract RunRecords.

    Each (measurement process, processing process) consecutive pair = one run.
    A measurement process is identified by having a "#sample/" input.
    A processing process is identified by having a "#data_file/" input.

    Per-run factor values are read from the sample referenced by each
    measurement process's first input, falling back to study_factor_values.
    """
    # Estimate maximum run count for zero-padding.
    meas_count = sum(
        1 for p in ordered_procs
        if p.get("inputs") and p["inputs"][0].get("@id", "").startswith("#sample/")
    )
    max_runs = max(meas_count, 1)

    runs: list[RunRecord] = []
    run_number = 1
    i = 0

    while i < len(ordered_procs):
        proc = ordered_procs[i]
        inputs = proc.get("inputs", [])

        is_measurement = (
            bool(inputs) and inputs[0].get("@id", "").startswith("#sample/")
        )

        if not is_measurement:
            # Unexpected processing-only process — skip.
            logger.debug("Skipping non-measurement process at chain index %d.", i)
            i += 1
            continue

        # --- Measurement process ---
        raw_file: DataFile | None = None
        meas_params: list[ParameterValue] = []

        for output in proc.get("outputs", []):
            oid = output.get("@id", "")
            if oid.startswith("#data_file/"):
                raw_file = data_file_index.get(oid)
                break

        for pv in proc.get("parameterValues", []):
            try:
                meas_params.append(_resolve_param_value(pv, param_def_index, resolver))
            except (TypeError, ValueError, KeyError, AttributeError, PydanticValidationError) as exc:
                logger.warning(
                    "Skipping invalid measurement parameterValue in process '%s': %s",
                    proc.get("@id", "?"),
                    exc,
                )

        # --- Processing process (next in chain) ---
        proc_file: DataFile | None = None
        proc_params: list[ParameterValue] = []

        if i + 1 < len(ordered_procs):
            next_proc = ordered_procs[i + 1]
            next_inputs = next_proc.get("inputs", [])
            next_is_processing = (
                bool(next_inputs)
                and next_inputs[0].get("@id", "").startswith("#data_file/")
            )

            if next_is_processing:
                for output in next_proc.get("outputs", []):
                    oid = output.get("@id", "")
                    if oid.startswith("#data_file/"):
                        proc_file = data_file_index.get(oid)
                        break

                for pv in next_proc.get("parameterValues", []):
                    try:
                        proc_params.append(
                            _resolve_param_value(pv, param_def_index, resolver)
                        )
                    except (
                        TypeError,
                        ValueError,
                        KeyError,
                        AttributeError,
                        PydanticValidationError,
                    ) as exc:
                        logger.warning(
                            "Skipping invalid processing parameterValue in process '%s': %s",
                            next_proc.get("@id", "?"),
                            exc,
                        )

                i += 2  # Consumed both measurement and processing.
            else:
                i += 1  # Only consumed the measurement.
        else:
            i += 1  # Last process, no processing step.

        # Resolve per-run factor values from the measurement process's input sample.
        run_factor_values = dict(study_factor_values)
        if sample_factor_index:
            sample_id = inputs[0].get("@id", "") if inputs else ""
            per_run = sample_factor_index.get(sample_id)
            if per_run:
                run_factor_values = per_run

        run_id = _make_run_id(run_number, max_runs)
        runs.append(
            RunRecord(
                run_id=run_id,
                run_number=run_number,
                raw_file=raw_file,
                processed_file=proc_file,
                measurement_params=meas_params,
                processing_params=proc_params,
                factor_values=run_factor_values,
            )
        )
        run_number += 1

    logger.debug("Extracted %d runs from %d processes.", len(runs), len(ordered_procs))
    return runs
