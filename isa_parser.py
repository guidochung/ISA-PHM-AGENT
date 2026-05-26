"""
ISA-PHM JSON Parser
-------------------
Loads an ISA-PHM JSON file, builds a lookup index of all @id references,
and exposes clean query methods used by the tool functions.

Post-upgrade additions (2026-04-23):
- get_ai_context: normalized semantic manifest (factors + types, assay types,
  experiment_type, PHM objective classification).
- validate_dataset: structured issue report (errors/warnings/info) across
  schema, file linkage, and PHM semantic gates.
- get_experiment_matrix: per-study factor × value matrix with coverage %.
- get_run_inventory: per-sample run ordering + factor values + file paths.
- get_label_coverage: per-study fault/operating label presence summary.
- get_sensor_availability: per-study assay measurement-type map (sparse-map aware).
- classify_phm_objective: heuristic → detection / diagnostics /
  health assessment / prognosis.

Integration with ISA-PHM-Wrapper (2026-05-11):
JSON ingestion is delegated to the `isa_phm` package (local wrapper).
Inputs pass through wrapper.ISAParser → wrapper.ISAPreprocessor before this
class sees them, providing:
  - ISA-JSON top-level key validation (+ optional isatools schema check)
  - 5 auto-fix repair rules: ID whitespace, null descriptions, file-path
    resolution, extension casing, name<-filename backfill
  - a RepairLog stored on the returned parser for transparency.
The 22 tool query methods below are unchanged.
"""

import json
import logging
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Wrapper-backed loading pipeline. Import errors are surfaced at load time
# rather than import time so unit tests that construct ISAParser(dict) keep
# working without isa_phm installed.
try:
    from isa_phm import ISAWrapper as _ISAWrapper
    from isa_phm.parser import ISAParser as _WrapperParser
    from isa_phm.preprocessor import ISAPreprocessor as _WrapperPreprocessor
    _WRAPPER_AVAILABLE = True
    _WRAPPER_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - environment dependent
    _WRAPPER_AVAILABLE = False
    _WRAPPER_IMPORT_ERROR = _exc
    _ISAWrapper = None  # type: ignore

_logger = logging.getLogger("isa_phm.prototype")


def _normalize_rel(p: str) -> str:
    """Normalise an ISA-style relative path for comparison."""
    return p.replace("\\", "/").lstrip("./").lstrip("/").strip().lower()


FAULT_LIKE_TYPES = {"damage", "fault", "fault type", "fault severity"}
OPERATING_LIKE_TYPES = {"operating condition", "operating_condition", "load", "speed"}
DEGRADATION_LIKE_NAMES = {"vb", "rul", "wear", "flank", "degradation"}


class ISAParser:
    def __init__(self, data: dict):
        self.data = data
        self._index: dict[str, Any] = {}
        self._build_index(self.data)
        # Populated by load_from_file / load_from_bytes when the wrapper
        # pipeline runs. Empty when the parser is constructed directly
        # from a dict (legacy / unit-test path).
        self.repair_log: list[dict] = []
        self.preprocessing_source: str = "direct"

        # Wrapper attachment — only populated when a data_root is configured
        # via attach_wrapper(). Signal-level tools route through this.
        self._source_bytes: bytes | None = None
        self._source_filename: str = ""
        self._tempfile_path: Path | None = None
        self.wrapper: Any = None  # ISAWrapper instance when attached
        self.wrapper_status: str = "not_attached"
        self.wrapper_error: str = ""

    # ------------------------------------------------------------------
    # Wrapper attachment for signal-level tools
    # ------------------------------------------------------------------
    def set_source_bytes(self, content: bytes, filename: str) -> None:
        """Store the raw ISA-JSON bytes so the wrapper can be (re)initialised
        later against a user-provided data_root."""
        self._source_bytes = content
        self._source_filename = filename

    def attach_wrapper(self, data_root: str | None) -> tuple[bool, str]:
        """Initialise an ISAWrapper bound to the given data directory.

        Idempotent: re-calling with a new data_root rebuilds the wrapper.
        Returns (success, message).
        """
        if not _WRAPPER_AVAILABLE or _ISAWrapper is None:
            self.wrapper = None
            self.wrapper_status = "wrapper_unavailable"
            self.wrapper_error = f"isa_phm import failed: {_WRAPPER_IMPORT_ERROR!r}"
            return False, self.wrapper_error

        if not data_root:
            self.wrapper = None
            self.wrapper_status = "no_data_root"
            self.wrapper_error = ""
            return False, "No data directory provided."

        data_root_path = Path(data_root).expanduser()
        if not data_root_path.exists() or not data_root_path.is_dir():
            self.wrapper = None
            self.wrapper_status = "invalid_data_root"
            self.wrapper_error = f"Data directory does not exist or is not a folder: {data_root_path}"
            return False, self.wrapper_error

        # ISAWrapper requires a file path. If we only have bytes (Streamlit
        # upload), persist them to a stable temp file.
        if self._tempfile_path is None or not self._tempfile_path.exists():
            if self._source_bytes is None:
                self.wrapper = None
                self.wrapper_status = "no_source"
                self.wrapper_error = "ISA-JSON source bytes were not retained on this parser."
                return False, self.wrapper_error
            suffix = ".json" if not self._source_filename.endswith(".json") else ""
            tmp = tempfile.NamedTemporaryFile(
                prefix="isa_phm_", suffix=f"{Path(self._source_filename).stem}{suffix}.json",
                delete=False,
            )
            tmp.write(self._source_bytes)
            tmp.close()
            self._tempfile_path = Path(tmp.name)

        try:
            self.wrapper = _ISAWrapper(
                path=self._tempfile_path,
                data_root=data_root_path,
                strict_validation=False,   # don't require isatools schema strict pass
                csv_bad_lines="warn",      # be lenient on slightly malformed CSVs
            )
            self.wrapper_status = "attached"
            self.wrapper_error = ""
            return True, f"Wrapper attached. data_root={data_root_path}"
        except Exception as exc:
            self.wrapper = None
            self.wrapper_status = "init_failed"
            self.wrapper_error = f"{type(exc).__name__}: {exc}"
            return False, self.wrapper_error

    def detach_wrapper(self) -> None:
        self.wrapper = None
        self.wrapper_status = "not_attached"
        self.wrapper_error = ""

    # ------------------------------------------------------------------
    # Translation helpers: prototype-level identifiers → wrapper IDs
    # ------------------------------------------------------------------
    def get_assay_filename(self, study_title: str, sensor_alias: str) -> str | None:
        """Map (study_title, sensor_alias) → assay filename used by the wrapper."""
        study = self._find_study(study_title)
        if study is None:
            return None
        assay = self._find_assay(study, sensor_alias)
        if assay is None:
            return None
        return assay.get("filename")

    def get_run_id_for_path(
        self, study_title: str, sensor_alias: str, relative_path: str
    ) -> str | None:
        """Find the wrapper run_id whose data file matches the given relative path.

        Matches by suffix because the wrapper stores resolved absolute paths,
        while the agent passes the original ISA relative path.
        """
        if self.wrapper is None:
            return None
        filename = self.get_assay_filename(study_title, sensor_alias)
        if filename is None:
            return None
        target = _normalize_rel(relative_path)
        for study in self.wrapper.investigation.studies:
            if study.title.lower() != study_title.lower():
                continue
            for assay in study.assays:
                if assay.assay_id != filename:
                    continue
                for run in assay.runs:
                    for df in (run.raw_file, run.processed_file):
                        if df is None:
                            continue
                        candidate = _normalize_rel(str(getattr(df, "path", "")))
                        if candidate == target or candidate.endswith("/" + target) or target.endswith("/" + candidate):
                            return run.run_id
        return None

    # ------------------------------------------------------------------
    # Index builder — recursively registers every object that has an @id
    # ------------------------------------------------------------------
    def _build_index(self, obj: Any) -> None:
        if isinstance(obj, dict):
            if "@id" in obj:
                self._index[obj["@id"]] = obj
            for v in obj.values():
                self._build_index(v)
        elif isinstance(obj, list):
            for item in obj:
                self._build_index(item)

    def _resolve(self, ref: Any) -> Any:
        """Return the full object for an @id reference dict, or the object itself."""
        if isinstance(ref, dict) and "@id" in ref and len(ref) == 1:
            return self._index.get(ref["@id"], ref)
        return ref

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _comment_value(self, comments: list, name: str, default: str = "") -> str:
        for c in comments:
            if c.get("name") == name:
                return c.get("value", default)
        return default

    def _person_name(self, person: dict) -> str:
        parts = [person.get("firstName", ""), person.get("midInitials", ""), person.get("lastName", "")]
        return " ".join(p for p in parts if p).strip()

    def _find_study(self, study_title: str) -> dict | None:
        for s in self.data.get("studies", []):
            if s.get("title", "").lower() == study_title.lower():
                return s
        return None

    def _find_assay(self, study: dict, sensor_alias: str) -> dict | None:
        for a in study.get("assays", []):
            alias = self._comment_value(a.get("comments", []), "sensor alias")
            if alias.lower() == sensor_alias.lower():
                return a
        return None

    def _factor_type(self, factor: dict) -> str:
        ft = factor.get("factorType", {})
        if isinstance(ft, dict):
            return ft.get("annotationValue", "")
        return ""

    def _factor_index(self, study: dict) -> dict[str, dict]:
        """Map study_factor @id → factor object."""
        return {f.get("@id", ""): f for f in study.get("factors", [])}

    def _characteristic_index(self, study: dict) -> dict[str, str]:
        """Map characteristic_category @id → annotation value."""
        out = {}
        for cc in study.get("characteristicCategories", []):
            cid = cc.get("@id", "")
            name = cc.get("characteristicType", {}).get("annotationValue", "")
            out[cid] = name
        return out

    def _assay_alias(self, assay: dict) -> str:
        return self._comment_value(assay.get("comments", []), "sensor alias")

    def _assay_measurement_type(self, assay: dict) -> str:
        return assay.get("measurementType", {}).get("annotationValue", "")

    # ------------------------------------------------------------------
    # Original public query methods
    # ------------------------------------------------------------------

    def get_investigation_overview(self) -> dict:
        d = self.data
        people = []
        for p in d.get("people", []):
            roles = [r.get("annotationValue", "") for r in p.get("roles", [])]
            people.append({
                "name": self._person_name(p),
                "affiliation": p.get("affiliation", ""),
                "email": p.get("email", ""),
                "roles": [r for r in roles if r],
            })

        publications = []
        for pub in d.get("publications", []):
            publications.append({
                "title": pub.get("title", ""),
                "doi": pub.get("doi", ""),
                "status": pub.get("status", {}).get("annotationValue", ""),
            })

        return {
            "description": d.get("description", ""),
            "identifier": d.get("identifier", ""),
            "experiment_type": self._comment_value(d.get("comments", []), "experiment_type"),
            "license": self._comment_value(d.get("comments", []), "license"),
            "public_release_date": d.get("publicReleaseDate", ""),
            "people": people,
            "publications": publications,
            "number_of_studies": len(d.get("studies", [])),
        }

    def list_studies(self) -> list:
        result = []
        for s in self.data.get("studies", []):
            assay_aliases = []
            for a in s.get("assays", []):
                alias = self._assay_alias(a)
                if alias:
                    assay_aliases.append(alias)

            result.append({
                "title": s.get("title", ""),
                "description": s.get("description", ""),
                "identifier": s.get("identifier", ""),
                "submission_date": s.get("submissionDate", ""),
                "public_release_date": s.get("publicReleaseDate", ""),
                "number_of_assays": len(s.get("assays", [])),
                "sensor_aliases": assay_aliases,
                "number_of_samples": len(s.get("materials", {}).get("samples", [])),
                "factors": [f.get("factorName", "") for f in s.get("factors", [])],
            })
        return result

    def get_study_details(self, study_title: str) -> dict | str:
        study = self._find_study(study_title)
        if study is None:
            titles = [s.get("title", "") for s in self.data.get("studies", [])]
            return f"Study '{study_title}' not found. Available studies: {titles}"

        protocols = []
        for proto in study.get("protocols", []):
            params = []
            for param in proto.get("parameters", []):
                pname = param.get("parameterName", {})
                params.append(pname.get("annotationValue", ""))
            protocols.append({
                "name": proto.get("name", ""),
                "description": proto.get("description", ""),
                "parameters": [p for p in params if p],
            })

        char_cats = list(self._characteristic_index(study).values())

        assay_summaries = []
        for a in study.get("assays", []):
            alias = self._assay_alias(a)
            mtype = self._assay_measurement_type(a)
            assay_summaries.append({
                "sensor_alias": alias,
                "measurement_type": mtype,
                "number_of_data_files": len(a.get("dataFiles", [])),
                "number_of_samples": len(a.get("materials", {}).get("samples", [])),
            })

        design_descriptors = [
            d.get("annotationValue", "")
            for d in study.get("studyDesignDescriptors", [])
        ]

        factors_detailed = []
        for f in study.get("factors", []):
            factors_detailed.append({
                "name": f.get("factorName", ""),
                "type": self._factor_type(f),
                "unit": self._comment_value(f.get("comments", []), "unit"),
            })

        return {
            "title": study.get("title", ""),
            "description": study.get("description", ""),
            "identifier": study.get("identifier", ""),
            "submission_date": study.get("submissionDate", ""),
            "public_release_date": study.get("publicReleaseDate", ""),
            "study_design_descriptors": [d for d in design_descriptors if d],
            "factors": factors_detailed,
            "characteristic_categories": [c for c in char_cats if c],
            "protocols": protocols,
            "assays": assay_summaries,
            "number_of_samples": len(study.get("materials", {}).get("samples", [])),
        }

    def get_assay_details(self, study_title: str, sensor_alias: str) -> dict | str:
        study = self._find_study(study_title)
        if study is None:
            titles = [s.get("title", "") for s in self.data.get("studies", [])]
            return f"Study '{study_title}' not found. Available studies: {titles}"

        assay = self._find_assay(study, sensor_alias)
        if assay is None:
            aliases = [self._assay_alias(a) for a in study.get("assays", [])]
            return f"Sensor '{sensor_alias}' not found in study '{study_title}'. Available sensors: {aliases}"

        mtype = self._assay_measurement_type(assay)
        technology_type = assay.get("technologyType", {}).get("annotationValue", "")
        technology_platform = assay.get("technologyPlatform", "")

        data_files = [df.get("name", "") for df in assay.get("dataFiles", [])]

        process_params = []
        for proc in assay.get("processSequence", []):
            proto_ref = proc.get("executesProtocol", {})
            proto = self._resolve(proto_ref)
            proto_name = proto.get("name", "") if isinstance(proto, dict) else ""

            param_name_map = {}
            if isinstance(proto, dict):
                for pp in proto.get("parameters", []):
                    pid = pp.get("@id", "")
                    pname = pp.get("parameterName", {}).get("annotationValue", "")
                    param_name_map[pid] = pname

            resolved_params = {}
            for pv in proc.get("parameterValues", []):
                cat_id = pv.get("category", {}).get("@id", "")
                param_name = param_name_map.get(cat_id, cat_id)
                value = pv.get("value", "")
                unit_ref = pv.get("unit", {})
                unit_obj = self._resolve(unit_ref) if unit_ref else {}
                unit = unit_obj.get("annotationValue", "") if isinstance(unit_obj, dict) else ""
                resolved_params[param_name] = f"{value} {unit}".strip() if unit else str(value)

            if resolved_params:
                process_params.append({
                    "process_name": proc.get("name", ""),
                    "protocol": proto_name,
                    "parameters": resolved_params,
                })

        return {
            "sensor_alias": sensor_alias,
            "study": study_title,
            "measurement_type": mtype,
            "technology_type": technology_type,
            "technology_platform": technology_platform,
            "number_of_data_files": len(data_files),
            "data_files": data_files,
            "processes": process_params,
        }

    def list_data_files(self, study_title: str | None = None, sensor_alias: str | None = None) -> list | str:
        studies = self.data.get("studies", [])
        if study_title:
            study = self._find_study(study_title)
            if study is None:
                titles = [s.get("title", "") for s in studies]
                return f"Study '{study_title}' not found. Available studies: {titles}"
            studies = [study]

        results = []
        for s in studies:
            assays = s.get("assays", [])
            if sensor_alias:
                assay = self._find_assay(s, sensor_alias)
                assays = [assay] if assay else []

            for a in assays:
                alias = self._assay_alias(a)
                for df in a.get("dataFiles", []):
                    results.append({
                        "study": s.get("title", ""),
                        "sensor": alias,
                        "file": df.get("name", ""),
                        "type": df.get("type", ""),
                    })
        return results

    def search_metadata(self, term: str) -> dict:
        term_lower = term.lower()
        matches = {
            "investigation_fields": [],
            "studies": [],
            "assays": [],
            "protocols": [],
            "people": [],
        }

        for field in ["description", "identifier"]:
            val = self.data.get(field, "")
            if term_lower in val.lower():
                matches["investigation_fields"].append({"field": field, "value": val[:300]})

        for p in self.data.get("people", []):
            name = self._person_name(p)
            affil = p.get("affiliation", "")
            if term_lower in name.lower() or term_lower in affil.lower():
                matches["people"].append({"name": name, "affiliation": affil})

        for s in self.data.get("studies", []):
            study_title = s.get("title", "")
            for field in ["title", "description"]:
                val = s.get(field, "")
                if term_lower in val.lower():
                    matches["studies"].append({"study": study_title, "field": field, "value": val[:300]})
                    break

            for f in s.get("factors", []):
                if term_lower in f.get("factorName", "").lower():
                    matches["studies"].append({"study": study_title, "field": "factor", "value": f.get("factorName", "")})

            for proto in s.get("protocols", []):
                pname = proto.get("name", "")
                pdesc = proto.get("description", "")
                if term_lower in pname.lower() or term_lower in pdesc.lower():
                    matches["protocols"].append({"study": study_title, "protocol": pname, "description": pdesc[:200]})
                for param in proto.get("parameters", []):
                    pval = param.get("parameterName", {}).get("annotationValue", "")
                    if term_lower in pval.lower():
                        matches["protocols"].append({"study": study_title, "protocol": pname, "parameter": pval})

            for a in s.get("assays", []):
                alias = self._assay_alias(a)
                mtype = self._assay_measurement_type(a)
                if term_lower in alias.lower() or term_lower in mtype.lower():
                    matches["assays"].append({"study": study_title, "sensor": alias, "measurement_type": mtype})

                for proc in a.get("processSequence", []):
                    for pv in proc.get("parameterValues", []):
                        val = str(pv.get("value", ""))
                        if term_lower in val.lower():
                            matches["assays"].append({
                                "study": study_title,
                                "sensor": alias,
                                "process": proc.get("name", ""),
                                "matching_value": val,
                            })

        for key in matches:
            seen = set()
            deduped = []
            for item in matches[key]:
                s_item = str(item)
                if s_item not in seen:
                    seen.add(s_item)
                    deduped.append(item)
            matches[key] = deduped

        total = sum(len(v) for v in matches.values())
        matches["total_matches"] = total
        matches["search_term"] = term
        return matches

    # ------------------------------------------------------------------
    # Upgrade (2026-04-23): semantic helpers
    # ------------------------------------------------------------------

    def _classify_factor(self, factor: dict) -> str:
        """Classify a factor as fault / operating / degradation / other based on
        its factorType and factorName."""
        ftype = self._factor_type(factor).lower().strip()
        fname = (factor.get("factorName") or "").lower().strip()

        if ftype in FAULT_LIKE_TYPES:
            return "fault"
        if ftype in OPERATING_LIKE_TYPES:
            return "operating_condition"
        if any(k in fname for k in DEGRADATION_LIKE_NAMES):
            return "degradation"
        return "other"

    def get_ai_context(self, include_semantics: bool = True) -> dict:
        """Normalized PHM view of the investigation: experiment_type, studies
        with factor classifications, assay measurement types, and a semantic
        manifest. Use this before analysis; everything is derived from metadata,
        no CSV access needed."""
        overview = self.get_investigation_overview()
        studies_out = []
        all_factor_names: set[str] = set()
        all_measurement_types: set[str] = set()

        for s in self.data.get("studies", []):
            study_factors = []
            for f in s.get("factors", []):
                cls = self._classify_factor(f)
                study_factors.append({
                    "name": f.get("factorName", ""),
                    "type": self._factor_type(f),
                    "unit": self._comment_value(f.get("comments", []), "unit"),
                    "phm_class": cls,
                })
                if f.get("factorName"):
                    all_factor_names.add(f["factorName"])

            assays = []
            for a in s.get("assays", []):
                mtype = self._assay_measurement_type(a)
                alias = self._assay_alias(a)
                if mtype:
                    all_measurement_types.add(mtype)
                assays.append({
                    "sensor_alias": alias,
                    "measurement_type": mtype,
                    "technology_type": a.get("technologyType", {}).get("annotationValue", ""),
                    "n_data_files": len(a.get("dataFiles", [])),
                    "n_samples": len(a.get("materials", {}).get("samples", [])),
                })

            n_runs = len(s.get("materials", {}).get("samples", []))

            studies_out.append({
                "study_id": s.get("identifier", ""),
                "title": s.get("title", ""),
                "description": s.get("description", ""),
                "n_assays": len(s.get("assays", [])),
                "n_runs": n_runs,
                "factors": study_factors,
                "assays": assays,
            })

        result = {
            "experiment_type": overview["experiment_type"],
            "description": overview["description"],
            "identifier": overview["identifier"],
            "n_studies": overview["number_of_studies"],
            "studies": studies_out,
            "phm_objective": self.classify_phm_objective(),
        }

        if include_semantics:
            fault_factors = []
            operating_factors = []
            degradation_factors = []
            for s in studies_out:
                for f in s["factors"]:
                    entry = {"study": s["title"], "factor": f["name"], "unit": f["unit"], "type": f["type"]}
                    if f["phm_class"] == "fault":
                        fault_factors.append(entry)
                    elif f["phm_class"] == "operating_condition":
                        operating_factors.append(entry)
                    elif f["phm_class"] == "degradation":
                        degradation_factors.append(entry)

            result["semantic_manifest"] = {
                "fault_factors": fault_factors,
                "operating_condition_factors": operating_factors,
                "degradation_factors": degradation_factors,
                "all_factor_names": sorted(all_factor_names),
                "all_measurement_types": sorted(all_measurement_types),
            }

        return result

    def classify_phm_objective(self) -> dict:
        """Heuristic PHM objective classification from experiment_type +
        factor types. Returns {objective, confidence, reasoning}."""
        exp_type = self._comment_value(self.data.get("comments", []), "experiment_type").lower()
        description = self.data.get("description", "").lower()

        has_fault = False
        has_operating = False
        has_degradation = False
        max_runs_per_study = 0

        for s in self.data.get("studies", []):
            max_runs_per_study = max(
                max_runs_per_study,
                len(s.get("materials", {}).get("samples", [])),
            )
            for f in s.get("factors", []):
                cls = self._classify_factor(f)
                if cls == "fault":
                    has_fault = True
                elif cls == "operating_condition":
                    has_operating = True
                elif cls == "degradation":
                    has_degradation = True

        reasoning_parts = []
        objective = "unknown"
        confidence = "low"

        if "prognostic" in exp_type or has_degradation or max_runs_per_study > 5:
            objective = "prognosis"
            confidence = "high" if "prognostic" in exp_type and has_degradation else "medium"
            reasoning_parts.append(f"experiment_type='{exp_type}'")
            if has_degradation:
                reasoning_parts.append("degradation/RUL-like factor present")
            if max_runs_per_study > 5:
                reasoning_parts.append(f"max {max_runs_per_study} runs per study (trajectory-like)")
        elif "diagnostic" in exp_type or has_fault:
            objective = "diagnostics"
            confidence = "high" if "diagnostic" in exp_type and has_fault else "medium"
            reasoning_parts.append(f"experiment_type='{exp_type}'")
            if has_fault:
                reasoning_parts.append("fault-type factor present")
        elif "detection" in exp_type:
            objective = "detection"
            confidence = "medium"
            reasoning_parts.append(f"experiment_type='{exp_type}'")
        elif has_operating and not has_fault and not has_degradation:
            objective = "health assessment"
            confidence = "low"
            reasoning_parts.append("only operating-condition factors found — not diagnostic/prognostic")

        # Inconsistency detection
        inconsistency = None
        if "prognostic" in exp_type and "diagnostic" in description and "prognostic" not in description:
            inconsistency = (
                f"experiment_type is '{exp_type}' but description says diagnostics. "
                "Flag this conflict before reasoning about labels."
            )
        elif "diagnostic" in exp_type and "prognostic" in description and "diagnostic" not in description:
            inconsistency = (
                f"experiment_type is '{exp_type}' but description says prognostics. "
                "Flag this conflict before reasoning about labels."
            )

        return {
            "objective": objective,
            "confidence": confidence,
            "reasoning": "; ".join(reasoning_parts) if reasoning_parts else "no strong signals found",
            "inconsistency_warning": inconsistency,
        }

    def validate_dataset(
        self,
        semantic_strict: bool = False,
        check_merge: bool = False,
        merge_target_parser: "ISAParser | None" = None,
    ) -> dict:
        """Run the 4 PHM quality gates:
        1) structural, 2) semantic, 3) prognostic readiness,
        4) merge readiness (only if check_merge=True; needs merge_target_parser)."""
        errors: list[str] = []
        warnings: list[str] = []
        info: list[str] = []

        # Gate 1 — structural
        if not self.data.get("identifier"):
            warnings.append("Investigation identifier is empty.")
        if not self.data.get("description"):
            warnings.append("Investigation description is empty.")
        if not self._comment_value(self.data.get("comments", []), "experiment_type"):
            errors.append("experiment_type comment is missing at the investigation level.")
        if not self.data.get("studies"):
            errors.append("No studies found in investigation.")

        for s in self.data.get("studies", []):
            title = s.get("title", "<untitled>")
            if not s.get("factors"):
                warnings.append(f"Study '{title}' has no factors defined.")
            if not s.get("assays"):
                errors.append(f"Study '{title}' has no assays.")
            if not s.get("materials", {}).get("samples"):
                warnings.append(f"Study '{title}' has no samples (runs).")

            for a in s.get("assays", []):
                alias = self._assay_alias(a) or "<no-alias>"
                if not a.get("dataFiles"):
                    warnings.append(f"Assay {title}/{alias} has no dataFiles listed.")
                if not self._assay_measurement_type(a):
                    warnings.append(f"Assay {title}/{alias} has no measurementType.")

        # Gate 2 — PHM semantic
        ctx = self.get_ai_context(include_semantics=True)
        sm = ctx["semantic_manifest"]
        if not sm["fault_factors"] and not sm["degradation_factors"]:
            msg = "No fault-type or degradation factors found across any study."
            if semantic_strict:
                errors.append(msg)
            else:
                warnings.append(msg)
        if not sm["operating_condition_factors"]:
            msg = "No operating-condition factors found across any study."
            if semantic_strict:
                errors.append(msg)
            else:
                warnings.append(msg)

        # Gate 3 — PHM ambition
        objective = ctx["phm_objective"]
        if objective["inconsistency_warning"]:
            warnings.append(f"PHM objective inconsistency: {objective['inconsistency_warning']}")

        if objective["objective"] == "prognosis":
            for s in self.data.get("studies", []):
                n_runs = len(s.get("materials", {}).get("samples", []))
                if n_runs < 2:
                    warnings.append(
                        f"Prognosis objective inferred, but study '{s.get('title', '')}' "
                        f"has only {n_runs} run(s) — cannot form a trajectory."
                    )

        # Gate 4 — Merge readiness (only when explicitly requested with a partner)
        merge_status = "skipped"
        merge_details: dict = {}
        if check_merge:
            if merge_target_parser is None:
                warnings.append(
                    "check_merge=True but no merge_target_parser provided; skipped merge gate."
                )
            else:
                target_ctx = merge_target_parser.get_ai_context(include_semantics=True)
                target_sm = target_ctx["semantic_manifest"]
                this_factors = set(sm["all_factor_names"])
                target_factors = set(target_sm["all_factor_names"])
                this_mtypes = set(sm["all_measurement_types"])
                target_mtypes = set(target_sm["all_measurement_types"])

                shared_factors = this_factors & target_factors
                shared_mtypes = this_mtypes & target_mtypes

                experiment_match = ctx["experiment_type"] == target_ctx["experiment_type"]

                merge_details = {
                    "experiment_type_match": experiment_match,
                    "shared_factor_names": sorted(shared_factors),
                    "shared_measurement_types": sorted(shared_mtypes),
                    "this_only_factors": sorted(this_factors - target_factors),
                    "target_only_factors": sorted(target_factors - this_factors),
                    "factor_overlap_pct": (
                        round(100.0 * len(shared_factors) / max(len(this_factors | target_factors), 1), 1)
                    ),
                }

                if not experiment_match:
                    msg = (
                        f"Merge gate: experiment_type mismatch "
                        f"({ctx['experiment_type']} vs {target_ctx['experiment_type']}). "
                        "Cross-type fusion is rarely meaningful."
                    )
                    if semantic_strict:
                        errors.append(msg)
                    else:
                        warnings.append(msg)
                if not shared_mtypes:
                    msg = "Merge gate: no shared sensor measurement types between datasets."
                    if semantic_strict:
                        errors.append(msg)
                    else:
                        warnings.append(msg)
                if not shared_factors:
                    warnings.append(
                        "Merge gate: no shared study factor names — factor crosswalk required."
                    )

                merge_status = (
                    "pass" if (experiment_match and shared_mtypes and shared_factors) else "warn"
                )
                if not experiment_match and not shared_mtypes:
                    merge_status = "fail"

        status = "fail" if errors else ("warn" if warnings else "pass")
        return {
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "phm_objective": objective,
            "merge_details": merge_details if check_merge else None,
            "gate_summary": {
                "1_structural": "pass" if not any("investigation" in e.lower() or "No studies" in e for e in errors) else "fail",
                "2_semantic": "pass" if sm["operating_condition_factors"] and (sm["fault_factors"] or sm["degradation_factors"]) else "warn",
                "3_ambition": "pass" if objective["objective"] != "unknown" and not objective["inconsistency_warning"] else "warn",
                "4_merge_readiness": merge_status,
            },
        }

    def get_experiment_matrix(self) -> dict:
        """Per-study × factor matrix.
        Because ISA-PHM samples often store factor *values* as per-run setting-file
        paths rather than literal values, this matrix reports: for each study,
        the factor, its unit, how many samples carry a factorValue entry for
        that factor, and the first 3 example values (raw). Coverage is
        samples_with_value / total_samples."""
        rows = []
        for s in self.data.get("studies", []):
            title = s.get("title", "")
            factors = s.get("factors", [])
            samples = s.get("materials", {}).get("samples", [])
            total = len(samples)
            factor_map = self._factor_index(s)

            per_factor_values: dict[str, list] = defaultdict(list)
            for samp in samples:
                for fv in samp.get("factorValues", []):
                    cid = fv.get("category", {}).get("@id", "")
                    per_factor_values[cid].append(fv.get("value", ""))

            for f in factors:
                fid = f.get("@id", "")
                name = f.get("factorName", "")
                unit = self._comment_value(f.get("comments", []), "unit")
                vals = per_factor_values.get(fid, [])
                rows.append({
                    "study": title,
                    "factor": name,
                    "phm_class": self._classify_factor(f),
                    "unit": unit,
                    "n_values": len(vals),
                    "n_samples": total,
                    "coverage_pct": round(100.0 * len(vals) / total, 1) if total else 0.0,
                    "example_values": vals[:3],
                })

        return {
            "rows": rows,
            "note": (
                "ISA-PHM sample.factorValues.value is sometimes a per-run setting-file "
                "path (e.g. './Case_01/Settings/CS/Case_01_time_run_01.csv'), not a "
                "literal numeric value. Agents should flag this when a user asks for "
                "literal factor values per run."
            ),
        }

    def get_run_inventory(self, study_title: str) -> dict | str:
        """Per-sample run inventory for a study: sample name, factor values
        (resolved by factor name), and the data-file paths from each assay that
        share its @id."""
        study = self._find_study(study_title)
        if study is None:
            titles = [s.get("title", "") for s in self.data.get("studies", [])]
            return f"Study '{study_title}' not found. Available studies: {titles}"

        factor_map = self._factor_index(study)
        samples = study.get("materials", {}).get("samples", [])

        sample_to_files: dict[str, list] = defaultdict(list)
        for a in study.get("assays", []):
            alias = self._assay_alias(a)
            for proc in a.get("processSequence", []):
                input_ids = [i.get("@id", "") for i in proc.get("inputs", [])]
                output_ids = [o.get("@id", "") for o in proc.get("outputs", [])]
                for in_id in input_ids:
                    for out_id in output_ids:
                        ref = self._resolve({"@id": out_id})
                        if isinstance(ref, dict):
                            fname = ref.get("name", "")
                            if fname.endswith(".csv") or ".csv" in fname:
                                sample_to_files[in_id].append({
                                    "sensor": alias,
                                    "file": fname,
                                })

        runs = []
        for idx, samp in enumerate(samples, start=1):
            sid = samp.get("@id", "")
            factor_vals = {}
            for fv in samp.get("factorValues", []):
                cid = fv.get("category", {}).get("@id", "")
                fname = factor_map.get(cid, {}).get("factorName", cid)
                factor_vals[fname] = fv.get("value", "")
            characteristics = {}
            char_index = self._characteristic_index(study)
            for c in samp.get("characteristics", []):
                cid = c.get("category", {}).get("@id", "")
                cname = char_index.get(cid, cid)
                characteristics[cname] = c.get("value", "")

            runs.append({
                "run_index": idx,
                "sample_name": samp.get("name", ""),
                "sample_id": sid,
                "factor_values": factor_vals,
                "characteristics": characteristics,
                "data_files": sample_to_files.get(sid, []),
            })

        expected_runs = int(self._comment_value(study.get("comments", []), "total_runs") or 0)
        continuity = {
            "n_samples": len(samples),
            "declared_total_runs": expected_runs,
            "match": expected_runs == len(samples) if expected_runs else None,
        }

        return {
            "study": study_title,
            "runs": runs,
            "continuity": continuity,
        }

    def get_label_coverage(self) -> dict:
        """Per-study label coverage: fault label, degradation label, operating
        condition labels — how many runs have a non-empty factorValue."""
        rows = []
        for s in self.data.get("studies", []):
            title = s.get("title", "")
            samples = s.get("materials", {}).get("samples", [])
            factors = s.get("factors", [])
            factor_map = self._factor_index(s)
            total = len(samples)

            counts = {"fault": 0, "degradation": 0, "operating_condition": 0, "other": 0}
            factor_cls = {f.get("@id", ""): self._classify_factor(f) for f in factors}
            factor_populated = defaultdict(int)

            for samp in samples:
                seen_classes = set()
                for fv in samp.get("factorValues", []):
                    cid = fv.get("category", {}).get("@id", "")
                    cls = factor_cls.get(cid, "other")
                    val = fv.get("value", "")
                    if val not in ("", None):
                        factor_populated[cid] += 1
                        if cls not in seen_classes:
                            counts[cls] = counts.get(cls, 0) + 1
                            seen_classes.add(cls)

            rows.append({
                "study": title,
                "n_runs": total,
                "fault_label_runs": counts["fault"],
                "degradation_label_runs": counts["degradation"],
                "operating_condition_runs": counts["operating_condition"],
                "coverage": {
                    "fault_pct": round(100.0 * counts["fault"] / total, 1) if total else 0.0,
                    "degradation_pct": round(100.0 * counts["degradation"] / total, 1) if total else 0.0,
                    "operating_pct": round(100.0 * counts["operating_condition"] / total, 1) if total else 0.0,
                },
            })
        return {"rows": rows}

    def get_sensor_availability(self) -> dict:
        """Per-study assay-type map. Useful for Sietze-style sparse / asynchronous
        modality designs where some studies use only vibration, others only
        electrical."""
        rows = []
        global_types: Counter = Counter()
        for s in self.data.get("studies", []):
            title = s.get("title", "")
            types: Counter = Counter()
            aliases = []
            for a in s.get("assays", []):
                mtype = self._assay_measurement_type(a) or "Unknown"
                types[mtype] += 1
                global_types[mtype] += 1
                aliases.append(self._assay_alias(a))
            rows.append({
                "study": title,
                "measurement_types": dict(types),
                "sensor_aliases": aliases,
            })

        all_types = sorted(global_types.keys())
        sparse_flag = False
        for r in rows:
            if set(r["measurement_types"].keys()) != set(all_types):
                sparse_flag = True
                r["sparse_vs_dataset"] = True
            else:
                r["sparse_vs_dataset"] = False

        return {
            "rows": rows,
            "all_measurement_types": all_types,
            "sparse_mapping_detected": sparse_flag,
            "note": (
                "sparse_mapping_detected=True means not every study has every assay "
                "type. In designs like Sietze, this is intentional (asynchronous "
                "modalities); don't flag as a quality failure by default."
            ),
        }

    # ------------------------------------------------------------------
    # Replication / J-series support
    # ------------------------------------------------------------------

    # Heuristic keyword groups for replication-aware bucketing.
    _REPL_BUCKETS = {
        "sampling": {"sample", "sampling", "rate", "frequency_range", "fs", "hz", "duration", "samples"},
        "filter": {"filter", "high pass", "low pass", "band", "cutoff"},
        "amplifier": {"amplifier", "gain", "amplification", "sensitivity"},
        "placement": {"location", "position", "axis", "placement", "mount"},
        "fault_intro": {"fault", "defect", "seeded", "introduction", "severity", "wear", "vb", "size"},
        "operating_condition": {"speed", "rpm", "load", "feed", "depth", "temperature", "torque", "voltage"},
        "rig": {"rig", "platform", "setup", "machine", "spindle", "bearing", "tool", "configuration"},
    }

    def _bucket_key(self, key: str) -> str:
        kl = (key or "").lower()
        for bucket, words in self._REPL_BUCKETS.items():
            if any(w in kl for w in words):
                return bucket
        return "other"

    def get_protocol_details(self, study_title: str) -> dict | str:
        """Replication-grade protocol extraction (Student Guide J-series).
        Walks protocols, processSequence parameter values, characteristic
        categories, and study factors — buckets them into the replication
        categories: sampling / filter / amplifier / placement / fault_intro /
        operating_condition / rig / other. Returns one record per assay."""
        study = self._find_study(study_title)
        if study is None:
            titles = [s.get("title", "") for s in self.data.get("studies", [])]
            return f"Study '{study_title}' not found. Available studies: {titles}"

        # Per-study buckets accumulated across all assays
        buckets: dict[str, list] = {b: [] for b in list(self._REPL_BUCKETS.keys()) + ["other"]}

        # 1) Study factors → bucket by factor name
        for f in study.get("factors", []):
            fname = f.get("factorName", "")
            unit = self._comment_value(f.get("comments", []), "unit")
            entry = {
                "source": "study_factor",
                "name": fname,
                "type": self._factor_type(f),
                "unit": unit,
            }
            buckets[self._bucket_key(fname)].append(entry)

        # 2) Characteristic categories (rig-level)
        for cid, cname in self._characteristic_index(study).items():
            buckets[self._bucket_key(cname)].append({
                "source": "characteristic_category",
                "name": cname,
            })

        # 3) Per-assay protocol + processSequence
        per_assay = []
        for a in study.get("assays", []):
            alias = self._assay_alias(a)
            mtype = self._assay_measurement_type(a)
            tech = a.get("technologyType", {}).get("annotationValue", "")
            platform = a.get("technologyPlatform", "")

            assay_params: dict[str, dict] = {}
            for proc in a.get("processSequence", []):
                proto_ref = proc.get("executesProtocol", {})
                proto = self._resolve(proto_ref)
                proto_name = proto.get("name", "") if isinstance(proto, dict) else ""

                pmap = {}
                if isinstance(proto, dict):
                    for pp in proto.get("parameters", []):
                        pid = pp.get("@id", "")
                        pname = pp.get("parameterName", {}).get("annotationValue", "")
                        pmap[pid] = pname

                for pv in proc.get("parameterValues", []):
                    cat_id = pv.get("category", {}).get("@id", "")
                    pname = pmap.get(cat_id, cat_id)
                    val = pv.get("value", "")
                    unit_obj = self._resolve(pv.get("unit", {})) if pv.get("unit") else {}
                    unit = unit_obj.get("annotationValue", "") if isinstance(unit_obj, dict) else ""
                    if pname not in assay_params:
                        assay_params[pname] = {
                            "value": val,
                            "unit": unit,
                            "protocol": proto_name,
                        }
                        bucket = self._bucket_key(pname)
                        buckets[bucket].append({
                            "source": f"assay/{alias}",
                            "name": pname,
                            "value": val,
                            "unit": unit,
                            "protocol": proto_name,
                        })

            per_assay.append({
                "sensor_alias": alias,
                "measurement_type": mtype,
                "technology_type": tech,
                "technology_platform": platform,
                "n_data_files": len(a.get("dataFiles", [])),
                "parameters": assay_params,
            })

        n_runs = len(study.get("materials", {}).get("samples", []))

        return {
            "study": study_title,
            "n_runs": n_runs,
            "study_design_descriptors": [
                d.get("annotationValue", "")
                for d in study.get("studyDesignDescriptors", [])
                if d.get("annotationValue")
            ],
            "buckets": {
                "rig_and_configuration": buckets["rig"],
                "fault_introduction": buckets["fault_intro"],
                "operating_conditions": buckets["operating_condition"],
                "sensor_placement": buckets["placement"],
                "sampling_and_acquisition": buckets["sampling"],
                "filter_settings": buckets["filter"],
                "amplifier_settings": buckets["amplifier"],
                "other": buckets["other"],
            },
            "assays": per_assay,
        }

    def get_replication_gaps(self, study_title: str) -> dict | str:
        """Score replication completeness for a study against the Student
        Guide J-series criteria. Returns present-and-sufficient vs
        missing-or-underspecified lists, plus an overall readiness rating."""
        details = self.get_protocol_details(study_title)
        if isinstance(details, str):
            return details

        criteria = [
            ("rig_documented", "rig_and_configuration", "Test rig / configuration documented"),
            ("fault_introduction_documented", "fault_introduction", "Fault introduction method documented"),
            ("operating_conditions_documented", "operating_conditions", "Operating conditions documented"),
            ("sensor_placement_documented", "sensor_placement", "Sensor placement / mounting documented"),
            ("sampling_documented", "sampling_and_acquisition", "Sampling rate / acquisition documented"),
            ("filter_documented", "filter_settings", "Filter settings documented"),
            ("amplifier_documented", "amplifier_settings", "Amplifier / gain settings documented"),
        ]

        present: list[dict] = []
        missing: list[dict] = []
        for key, bucket_name, label in criteria:
            entries = details["buckets"].get(bucket_name, [])
            if entries:
                present.append({"criterion": label, "n_fields": len(entries)})
            else:
                missing.append({"criterion": label, "fix": f"Add fields for {bucket_name.replace('_', ' ')} in the test setup."})

        # Hardware specifics: any assay with explicit technology platform = good
        has_platform = any(a.get("technology_platform") for a in details["assays"])
        if has_platform:
            present.append({"criterion": "Sensor hardware (technology platform) documented", "n_fields": sum(1 for a in details["assays"] if a.get("technology_platform"))})
        else:
            missing.append({
                "criterion": "Sensor hardware (technology platform) documented",
                "fix": "Add technologyPlatform per assay (e.g. 'Kistler 8702B500', 'PCB 352C33').",
            })

        # Citations / contributors at investigation level
        has_publication = bool(self.data.get("publications"))
        if has_publication:
            present.append({"criterion": "Citation / publication available", "n_fields": len(self.data.get("publications", []))})
        else:
            missing.append({
                "criterion": "Citation / publication available",
                "fix": "Add at least one publication entry to the investigation metadata.",
            })

        score = len(present)
        total = len(present) + len(missing)
        if total == 0:
            readiness = "unknown"
        elif score / total >= 0.8:
            readiness = "full"
        elif score / total >= 0.5:
            readiness = "partial"
        else:
            readiness = "not_reproducible"

        most_critical = None
        if missing:
            priority_order = [
                "Sampling rate / acquisition documented",
                "Sensor placement / mounting documented",
                "Test rig / configuration documented",
                "Fault introduction method documented",
                "Operating conditions documented",
            ]
            for crit in priority_order:
                for m in missing:
                    if m["criterion"] == crit:
                        most_critical = m["criterion"]
                        break
                if most_critical:
                    break
            if not most_critical:
                most_critical = missing[0]["criterion"]

        return {
            "study": study_title,
            "readiness": readiness,
            "score": f"{score}/{total}",
            "present_and_sufficient": present,
            "missing_or_underspecified": missing,
            "most_critical_missing": most_critical,
        }

    # ------------------------------------------------------------------
    # Sensor-to-visualization compatibility (Playbook §2.3)
    # ------------------------------------------------------------------

    # Measurement types that are intrinsically high-rate enough for FFT/PSD/Spectrogram
    HIGH_RATE_MTYPES = {"vibration", "acoustic emission", "current", "voltage", "force", "microphone", "audio"}
    # Sensor aliases that hint at slow process variables
    LOW_RATE_HINTS = {"temp", "temperature", "pressure", "rpm", "speed", "load_cell"}

    def get_sensor_compatibility(self, study_title: str, sensor_alias: str) -> dict | str:
        """Return whether the sensor is high-rate / low-rate / unknown and
        which plot kinds are compatible. Used by the agent before generating
        any plot (Playbook §2.3, Student Guide G-section)."""
        study = self._find_study(study_title)
        if study is None:
            titles = [s.get("title", "") for s in self.data.get("studies", [])]
            return f"Study '{study_title}' not found. Available studies: {titles}"
        assay = self._find_assay(study, sensor_alias)
        if assay is None:
            aliases = [self._assay_alias(a) for a in study.get("assays", [])]
            return f"Sensor '{sensor_alias}' not found in study '{study_title}'. Available sensors: {aliases}"

        mtype = (self._assay_measurement_type(assay) or "").lower()
        tech = (assay.get("technologyType", {}).get("annotationValue", "") or "").lower()
        alias = (sensor_alias or "").lower()

        # Try to infer sampling rate from process parameters
        sampling_rate_hz: float | None = None
        for proc in assay.get("processSequence", []):
            proto = self._resolve(proc.get("executesProtocol", {}))
            pmap = {}
            if isinstance(proto, dict):
                for pp in proto.get("parameters", []):
                    pmap[pp.get("@id", "")] = pp.get("parameterName", {}).get("annotationValue", "").lower()
            for pv in proc.get("parameterValues", []):
                pname = pmap.get(pv.get("category", {}).get("@id", ""), "")
                val = pv.get("value", "")
                unit_obj = self._resolve(pv.get("unit", {})) if pv.get("unit") else {}
                unit = (unit_obj.get("annotationValue", "") if isinstance(unit_obj, dict) else "").lower()
                if "sample" in pname and "rate" in pname or pname in {"fs", "sampling rate"}:
                    try:
                        rate = float(val)
                        if "khz" in unit:
                            rate *= 1000
                        sampling_rate_hz = rate
                    except (TypeError, ValueError):
                        pass

        # Tier classification
        is_high_rate = (
            any(h in mtype for h in self.HIGH_RATE_MTYPES)
            or any(h in tech for h in self.HIGH_RATE_MTYPES)
        )
        is_low_rate_hint = any(h in alias for h in self.LOW_RATE_HINTS) or any(
            h in mtype for h in {"temperature", "pressure", "rpm"}
        )

        if sampling_rate_hz is not None:
            tier = "high_rate" if sampling_rate_hz >= 1000 else "low_rate"
        elif is_high_rate and not is_low_rate_hint:
            tier = "high_rate"
        elif is_low_rate_hint:
            tier = "low_rate"
        else:
            tier = "unknown"

        # Compatibility table from Playbook §2.3
        all_plots = ["timeseries", "fft", "psd", "spectrogram", "lifecycle", "outlier_comparison"]
        if tier == "high_rate":
            compatible = list(all_plots)
            incompatible = []
        elif tier == "low_rate":
            compatible = ["timeseries", "lifecycle", "outlier_comparison"]
            incompatible = [
                {"plot": "fft", "reason": "FFT is not meaningful for slow process variables (effective sampling rate too low to resolve fault frequencies)."},
                {"plot": "psd", "reason": "PSD requires a high-rate signal; slow process variables produce uninformative spectra."},
                {"plot": "spectrogram", "reason": "Spectrograms require high-rate time-series data."},
            ]
        else:
            compatible = ["timeseries", "lifecycle", "outlier_comparison"]
            incompatible = [
                {"plot": "fft", "reason": "Sensor tier unknown (no sampling rate documented). Verify high-rate before requesting frequency-domain plots."},
                {"plot": "psd", "reason": "Sensor tier unknown."},
                {"plot": "spectrogram", "reason": "Sensor tier unknown."},
            ]

        return {
            "study": study_title,
            "sensor_alias": sensor_alias,
            "measurement_type": self._assay_measurement_type(assay),
            "technology_type": assay.get("technologyType", {}).get("annotationValue", ""),
            "sensor_tier": tier,
            "sampling_rate_hz": sampling_rate_hz,
            "compatible_plots": compatible,
            "incompatible_plots": incompatible,
            "fallback_suggestion": (
                "Use timeseries + lifecycle if frequency-domain plots are not available."
                if tier != "high_rate" else None
            ),
        }

    def compare_studies(self, study_titles: list[str]) -> dict:
        """Side-by-side comparison of multiple studies in the same investigation."""
        out = {}
        for title in study_titles:
            details = self.get_study_details(title)
            if isinstance(details, str):
                out[title] = {"error": details}
            else:
                out[title] = {
                    "factors": details["factors"],
                    "assays": details["assays"],
                    "n_samples": details["number_of_samples"],
                    "characteristic_categories": details["characteristic_categories"],
                }
        common_factor_names = None
        for title, d in out.items():
            if "error" in d:
                continue
            names = {f["name"] for f in d["factors"]}
            common_factor_names = names if common_factor_names is None else common_factor_names & names
        return {
            "studies": out,
            "common_factor_names": sorted(common_factor_names) if common_factor_names else [],
        }


def _serialize_repair_log(log: Any) -> list[dict]:
    """Convert a wrapper RepairLog into a plain list of dicts for JSON-safe storage."""
    out: list[dict] = []
    try:
        for action in log:
            if hasattr(action, "model_dump"):
                out.append(action.model_dump())
            elif isinstance(action, dict):
                out.append(action)
            else:
                out.append({"message": str(action)})
    except TypeError:
        # Not iterable — store nothing rather than crashing.
        return []
    return out


def _load_through_wrapper(
    json_text: str,
    source_hint: str,
    data_root: Path | None,
    strict_schema: bool = False,
) -> ISAParser:
    """Run the ISA-PHM-Wrapper parse + preprocess pipeline, then hand the
    repaired dict to our existing ISAParser. All 22 query methods continue
    to operate on the same dict shape, but the input has been schema-
    checked and auto-repaired upstream."""
    if not _WRAPPER_AVAILABLE:
        raise RuntimeError(
            "ISA-PHM-Wrapper is not installed. Run "
            "`pip install -e /Users/guido/Desktop/ISA/ISA-PHM-Wrapper` "
            f"(import error: {_WRAPPER_IMPORT_ERROR!r})."
        )

    wparser = _WrapperParser(strict=strict_schema)
    raw = wparser.load_string(json_text, source_hint=source_hint)

    # Preprocessor needs a data_root for path-resolution rules. When the user
    # uploads a JSON without specifying a data dir, fall back to cwd — auto-
    # fix rules for whitespace / descriptions / extension casing still apply;
    # only Rule 1 (file existence) becomes best-effort. A real data dir can
    # be wired in later by reloading through load_from_file with data_root.
    pp_root = data_root if data_root is not None else Path.cwd()

    try:
        pp = _WrapperPreprocessor(data_root=pp_root, auto_fix=True)
        repaired, repair_log = pp.preprocess(raw)
        repair_log_records = _serialize_repair_log(repair_log)
        source = f"wrapper (data_root={pp_root})"
    except Exception as exc:
        # Preprocessing is a best-effort enhancement, not a hard requirement.
        # If it fails (e.g. unexpected path traversal in a synthetic dataset),
        # fall back to the raw dict so the prototype still works.
        _logger.warning(
            "ISA-PHM-Wrapper preprocessing failed (%s); falling back to raw dict.",
            exc,
        )
        repaired = raw
        repair_log_records = []
        source = f"wrapper-parse-only (preprocess skipped: {exc})"

    parser = ISAParser(repaired)
    parser.repair_log = repair_log_records
    parser.preprocessing_source = source
    return parser


def load_from_file(path: str) -> ISAParser:
    p = Path(path)
    raw_bytes = p.read_bytes()
    parser = _load_through_wrapper(
        json_text=raw_bytes.decode("utf-8"),
        source_hint=str(p),
        data_root=p.parent,
    )
    parser.set_source_bytes(raw_bytes, p.name)
    # If the JSON sits next to its data, auto-attach immediately.
    parser.attach_wrapper(str(p.parent))
    return parser


def load_from_bytes(content: bytes, source_hint: str = "<upload>") -> ISAParser:
    parser = _load_through_wrapper(
        json_text=content.decode("utf-8"),
        source_hint=source_hint,
        data_root=None,
    )
    parser.set_source_bytes(content, source_hint)
    return parser
