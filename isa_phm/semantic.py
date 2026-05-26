"""Semantic normalization utilities for ISA-PHM metadata."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any, Literal

from .errors import ValidationError
from .schemas import (
    AssayModel,
    InvestigationModel,
    RunRecord,
    SemanticDiagnostics,
    SemanticField,
    SemanticManifest,
    StudyModel,
)


_WORD_RE = re.compile(r"[^a-z0-9]+")

_FAULT_TYPE_RE = re.compile(r"\b(fault|damage|degradation|defect|rul)\b", re.IGNORECASE)


def _norm(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(_WORD_RE.sub(" ", str(text).strip().lower()).split())


@dataclass(frozen=True)
class _MappingResult:
    semantic_key: str
    status: Literal["mapped", "unknown", "ambiguous"]
    confidence: float
    provenance: str
    candidates: list[str]


class SemanticNormalizer:
    """
    Conservative semantic mapper for free-text factor/parameter names.

    Mapping precedence:
    1) User override config aliases (exact normalized match)
    2) Built-in aliases (exact)
    3) Built-in fuzzy match (high threshold, single unambiguous hit)
    4) unknown / ambiguous
    """

    _FACTOR_ALIASES: dict[str, set[str]] = {
        "fault_type": {"fault type", "fault"},
        "fault_position": {"fault position", "fault location", "fault component"},
        "fault_severity": {"fault severity", "severity", "fault size", "fault depth", "fault diameter"},
        "damage_vb": {"vb", "flank wear", "tool wear", "wear"},
        "rul": {"rul", "remaining useful life"},
        "bearing_lifetime": {"bearing lifetime", "lifetime"},
        "operating_speed": {
            "motor speed",
            "spindle speed",
            "rotational speed",
            "cutting speed",
            "speed rpm",
        },
        "load": {"load", "axial load", "radial load", "force"},
        "pressure": {"pressure", "discharge pressure", "discharge perssure", "pressure axial", "pressure radial"},
        "flow": {"flow", "flow rate"},
        "current": {"current", "motor current"},
        "voltage": {"voltage"},
        "temperature": {"temperature", "ambient temperature"},
        "feed_rate": {"feed", "feed rate"},
        "depth_of_cut": {"depth of cut"},
        "material": {"material"},
    }

    _PARAM_ALIASES: dict[str, set[str]] = {
        "sampling_frequency": {"sampling frequency", "sampling rate", "sample rate", "fs"},
        "sampling_period": {"sampling period", "sample period", "dt"},
        "sampling_points": {"sampling points", "number of samples", "record length"},
        "filter_cutoff_frequency": {"filter cutoff frequency", "cutoff frequency"},
        "fft_size": {"fft size"},
        "window_function": {"window function"},
        "analysis_method": {"analysis method"},
        "feature_type": {"feature type"},
        "resampling_rate": {"resampling rate"},
    }

    _OPERATING_KEYS: frozenset[str] = frozenset(
        {
            "operating_speed",
            "load",
            "pressure",
            "flow",
            "current",
            "voltage",
            "temperature",
            "feed_rate",
            "depth_of_cut",
            "material",
        }
    )
    _FAULT_KEYS: frozenset[str] = frozenset(
        {"fault_type", "fault_position", "fault_severity", "damage_vb", "rul", "bearing_lifetime"}
    )

    def __init__(self, override_config_path: str | Path | None = None) -> None:
        self._factor_index = self._build_index(self._FACTOR_ALIASES)
        self._param_index = self._build_index(self._PARAM_ALIASES)
        self._override_factor: dict[str, str] = {}
        self._override_param: dict[str, str] = {}
        self._override_path: Path | None = None
        if override_config_path is not None:
            self.load_override_config(override_config_path)

    @staticmethod
    def _build_index(alias_map: dict[str, set[str]]) -> dict[str, set[str]]:
        index: dict[str, set[str]] = {}
        for semantic_key, aliases in alias_map.items():
            for alias in aliases | {semantic_key}:
                n = _norm(alias)
                if not n:
                    continue
                index.setdefault(n, set()).add(semantic_key)
        return index

    def load_override_config(self, path: str | Path) -> None:
        """
        Load override aliases from JSON.

        Supported keys:
        - factor / factors: {"alias": "semantic_key"}
        - parameter / parameters: {"alias": "semantic_key"}
        """
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        factor_map = raw.get("factor") or raw.get("factors") or {}
        param_map = raw.get("parameter") or raw.get("parameters") or {}

        self._override_factor = {
            _norm(alias): str(semantic_key)
            for alias, semantic_key in factor_map.items()
            if _norm(alias)
        }
        self._override_param = {
            _norm(alias): str(semantic_key)
            for alias, semantic_key in param_map.items()
            if _norm(alias)
        }
        self._override_path = p

    @property
    def override_path(self) -> Path | None:
        return self._override_path

    def normalize_factor(
        self,
        factor_name: str,
        factor_type: str | None = None,
        unit: str | None = None,
    ) -> SemanticField:
        factor_type_n = _norm(factor_type)
        allowed: set[str] | None = None
        if "operating condition" in factor_type_n:
            allowed = set(self._OPERATING_KEYS)
        elif _FAULT_TYPE_RE.search(factor_type_n):
            allowed = set(self._FAULT_KEYS)

        mapped = self._map_name(
            source_name=factor_name,
            source_kind="factor",
            override_map=self._override_factor,
            alias_index=self._factor_index,
            allowed_keys=allowed,
        )
        return SemanticField(
            source_name=factor_name,
            source_kind="factor",
            semantic_key=mapped.semantic_key,
            status=mapped.status,
            confidence=mapped.confidence,
            provenance=mapped.provenance,
            candidates=mapped.candidates,
            context={"factor_type": factor_type or "", "unit": unit or ""},
        )

    def normalize_parameter(
        self,
        parameter_name: str,
        source_kind: Literal["measurement_parameter", "processing_parameter"],
        unit: str | None = None,
    ) -> SemanticField:
        mapped = self._map_name(
            source_name=parameter_name,
            source_kind=source_kind,
            override_map=self._override_param,
            alias_index=self._param_index,
            allowed_keys=None,
        )
        return SemanticField(
            source_name=parameter_name,
            source_kind=source_kind,
            semantic_key=mapped.semantic_key,
            status=mapped.status,
            confidence=mapped.confidence,
            provenance=mapped.provenance,
            candidates=mapped.candidates,
            context={"unit": unit or ""},
        )

    def normalize_study_factors(self, study: StudyModel) -> list[SemanticField]:
        fields: list[SemanticField] = []
        for f in study.factors:
            fields.append(
                self.normalize_factor(
                    factor_name=f.factor_name,
                    factor_type=f.factor_type,
                    unit=f.unit,
                )
            )
        return fields

    def normalize_assay_parameters(
        self,
        assay: AssayModel,
        run: RunRecord | None = None,
    ) -> dict[str, list[SemanticField]]:
        target_run = run
        if target_run is None and assay.runs:
            target_run = assay.runs[0]

        if target_run is None:
            return {"measurement": [], "processing": []}

        measurement = [
            self.normalize_parameter(
                parameter_name=pv.parameter_name,
                source_kind="measurement_parameter",
                unit=pv.unit,
            )
            for pv in target_run.measurement_params
        ]
        processing = [
            self.normalize_parameter(
                parameter_name=pv.parameter_name,
                source_kind="processing_parameter",
                unit=pv.unit,
            )
            for pv in target_run.processing_params
        ]
        return {"measurement": measurement, "processing": processing}

    def build_manifest(self, investigation: InvestigationModel) -> SemanticManifest:
        return self.build_manifest_with_controls(investigation)

    def build_manifest_with_controls(
        self,
        investigation: InvestigationModel,
        *,
        strict: bool = False,
        max_unknown_ratio: float = 0.05,
        max_ambiguous_ratio: float = 0.0,
        require_override_config: bool = False,
    ) -> SemanticManifest:
        manifest = SemanticManifest(investigation_id=investigation.identifier)

        for study in investigation.studies:
            study_fields = self.normalize_study_factors(study)
            manifest.study_factors[study.study_id] = study_fields

            for assay in study.assays:
                key = f"{study.study_id}:{assay.assay_id}"
                normalized = self.normalize_assay_parameters(assay)
                manifest.assay_measurement_params[key] = normalized["measurement"]
                manifest.assay_processing_params[key] = normalized["processing"]

        manifest.diagnostics = self._build_diagnostics(manifest)
        violations = self.evaluate_strict_violations(
            manifest,
            max_unknown_ratio=max_unknown_ratio,
            max_ambiguous_ratio=max_ambiguous_ratio,
            require_override_config=require_override_config,
        )
        manifest.diagnostics.strict_violations = violations
        if strict and violations:
            raise ValidationError(
                "Semantic strict validation failed "
                f"(codes={violations}, "
                f"unknown_ratio={manifest.diagnostics.unknown_ratio:.4f}, "
                f"ambiguous_ratio={manifest.diagnostics.ambiguous_ratio:.4f}, "
                f"missing_override_fields={manifest.diagnostics.missing_override_fields})."
            )
        return manifest

    def evaluate_strict_violations(
        self,
        manifest: SemanticManifest,
        *,
        max_unknown_ratio: float = 0.05,
        max_ambiguous_ratio: float = 0.0,
        require_override_config: bool = False,
    ) -> list[str]:
        diagnostics = manifest.diagnostics
        violations: list[str] = []

        if require_override_config and self._override_path is None:
            violations.append("SEM_OVERRIDE_CONFIG_REQUIRED")

        if diagnostics.unknown_ratio > max_unknown_ratio:
            violations.append("SEM_UNKNOWN_RATIO_EXCEEDED")

        if diagnostics.ambiguous_ratio > max_ambiguous_ratio:
            violations.append("SEM_AMBIGUOUS_RATIO_EXCEEDED")

        if require_override_config and diagnostics.missing_override_fields:
            violations.append("SEM_MISSING_OVERRIDE_FIELDS")

        return violations

    def _build_diagnostics(self, manifest: SemanticManifest) -> SemanticDiagnostics:
        fields: list[SemanticField] = []
        for lst in manifest.study_factors.values():
            fields.extend(lst)
        for lst in manifest.assay_measurement_params.values():
            fields.extend(lst)
        for lst in manifest.assay_processing_params.values():
            fields.extend(lst)

        total = len(fields)
        mapped = sum(1 for f in fields if f.status == "mapped")
        unknown = sum(1 for f in fields if f.status == "unknown")
        ambiguous = sum(1 for f in fields if f.status == "ambiguous")
        unknown_ratio = (unknown / total) if total else 0.0
        ambiguous_ratio = (ambiguous / total) if total else 0.0
        missing_override_fields = sorted(
            {
                f.source_name
                for f in fields
                if f.status in {"unknown", "ambiguous"} and f.source_name
            }
        )

        return SemanticDiagnostics(
            total_fields=total,
            mapped_fields=mapped,
            unknown_fields=unknown,
            ambiguous_fields=ambiguous,
            unknown_ratio=round(float(unknown_ratio), 6),
            ambiguous_ratio=round(float(ambiguous_ratio), 6),
            missing_override_fields=missing_override_fields,
            strict_violations=[],
        )

    def _map_name(
        self,
        source_name: str,
        source_kind: Literal["factor", "measurement_parameter", "processing_parameter"],
        override_map: dict[str, str],
        alias_index: dict[str, set[str]],
        allowed_keys: set[str] | None,
    ) -> _MappingResult:
        source_norm = _norm(source_name)
        if not source_norm:
            return _MappingResult(
                semantic_key="unknown",
                status="unknown",
                confidence=0.0,
                provenance="empty_source",
                candidates=[],
            )

        override_hit = override_map.get(source_norm)
        if override_hit:
            return _MappingResult(
                semantic_key=override_hit,
                status="mapped",
                confidence=1.0,
                provenance="override_exact",
                candidates=[override_hit],
            )

        exact_candidates = alias_index.get(source_norm, set())
        if allowed_keys is not None:
            exact_candidates = {c for c in exact_candidates if c in allowed_keys}

        if len(exact_candidates) == 1:
            candidate = next(iter(exact_candidates))
            return _MappingResult(
                semantic_key=candidate,
                status="mapped",
                confidence=0.99,
                provenance="built_in_exact",
                candidates=[candidate],
            )
        if len(exact_candidates) > 1:
            cands = sorted(exact_candidates)
            return _MappingResult(
                semantic_key="unknown",
                status="ambiguous",
                confidence=0.0,
                provenance="built_in_ambiguous_exact",
                candidates=cands,
            )

        # Conservative fuzzy matching: require exactly one close alias and one semantic candidate.
        allowed_aliases = []
        for alias, candidate_keys in alias_index.items():
            if allowed_keys is None or any(c in allowed_keys for c in candidate_keys):
                allowed_aliases.append(alias)

        fuzzy = get_close_matches(source_norm, allowed_aliases, n=3, cutoff=0.93)
        if not fuzzy:
            return _MappingResult(
                semantic_key="unknown",
                status="unknown",
                confidence=0.0,
                provenance="no_match",
                candidates=[],
            )

        candidate_keys: set[str] = set()
        for alias in fuzzy:
            keys = alias_index.get(alias, set())
            if allowed_keys is not None:
                keys = {k for k in keys if k in allowed_keys}
            candidate_keys.update(keys)

        if len(candidate_keys) != 1 or len(fuzzy) != 1:
            return _MappingResult(
                semantic_key="unknown",
                status="ambiguous",
                confidence=0.0,
                provenance="built_in_ambiguous_fuzzy",
                candidates=sorted(candidate_keys),
            )

        candidate = next(iter(candidate_keys))
        confidence = SequenceMatcher(a=source_norm, b=fuzzy[0]).ratio()
        return _MappingResult(
            semantic_key=candidate,
            status="mapped",
            confidence=round(float(confidence), 4),
            provenance="built_in_fuzzy",
            candidates=[candidate],
        )
