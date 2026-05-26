"""
ISAPreprocessor — apply safe, logged repairs to a raw ISA-JSON dict before
metadata extraction.

Exactly five auto-fix rules (MVP scope, per wrapper.md):
  1. Resolve relative dataFile paths against data_root.
  2. Normalize file name extensions to lowercase.
  3. Fill dataFile.name from dataFile.filename when name is empty.
  4. Strip leading/trailing whitespace from all @id strings in the tree.
  5. Replace None/null description fields with empty string "".

Fatal conditions (raise PreprocessingError):
  - Investigation has zero studies.
  - A study has zero assays.
  - A resolved data file path escapes outside data_root (path traversal).

Windows absolute paths (e.g. D:\\DPL\\...) are handled cross-platform:
  - If the exact path exists on the current OS → keep as-is.
  - Otherwise attempt to find just the filename under data_root.
  - All resolutions are logged.
"""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .errors import PreprocessingError
from .schemas import RepairAction, RepairLog

logger = logging.getLogger("isa_phm")

_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[/\\]|^\\\\")


class ISAPreprocessor:
    """
    Apply safe repairs to a raw ISA-JSON dict.

    Parameters
    ----------
    data_root : Path
        Base directory for resolving relative data file paths.
    auto_fix : bool
        When True (default) apply all five repair rules.
        When False, any fixable issue raises PreprocessingError immediately.
    """

    def __init__(self, data_root: Path, auto_fix: bool = True) -> None:
        self._data_root = Path(data_root).resolve()
        self._auto_fix = auto_fix

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def preprocess(self, raw: dict) -> tuple[dict, RepairLog]:
        """
        Preprocess a raw ISA-JSON dict.

        Returns
        -------
        (repaired_dict, RepairLog)
            A deep copy with repairs applied and the repair log.

        Raises
        ------
        PreprocessingError
            On unfixable structural issues or when auto_fix=False.
        """
        repaired = copy.deepcopy(raw)
        log = RepairLog()

        # Rule 4 first: clean @id strings before we use them for anything.
        self._strip_id_whitespace(repaired, log)

        # Rule 5: fill null descriptions.
        self._fill_null_descriptions(repaired, log)

        # Fatal: must have at least one study.
        studies = repaired.get("studies", [])
        if not studies:
            raise PreprocessingError(
                "ISA-JSON contains zero studies. "
                "A valid ISA-PHM dataset must have at least one study."
            )

        for study in studies:
            assays = study.get("assays", [])
            if not assays:
                title = study.get("title", study.get("identifier", "?"))
                raise PreprocessingError(
                    f"Study '{title}' has zero assays. "
                    f"Each study must contain at least one assay (sensor channel)."
                )

            # Rules 1, 2, 3: fix data file paths in assays.
            for assay in assays:
                for df_entry in assay.get("dataFiles", []):
                    assay_id = assay.get("filename", "?")
                    self._fix_data_file(df_entry, assay_id, log)

        logger.info(
            "Preprocessing complete — %d repair actions applied.", len(log)
        )
        return repaired, log

    # ------------------------------------------------------------------
    # Rule 1, 2, 3: data file path resolution
    # ------------------------------------------------------------------

    def _fix_data_file(
        self, df_entry: dict, assay_id: str, log: RepairLog
    ) -> None:
        entity_id = df_entry.get("@id", "?")
        original_name = df_entry.get("name", "")

        # Rule 3: fill name from filename if name is empty.
        if not original_name and df_entry.get("filename"):
            new_name = df_entry["filename"]
            if self._auto_fix:
                df_entry["name"] = new_name
                log.add(RepairAction(
                    severity="INFO",
                    entity_type="DataFile",
                    entity_id=entity_id,
                    field_name="name",
                    old_value="",
                    new_value=new_name,
                    message="Filled name from filename field.",
                ))
            else:
                raise PreprocessingError(
                    f"DataFile '{entity_id}' in assay '{assay_id}' has no name. "
                    f"Enable auto_fix=True to fill from filename."
                )
            return  # Will be processed again next call if needed.

        if not original_name:
            # Raw files often have no name — that is valid. Just warn.
            log.add(RepairAction(
                severity="WARNING",
                entity_type="DataFile",
                entity_id=entity_id,
                field_name="name",
                old_value="",
                new_value="",
                message=(
                    f"DataFile in assay '{assay_id}' has no path/name. "
                    f"Loading this file will raise DataFileError."
                ),
            ))
            return

        # Rule 2: normalize extension to lowercase.
        fixed_name = self._fix_extension(original_name)
        if fixed_name != original_name:
            if self._auto_fix:
                df_entry["name"] = fixed_name
                log.add(RepairAction(
                    severity="INFO",
                    entity_type="DataFile",
                    entity_id=entity_id,
                    field_name="name",
                    old_value=original_name,
                    new_value=fixed_name,
                    message="File extension normalized to lowercase.",
                ))
                original_name = fixed_name
            else:
                raise PreprocessingError(
                    f"DataFile '{entity_id}' has uppercase extension '{original_name}'. "
                    f"Enable auto_fix=True to normalize."
                )

        # Rule 1: resolve path against data_root.
        resolved, was_repaired = self._resolve_path(original_name, entity_id, assay_id)
        if was_repaired and self._auto_fix:
            df_entry["name"] = resolved
            log.add(RepairAction(
                severity="INFO",
                entity_type="DataFile",
                entity_id=entity_id,
                field_name="name",
                old_value=original_name,
                new_value=resolved,
                message="Relative/Windows path resolved against data_root.",
            ))
        elif was_repaired and not self._auto_fix:
            raise PreprocessingError(
                f"DataFile '{entity_id}' path '{original_name}' requires resolution. "
                f"Enable auto_fix=True."
            )

    def _fix_extension(self, name: str) -> str:
        """Lower-case the file extension portion of a path string."""
        p = PureWindowsPath(name) if _WINDOWS_ABS_RE.match(name) else PurePosixPath(name)
        suffix = p.suffix
        if suffix and suffix != suffix.lower():
            return name[: -len(suffix)] + suffix.lower()
        return name

    def _resolve_path(
        self, name: str, entity_id: str, assay_id: str
    ) -> tuple[str, bool]:
        """
        Attempt to resolve a file path to an absolute path under data_root.

        Returns (resolved_str, was_repaired).
        Raises PreprocessingError on path traversal.
        """
        is_windows = bool(_WINDOWS_ABS_RE.match(name))

        if is_windows:
            # Windows absolute path — check if it exists on the current OS.
            win_path = Path(name)
            if win_path.exists():
                # Enforce traversal policy against the resolved on-disk target.
                self._check_traversal(win_path.resolve(strict=False), entity_id)
                return str(win_path), False  # Already correct on this machine.

            # Fall back: find just the filename under data_root.
            filename = PureWindowsPath(name).name
            candidate = self._data_root / filename
            if candidate.exists():
                # Harden fallback: validate the resolved target (symlinks included).
                self._check_traversal(candidate.resolve(strict=False), entity_id)
                logger.warning(
                    "DataFile '%s' in assay '%s': Windows path '%s' was resolved "
                    "via filename fallback to '%s'.",
                    entity_id,
                    assay_id,
                    name,
                    candidate,
                )
                return str(candidate), True

            # Cannot resolve, but don't fail — DataFileError at load time.
            logger.warning(
                "DataFile '%s' in assay '%s': Windows path '%s' not found "
                "on this system and no match under data_root. "
                "Loading this file will raise DataFileError.",
                entity_id, assay_id, name,
            )
            return name, False

        # Relative or POSIX absolute path.
        raw_path = Path(name)
        if raw_path.is_absolute():
            if raw_path.exists():
                self._check_traversal(raw_path.resolve(strict=False), entity_id)
                return str(raw_path), False
            # Absolute but doesn't exist — leave as-is.
            return str(raw_path), False

        # Relative path — join with data_root.
        resolved = (self._data_root / raw_path).resolve(strict=False)
        self._check_traversal(resolved, entity_id)
        return str(resolved), str(resolved) != str(raw_path.resolve(strict=False))

    def _check_traversal(self, resolved: Path, entity_id: str) -> None:
        """Raise PreprocessingError if resolved path escapes data_root."""
        try:
            resolved.relative_to(self._data_root)
        except ValueError:
            raise PreprocessingError(
                f"Path traversal detected for DataFile '{entity_id}': "
                f"resolved path '{resolved}' is outside data_root '{self._data_root}'. "
                f"This file reference will not be loaded."
            )

    # ------------------------------------------------------------------
    # Rule 4: strip @id whitespace (tree walk)
    # ------------------------------------------------------------------

    def _strip_id_whitespace(self, node: Any, log: RepairLog) -> None:
        if isinstance(node, dict):
            if "@id" in node:
                raw_id = node["@id"]
                cleaned = raw_id.strip() if isinstance(raw_id, str) else raw_id
                if cleaned != raw_id:
                    node["@id"] = cleaned
                    log.add(RepairAction(
                        severity="INFO",
                        entity_type="@id",
                        entity_id=cleaned,
                        field_name="@id",
                        old_value=raw_id,
                        new_value=cleaned,
                        message="Stripped whitespace from @id value.",
                    ))
            for value in node.values():
                self._strip_id_whitespace(value, log)
        elif isinstance(node, list):
            for item in node:
                self._strip_id_whitespace(item, log)

    # ------------------------------------------------------------------
    # Rule 5: fill null descriptions
    # ------------------------------------------------------------------

    def _fill_null_descriptions(self, node: Any, log: RepairLog) -> None:
        if isinstance(node, dict):
            if "description" in node and node["description"] is None:
                node["description"] = ""
                log.add(RepairAction(
                    severity="INFO",
                    entity_type="object",
                    entity_id=node.get("@id", node.get("identifier", "?")),
                    field_name="description",
                    old_value=None,
                    new_value="",
                    message="Replaced null description with empty string.",
                ))
            for value in node.values():
                self._fill_null_descriptions(value, log)
        elif isinstance(node, list):
            for item in node:
                self._fill_null_descriptions(item, log)
