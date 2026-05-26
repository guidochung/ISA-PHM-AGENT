"""
ISAParser — load and validate ISA-JSON files.

Validation order
----------------
1. Read file as text (raises ParseError on IO/encoding failure).
2. Decode JSON (raises ParseError on malformed JSON).
3. Validate structure with isatools (raises ValidationError on schema errors).
   If isatools is not installed and strict=True, raises ParseError immediately.
4. Return the raw Python dict for downstream processing.
"""

from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path

from .errors import ParseError, ValidationError

logger = logging.getLogger("isa_phm")
_MIN_TOP_LEVEL_KEYS = (
    "title",
    "description",
    "identifier",
    "studies",
    "ontologySourceReferences",
)

# Optional isatools import — required in strict mode.
try:
    from isatools import isajson as _isajson  # type: ignore[import]

    _ISATOOLS_AVAILABLE = True
except ImportError:
    _ISATOOLS_AVAILABLE = False


class ISAParser:
    """
    Load an ISA-JSON file into a raw Python dict and optionally validate it
    against the ISA-JSON schema using the isatools library.

    Parameters
    ----------
    strict : bool
        When True (default), isatools validation is run. If isatools is not
        installed, a ParseError is raised immediately.
        When False, validation is skipped (useful for quick exploration).
    """

    def __init__(self, strict: bool = True) -> None:
        self._strict = strict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: Path | str) -> dict:
        """
        Load an ISA-JSON file from disk.

        Parameters
        ----------
        path : Path | str
            Absolute or relative path to the ISA-JSON file.

        Returns
        -------
        dict
            Validated raw Python dict.

        Raises
        ------
        ParseError
            File not found, encoding failure, or malformed JSON.
        ValidationError
            ISA-JSON schema violation detected by isatools.
        """
        path = Path(path)
        logger.info("Loading ISA-JSON from '%s'.", path)

        try:
            json_str = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ParseError(f"ISA-JSON file not found: '{path}'.")
        except PermissionError:
            raise ParseError(f"Permission denied reading: '{path}'.")
        except UnicodeDecodeError:
            # Try Latin-1 fallback
            try:
                json_str = path.read_text(encoding="latin-1")
                logger.warning("File '%s' read with Latin-1 encoding fallback.", path)
            except Exception as exc:
                raise ParseError(
                    f"Cannot decode '{path}'. Tried UTF-8 and Latin-1. Error: {exc}"
                ) from exc

        return self.load_string(json_str, source_hint=str(path))

    def load_string(self, json_str: str, source_hint: str = "<string>") -> dict:
        """
        Parse and optionally validate an ISA-JSON string.

        Parameters
        ----------
        json_str : str
            Raw ISA-JSON content.
        source_hint : str
            Used in error messages to identify the source.

        Returns
        -------
        dict
        """
        # Step 1: JSON decode
        try:
            raw = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ParseError(
                f"Invalid JSON in '{source_hint}' at line {exc.lineno}, "
                f"column {exc.colno}: {exc.msg}."
            ) from exc

        if not isinstance(raw, dict):
            raise ParseError(
                f"ISA-JSON root must be a JSON object (dict), got {type(raw).__name__}."
            )
        self._validate_minimum_keys(raw, source_hint)

        # Step 2: isatools validation
        if self._strict:
            self._validate(json_str, source_hint)

        logger.debug("Parsed ISA-JSON from '%s' — %d top-level keys.", source_hint, len(raw))
        return raw

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def _validate(self, json_str: str, source_hint: str) -> None:
        """Run isatools schema validation. Raises ValidationError on failure."""
        if not _ISATOOLS_AVAILABLE:
            raise ParseError(
                "isatools is not installed but strict=True. "
                "Install it with: pip install isatools  "
                "Or use ISAParser(strict=False) to skip validation."
            )

        try:
            _isajson.load(StringIO(json_str))
            logger.debug("isatools validation passed for '%s'.", source_hint)
        except Exception as exc:
            # isatools raises a mix of exception types depending on version.
            raise ValidationError(
                f"ISA-JSON schema validation failed for '{source_hint}': {exc}"
            ) from exc

    def _validate_minimum_keys(self, raw: dict, source_hint: str) -> None:
        """Fail fast on obviously incomplete ISA-JSON root objects."""
        missing = [k for k in _MIN_TOP_LEVEL_KEYS if k not in raw]
        if missing:
            raise ParseError(
                f"ISA-JSON '{source_hint}' is missing required top-level keys: {missing}."
            )
