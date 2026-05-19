"""
Static phenotype mappings and resolver for annotation automation.

Provides loader and precedence-based resolver for mapping column headers to
Neurobagel standardized variables.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path, missing_prefix: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{missing_prefix}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_static_mappings(resource_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load built-in static phenotype mappings.

    Args:
        resource_path: Optional override path; defaults to bundled phenotype_mappings.json

    Returns:
        Dictionary of mappings with context and column definitions.
    """
    if resource_path is None:
        resource_path = (
            Path(__file__).parent.parent.parent
            / "resources"
            / "phenotype_mappings.json"
        )
    return _load_json(resource_path, "Phenotype mappings file not found")


def merge_mappings(
    builtin: Dict[str, Any], user_mappings: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Merge user-supplied mappings with built-in mappings.

    User mappings take precedence over built-in mappings.

    Args:
        builtin: Built-in mappings registry
        user_mappings: Optional user-supplied mappings (same schema as builtin)

    Returns:
        Merged mappings dictionary with user overrides applied.
    """
    merged = builtin.copy()

    if user_mappings:
        for section in ("@context", "mappings"):
            if section in user_mappings:
                merged.setdefault(section, {}).update(user_mappings[section])

    return merged


def load_user_mappings(path: str | Path) -> Dict[str, Any]:
    """
    Load user-supplied phenotype mappings from JSON file.

    Args:
        path: Path (str or Path object) to user mapping JSON file

    Returns:
        User mappings dictionary

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is invalid JSON
    """
    return _load_json(Path(path), "User mappings file not found")
