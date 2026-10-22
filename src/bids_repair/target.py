"""Exact-version schema acquisition and offline checks for the native validator."""

import json
import os
import re
import tempfile
from contextlib import contextmanager
from importlib.resources import as_file
from pathlib import Path
from typing import Any, Iterator

import httpx
from bidsschematools import data
from bidsschematools.rules import regexify_all, regexify_filename_rules
from bidsschematools.schema import load_schema
from bidsschematools.types import Namespace
from bidsschematools.validator import validate_all

from bids_repair.paths import assert_safe_path

SCHEMA_URL = "https://bids-specification.readthedocs.io/en/v{version}/schema.json"


def _check_schema_version(schema: Namespace, version: str) -> None:
    if not isinstance(schema.get("bids_version"), str):
        raise ValueError("Local BIDS schema must declare its bids_version")
    if schema.bids_version != version:
        raise ValueError(
            f"Schema targets BIDS {schema.bids_version}, not requested BIDS {version}."
        )


def _check_schema(path: Path, version: str) -> Namespace:
    schema = _read_schema_json(path) if path.is_file() else load_schema(str(path))
    _check_schema_version(schema, version)
    return schema


def _check_filename_rules(path: Path, schema: Namespace) -> None:
    try:
        rules = [
            rule
            for group in (schema.rules.files.common, schema.rules.files.raw)
            for rule in regexify_filename_rules(group, schema, level=2)
        ]
        if not rules:
            raise ValueError("No filename rules found")
        for rule in rules:
            re.compile(rule["regex"])
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        re.error,
    ) as exc:
        raise ValueError(
            f"Schema at {path} is incompatible with the installed "
            f"bidsschematools filename checks: {exc}"
        ) from exc


def _read_schema_json(path: Path) -> Namespace:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Schema JSON must contain an object")
    return Namespace.build(document)


def _check_downloaded_schema(path: Path, version: str) -> None:
    schema = _read_schema_json(path)
    format_version = schema.get("schema_version")
    if not isinstance(format_version, str) or not format_version.strip():
        raise ValueError("Schema must declare a non-empty schema_version")
    _check_schema_version(schema, version)
    _check_filename_rules(path, schema)


def acquire_target_schema(version: str, schema_path: Path | None = None) -> Path | None:
    """Resolve a CLI schema locally first, downloading only the exact release."""
    if schema_path is not None:
        _check_filename_rules(schema_path, _check_schema(schema_path, version))
        return schema_path

    with as_file(data.load.readable("schema.json")) as bundled:
        schema = load_schema(str(bundled))
        if schema.bids_version == version:
            _check_filename_rules(bundled, schema)
            return None

    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None:
        raise ValueError(
            f"Cannot fetch a schema for invalid BIDS release {version!r}; "
            "supply --bids-version MAJOR.MINOR.PATCH or --schema PATH."
        )

    cache_home = os.environ.get("XDG_CACHE_HOME")
    cache = (
        (Path(cache_home) if cache_home else Path.home() / ".cache")
        / "bids-repair"
        / "schemas"
        / version
        / "schema.json"
    )
    if cache.exists() or cache.is_symlink():
        try:
            _check_downloaded_schema(cache, version)
        except (OSError, UnicodeError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot use cached BIDS schema {cache}: {exc}. "
                "Remove the invalid cache entry or supply --schema PATH."
            ) from exc
        return cache

    url = SCHEMA_URL.format(version=version)
    try:
        response = httpx.get(url, timeout=30, follow_redirects=False)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = (
            "No published schema is available for this release"
            if exc.response.status_code == 404
            else f"HTTP {exc.response.status_code}"
        )
        raise RuntimeError(
            f"Cannot fetch BIDS {version} schema from {url}: {detail}. "
            "Supply --schema PATH to a matching local official schema."
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Cannot fetch BIDS {version} schema from {url}: {exc}. "
            "Check network access or supply --schema PATH."
        ) from exc

    temporary: Path | None = None
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=cache.parent, suffix=".json", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(response.content)
        _check_downloaded_schema(temporary, version)
        os.replace(temporary, cache)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot validate or cache BIDS {version} schema at {cache}: {exc}. "
            "Supply --schema PATH to a matching compatible local official schema."
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return cache


@contextmanager
def target_schema(
    version: str, schema_path: str | Path | None = None
) -> Iterator[Path]:
    """Never use bidsschematools' network/version-fallback schema selection."""
    if schema_path is not None:
        path = Path(schema_path)
        _check_schema(path, version)
        yield path
        return

    with as_file(data.load.readable("schema.json")) as path:
        schema = load_schema(str(path))
        if schema.bids_version != version:
            raise ValueError(
                f"Unsupported target BIDS version {version}; the bundled schema is "
                f"{schema.bids_version}. Supply --schema PATH to a matching local "
                "BIDS schema JSON file or schema directory."
            )
        yield path


def add_target_findings(
    root: Path,
    report: dict[str, Any],
    version: str,
    schema_path: str | Path | None = None,
) -> None:
    with target_schema(version, schema_path) as schema:
        regex_schema, _ = regexify_all(str(schema))
        for dataset_report in report["datasets"]:
            owner = Path(dataset_report.get("dataset", root))
            if not owner.is_absolute():
                owner = root / owner
            owner = owner.absolute()
            assert_safe_path(root, owner, must_exist=True)

            paths = []
            for path in sorted(owner.rglob("*")):
                relative = path.relative_to(owner)
                if any(part.startswith(".") for part in relative.parts):
                    continue
                if relative.parts[0] in {
                    "derivatives",
                    "sourcedata",
                    "code",
                    "stimuli",
                    "models",
                }:
                    continue
                assert_safe_path(root, path, must_exist=True)
                if path.is_file():
                    paths.append(str(path))

            result = validate_all(paths, regex_schema)
            issues = dataset_report.setdefault("issues", [])
            for path in sorted(result["path_tracking"]):
                issues.append(
                    {
                        "code": "INVALID_FILENAME",
                        "severity": "warning",
                        "path": path,
                        "message": f"Path does not match the BIDS {version} filename schema.",
                    }
                )
                dataset_report["warnings"] = dataset_report.get("warnings", 0) + 1

    report["target_bids_version"] = version
    report["coverage"] = (
        "Native structural validation plus explicit-version schema filename checks; "
        "not full sidecar or NIfTI semantic validation."
    )
