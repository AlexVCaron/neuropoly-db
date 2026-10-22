"""Offline exact-version acquisition and standalone CLI version selection."""

import json
from unittest.mock import patch

import httpx
import pytest
from bidsschematools.schema import load_schema
from typer.testing import CliRunner

from bids_repair.cli import app
from bids_repair.target import SCHEMA_URL, acquire_target_schema, target_schema

VERSION = load_schema().bids_version
REMOTE_VERSION = "1.9.0"
runner = CliRunner()


@pytest.fixture
def schema_document():
    # Representative pre-1.0 schema structure, with string entity requirements.
    return {
        "bids_version": REMOTE_VERSION,
        "schema_version": "0.7.3-dev",
        "objects": {
            "entities": {"subject": {"name": "sub", "format": "label"}},
            "formats": {"label": {"pattern": "[a-zA-Z0-9]+"}},
        },
        "rules": {
            "entities": ["subject"],
            "files": {
                "common": {
                    "core": {
                        "description": {
                            "path": "dataset_description.json",
                            "level": "required",
                        }
                    }
                },
                "raw": {
                    "anat": {
                        "image": {
                            "suffixes": ["T1w"],
                            "extensions": [".nii.gz"],
                            "datatypes": ["anat"],
                            "entities": {"subject": "required"},
                        }
                    }
                },
            },
        },
    }


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return (
        tmp_path / "cache" / "bids-repair" / "schemas" / REMOTE_VERSION / "schema.json"
    )


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "Fixture",
                "BIDSVersion": VERSION,
            }
        )
    )
    return root


def response(document=None, status=200, content=None):
    return httpx.Response(
        status,
        json=document,
        content=content,
        request=httpx.Request("GET", SCHEMA_URL.format(version=REMOTE_VERSION)),
    )


def test_exact_schema_download_is_validated_cached_and_reused(cache, schema_document):
    with patch(
        "bids_repair.target.httpx.get", return_value=response(schema_document)
    ) as get:
        assert acquire_target_schema(REMOTE_VERSION) == cache
        assert acquire_target_schema(REMOTE_VERSION) == cache

    get.assert_called_once_with(
        SCHEMA_URL.format(version=REMOTE_VERSION),
        timeout=30,
        follow_redirects=False,
    )
    assert json.loads(cache.read_text()) == schema_document
    assert list(cache.parent.iterdir()) == [cache]
    with target_schema(REMOTE_VERSION, cache) as selected:
        assert selected == cache


def test_override_and_bundle_never_fetch(cache, schema_document, tmp_path):
    local = tmp_path / "local.json"
    local.write_text(json.dumps(schema_document))

    with patch("bids_repair.target.httpx.get") as get:
        assert acquire_target_schema(REMOTE_VERSION, local) == local
        assert acquire_target_schema(VERSION) is None
        with pytest.raises(ValueError, match="not requested"):
            acquire_target_schema("0.0.0", local)

    get.assert_not_called()
    assert not cache.exists()


def test_local_schema_directory_override(cache, tmp_path):
    directory = tmp_path / "schema"
    directory.mkdir()
    (directory / "BIDS_VERSION").write_text(REMOTE_VERSION)
    (directory / "SCHEMA_VERSION").write_text("0.7.3-dev")
    (directory / "objects.yaml").write_text("placeholder: true\n")
    (directory / "rules.yaml").write_text(
        "files:\n  common:\n    core:\n      description:\n"
        "        path: dataset_description.json\n        level: required\n"
        "  raw: {}\n"
    )

    with patch("bids_repair.target.httpx.get") as get:
        assert acquire_target_schema(REMOTE_VERSION, directory) == directory

    get.assert_not_called()


@pytest.mark.parametrize(
    "change, message",
    [
        ({"bids_version": "1.8.0"}, "not requested"),
        ({"bids_version": None}, "declare its bids_version"),
        ({"schema_version": None}, "schema_version"),
        ({"schema_version": ""}, "schema_version"),
        ({"rules": {}}, "incompatible"),
    ],
)
def test_invalid_download_never_publishes(cache, schema_document, change, message):
    schema_document.update(change)

    with patch("bids_repair.target.httpx.get", return_value=response(schema_document)):
        with pytest.raises(RuntimeError, match=message):
            acquire_target_schema(REMOTE_VERSION)

    assert not cache.exists()
    assert not list(cache.parent.iterdir())


@pytest.mark.parametrize("content", [b"not json", b"[]", b'"\xff"'])
def test_malformed_download_never_publishes(cache, content):
    with patch("bids_repair.target.httpx.get", return_value=response(content=content)):
        with pytest.raises(RuntimeError, match="--schema PATH"):
            acquire_target_schema(REMOTE_VERSION)

    assert not cache.exists()
    assert not list(cache.parent.iterdir())


@pytest.mark.parametrize("status", [404, 302, 500])
def test_http_errors_are_actionable(cache, status):
    with patch("bids_repair.target.httpx.get", return_value=response(status=status)):
        with pytest.raises(RuntimeError, match="--schema PATH"):
            acquire_target_schema(REMOTE_VERSION)

    assert not cache.exists()


@pytest.mark.parametrize(
    "error", [httpx.ConnectError("offline"), httpx.ReadTimeout("timed out")]
)
def test_network_errors_are_actionable(cache, error):
    with patch("bids_repair.target.httpx.get", side_effect=error):
        with pytest.raises(RuntimeError, match="Check network access"):
            acquire_target_schema(REMOTE_VERSION)

    assert not cache.exists()


def test_bad_cache_is_reported_without_network_fallback(cache):
    cache.parent.mkdir(parents=True)
    cache.write_text("broken")

    with patch("bids_repair.target.httpx.get") as get:
        with pytest.raises(RuntimeError, match="Remove the invalid cache entry"):
            acquire_target_schema(REMOTE_VERSION)

    get.assert_not_called()
    assert cache.read_text() == "broken"


@pytest.mark.parametrize(
    "change",
    [
        {"bids_version": "1.8.0"},
        {"rules": {}},
    ],
)
def test_cache_is_rechecked_after_a_previous_success(cache, schema_document, change):
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(schema_document))
    assert acquire_target_schema(REMOTE_VERSION) == cache
    schema_document.update(change)
    cache.write_text(json.dumps(schema_document))
    with patch("bids_repair.target.httpx.get") as get:
        with pytest.raises(RuntimeError, match="cached BIDS schema"):
            acquire_target_schema(REMOTE_VERSION)
    get.assert_not_called()


def test_cache_write_failure_cleans_temporary_file(cache, schema_document):
    with (
        patch("bids_repair.target.httpx.get", return_value=response(schema_document)),
        patch(
            "bids_repair.target.os.replace", side_effect=PermissionError("read-only")
        ),
    ):
        with pytest.raises(RuntimeError, match="read-only"):
            acquire_target_schema(REMOTE_VERSION)
    assert not list(cache.parent.iterdir())


def test_cache_directory_failure_is_actionable(cache, schema_document):
    cache.parent.parent.mkdir(parents=True)
    cache.parent.write_text("not a directory")
    with patch("bids_repair.target.httpx.get", return_value=response(schema_document)):
        with pytest.raises(RuntimeError, match="--schema PATH"):
            acquire_target_schema(REMOTE_VERSION)


def test_invalid_regex_is_rejected_before_caching(cache, schema_document):
    schema_document["objects"]["formats"]["label"]["pattern"] = "["
    with patch("bids_repair.target.httpx.get", return_value=response(schema_document)):
        with pytest.raises(RuntimeError, match="incompatible"):
            acquire_target_schema(REMOTE_VERSION)
    assert not cache.exists()


@pytest.mark.parametrize(
    "version", ["../1.9.0", "https://example.com", "latest", "1.9"]
)
def test_invalid_remote_version_does_not_fetch(cache, version):
    with patch("bids_repair.target.httpx.get") as get:
        with pytest.raises(ValueError, match="invalid BIDS release"):
            acquire_target_schema(version)
    get.assert_not_called()


def test_library_schema_selection_stays_offline(cache):
    with patch("bids_repair.target.httpx.get") as get:
        with pytest.raises(ValueError, match="Unsupported target"):
            with target_schema(REMOTE_VERSION):
                pytest.fail("An unsupported offline target must fail")
    get.assert_not_called()


@pytest.mark.parametrize(
    "command, function",
    [
        ([], "plan_bids_repair"),
        (["validate"], "get_bids_validation_report"),
        (["standardize"], "standardize_dataset"),
        (["plan"], "plan_bids_repair"),
    ],
)
def test_dataset_version_is_default_without_prompt(dataset, command, function):
    report = {"datasets": [{"issues": [], "errors": 0}], "findings": []}
    with (
        patch(f"bids_repair.cli.{function}", return_value=report) as operation,
        patch("bids_repair.cli._interactive", return_value=False),
        patch("bids_repair.cli.typer.prompt") as prompt,
        patch("bids_repair.target.httpx.get") as get,
    ):
        result = runner.invoke(app, [*command, str(dataset), "--format", "json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)
    assert operation.call_args.kwargs["target_version"] == VERSION
    assert "differs" not in result.stderr
    prompt.assert_not_called()
    get.assert_not_called()


@pytest.mark.parametrize(
    "command, function",
    [
        ([], "plan_bids_repair"),
        (["validate"], "get_bids_validation_report"),
        (["standardize"], "standardize_dataset"),
    ],
)
def test_explicit_target_warns_once_and_preserves_json_and_metadata(
    dataset,
    command,
    function,
):
    description = dataset / "dataset_description.json"
    description.write_text('{"Name":"Fixture","BIDSVersion":"1.0.0"}')
    before = description.read_bytes()
    with patch(
        f"bids_repair.cli.{function}", return_value={"datasets": [{"issues": []}]}
    ) as operation:
        result = runner.invoke(
            app,
            [
                *command,
                str(dataset),
                "--bids-version",
                VERSION,
                "--format",
                "json",
            ],
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)
    assert result.stderr.count("differs") == 1
    assert VERSION in result.stderr
    assert "1.0.0" in result.stderr
    assert operation.call_args.kwargs["target_version"] == VERSION
    assert description.read_bytes() == before


def test_latest_is_resolved_before_mismatch_comparison(dataset):
    with (
        patch("bids_repair.cli.resolve_bids_version", return_value=VERSION) as resolve,
        patch("bids_repair.cli.plan_bids_repair", return_value={"findings": []}),
    ):
        result = runner.invoke(app, [str(dataset), "--bids-version", "latest"])
    assert result.exit_code == 0, result.output
    resolve.assert_called_once_with("latest")
    assert "differs" not in result.stderr


def test_whitespace_in_versions_does_not_cause_a_false_warning(dataset):
    (dataset / "dataset_description.json").write_text(
        json.dumps(
            {
                "BIDSVersion": f"  {VERSION}  ",
            }
        )
    )
    with patch("bids_repair.cli.plan_bids_repair", return_value={"findings": []}):
        result = runner.invoke(
            app,
            [
                str(dataset),
                "--bids-version",
                f" {VERSION} ",
                "--format",
                "json",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "differs" not in result.stderr


def test_description_io_errors_are_not_hidden_by_explicit_target(dataset):
    with (
        patch(
            "bids_repair.cli.read_dataset_description",
            side_effect=PermissionError("denied"),
        ),
        patch("bids_repair.cli.plan_bids_repair") as operation,
    ):
        result = runner.invoke(
            app,
            [
                str(dataset),
                "--bids-version",
                VERSION,
                "--format",
                "json",
            ],
        )
    assert result.exit_code == 1
    assert "denied" in json.loads(result.stdout)["error"]["message"]
    operation.assert_not_called()


def test_explicit_target_allows_malformed_description_to_reach_repair(dataset):
    (dataset / "dataset_description.json").write_text("{")
    with patch(
        "bids_repair.cli.plan_bids_repair", return_value={"findings": []}
    ) as operation:
        result = runner.invoke(
            app,
            [
                str(dataset),
                "--bids-version",
                VERSION,
                "--format",
                "json",
            ],
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)
    assert operation.call_args.kwargs["bids_version"] == VERSION
    assert "invalid" in result.stderr


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        "[]",
        "broken",
        '{"BIDSVersion":null}',
        '{"BIDSVersion":42}',
        '{"BIDSVersion":"latest"}',
        '{"BIDSVersion":"banana"}',
    ],
)
def test_invalid_declaration_requires_unattended_target(dataset, content):
    (dataset / "dataset_description.json").write_text(content)
    with patch("bids_repair.cli._interactive", return_value=False):
        result = runner.invoke(app, [str(dataset)])
    assert result.exit_code == 2, result.output
    assert "Supply --bids-version" in result.output
    assert "Warning:" in result.stderr


def test_interactive_json_prompt_goes_to_stderr(dataset):
    (dataset / "dataset_description.json").unlink()
    with (
        patch("bids_repair.cli._interactive", return_value=True),
        patch("bids_repair.cli.plan_bids_repair", return_value={"findings": []}),
    ):
        result = runner.invoke(
            app, [str(dataset), "--format", "json"], input=f"{VERSION}\n"
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"findings": []}
    assert "Target BIDS standard version" in result.stderr


def test_declared_version_symlink_is_rejected(dataset, tmp_path):
    description = dataset / "dataset_description.json"
    description.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"BIDSVersion": VERSION}))
    description.symlink_to(outside)
    with patch("bids_repair.cli.plan_bids_repair") as operation:
        result = runner.invoke(app, [str(dataset), "--bids-version", VERSION])
    assert result.exit_code == 1
    assert "escapes" in result.output or "symlink" in result.output
    operation.assert_not_called()


@pytest.mark.parametrize(
    "command, function",
    [
        (["--apply"], "validate_and_repair_bids_dataset"),
        (["--dry-run"], "plan_bids_repair"),
    ],
)
def test_remote_schema_is_passed_through_cli(
    dataset, cache, schema_document, command, function
):
    application = {
        "unresolved": [],
        "operations": [],
        "severity_gate": {"threshold": "warning", "passed": True},
    }
    with (
        patch("bids_repair.target.httpx.get", return_value=response(schema_document)),
        patch(
            f"bids_repair.cli.{function}", return_value={"repair": application}
        ) as operation,
    ):
        result = runner.invoke(
            app,
            [
                str(dataset),
                *command,
                "--bids-version",
                REMOTE_VERSION,
                "--format",
                "json",
            ],
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)
    assert operation.call_args.kwargs["schema_path"] == cache
    assert operation.call_args.kwargs["target_version"] == REMOTE_VERSION
    assert result.stderr.count("differs") == 1


@pytest.mark.parametrize(
    "command, function",
    [
        (["validate"], "get_bids_validation_report"),
        (["standardize"], "standardize_dataset"),
    ],
)
def test_dataset_target_can_download_schema_for_each_subcommand(
    dataset,
    cache,
    schema_document,
    command,
    function,
):
    (dataset / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "Fixture",
                "BIDSVersion": REMOTE_VERSION,
            }
        )
    )
    with (
        patch("bids_repair.target.httpx.get", return_value=response(schema_document)),
        patch(
            f"bids_repair.cli.{function}", return_value={"datasets": [{"issues": []}]}
        ) as operation,
    ):
        result = runner.invoke(app, [*command, str(dataset), "--format", "json"])
    assert result.exit_code == 0, result.output
    assert operation.call_args.kwargs["schema_path"] == cache
    assert operation.call_args.kwargs["target_version"] == REMOTE_VERSION
    assert "differs" not in result.stderr


def test_real_apply_warns_and_preserves_valid_version_declaration(dataset):
    description = dataset / "dataset_description.json"
    description.write_text('{"Name":"Fixture","BIDSVersion":"1.0.0"}')
    (dataset / "participants.tsv").write_text("participant_id\nsub-01\n")
    image = dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    image.parent.mkdir(parents=True)
    image.touch()
    before = description.read_bytes()
    result = runner.invoke(
        app,
        [
            str(dataset),
            "--apply",
            "--bids-version",
            VERSION,
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["target_bids_version"] == VERSION
    assert description.read_bytes() == before
    assert result.stderr.count("differs") == 1


@pytest.mark.parametrize("content", ["[]", "broken", '{"bids_version":"0.0.0"}'])
def test_invalid_local_override_does_not_download(dataset, cache, tmp_path, content):
    local = tmp_path / "local.json"
    local.write_text(content)
    with patch("bids_repair.target.httpx.get") as get:
        result = runner.invoke(
            app,
            [
                str(dataset),
                "--bids-version",
                VERSION,
                "--schema",
                str(local),
                "--format",
                "json",
            ],
        )
    assert result.exit_code == 1
    assert "error" in json.loads(result.stdout)
    get.assert_not_called()


@pytest.mark.parametrize("command", [[], ["--apply"], ["validate"], ["standardize"]])
def test_acquisition_failure_precedes_any_dataset_operation(dataset, cache, command):
    before = {p: p.read_bytes() for p in dataset.iterdir()}
    arguments = (
        [command[0], str(dataset)]
        if command and not command[0].startswith("--")
        else [str(dataset), *command]
    )
    with (
        patch("bids_repair.target.httpx.get", return_value=response(status=404)),
        patch("bids_repair.cli.plan_bids_repair") as plan,
        patch("bids_repair.cli.validate_and_repair_bids_dataset") as apply,
        patch("bids_repair.cli.get_bids_validation_report") as validate,
        patch("bids_repair.cli.standardize_dataset") as standardize,
    ):
        result = runner.invoke(
            app,
            [
                *arguments,
                "--bids-version",
                REMOTE_VERSION,
                "--format",
                "json",
            ],
        )
    assert result.exit_code == 1
    assert "--schema PATH" in json.loads(result.stdout)["error"]["message"]
    for operation in (plan, apply, validate, standardize):
        operation.assert_not_called()
    assert {p: p.read_bytes() for p in dataset.iterdir()} == before
