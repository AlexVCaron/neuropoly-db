# Validate and repair a local BIDS-like dataset

**`bids-repair`** receives a local BIDS-like directory. It owns generic BIDS validation, structural repair, layout normalization, tables, and sidecars.

## Install

Use the [normal installation](install.md). Then validate the `bids-repair` command is available:

```Shell
bids-repair --help
```

Validation also requires the native `bids-validator-rust` executable on `PATH` or
beside the Python interpreter. If absent, you can use the following utility script to install it:

```Shell
bash scripts/install_bids_validator.sh
```

## Plan, validate, apply, recover

```bash
# Default operation: a read-only repair plan.
bids-repair ./my-bids --bids-version 1.11.1 --dry-run

# Read-only validation, without proposing/applying repairs.
bids-repair validate ./my-bids --bids-version 1.11.1 --format json

# Explicitly apply supported safe repairs, then revalidate.
bids-repair ./my-bids --bids-version 1.11.1 --apply

# Roll back an incomplete journaled repair. No version is needed.
bids-repair recover ./my-bids
```

The target version comes from:

- The `--bids-version` argument
- A valid `BIDSVersion` in the supplied `dataset_description.json`
- Interactively via the command-line

### Target version and validation coverage

BIDS schema selection uses, in order:

1. An explicit `--schema` local JSON file or schema source directory.
2. The bundled schema, if its BIDS version matches the target.
3. A validated cached schema for that exact BIDS release.
4. An automatic download of the [official compiled schema](https://bidsschematools.readthedocs.io/en/latest/description.html#schema-publication)
   from `https://bids-specification.readthedocs.io/en/vVERSION/schema.json`.

Schemas are cached at `${XDG_CACHE_HOME:-~/.cache}/bids-repair/schemas/VERSION/schema.json`.
To provide a local schema, use `--schema`:

```bash
bids-repair ./my-bids --bids-version 1.9.0 --schema /path/to/1.9.0/schema.json
```

The schema's embedded BIDS version must match the requested version.

### Repair controls and safety

| Argument                                         | Description                                                                                                                                                                                              |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--suffix-alias OLD=NEW`                       | supplies an explicit suffix translation; repeat it for multiple translations. No scientific alias is inferred from provider identity.                                                                   |
| `--bids-severity-threshold info\|warning\|error` | controls the **final** repair gate:<br />only findings strictly above it fail. The default `warning` permits warnings,<br />but fails remaining errors. Validation-only commands fail on errors. |
| `--format json`                                | returns a machine-readable result on stdout. Repair summaries<br />and manifest locations are written to stderr.                                                                                         |
| `--apply`                                      | planning is read-only; use to apply the changes to the actual dataset                                                                                                                                    |
| `--dry-run`                                    | runs through the application code without calling the commands                                                                                                                                           |

## Explicit participant standardization

```bash
bids-repair standardize ./my-bids --bids-version 1.11.1 \
  --header-map headers.json --column-metadata columns.json --dry-run
```

- Header maps accept arbitrary desired BIDS column names and will convert all matched `aliases` to them:

  ```json
  {
    "participant_id": {"aliases": ["SubjectID", "subject_id"]},
    "measurement": {"aliases": ["LocalMeasure"]}
  }
  ```
- Column metadata supplies plain BIDS sidecar entries used to fill the `participants.json` schema file:

  ```json
  {
    "measurement": {"Description": "Local measurement", "Units": "mm"}
  }
  ```
- Use `--add-column NAME` to explicitly add a missing column filled with `n/a`.
- Use `--prepare-layout` to explicitly run the existing generic layout preparation
  (including rawdata flattening and missing participants-table generation).
