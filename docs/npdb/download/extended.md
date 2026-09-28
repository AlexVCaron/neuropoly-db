# `npdb download`

- [`npdb download`](#npdb-download)
  - [Download backends](#download-backends)
    - [HTTP backend (automatic)](#http-backend-automatic)
    - [Git backend (automatic)](#git-backend-automatic)
    - [Git-annex backend (automatic)](#git-annex-backend-automatic)

## Download backends

The command auto-selects a download backend per dataset using query-result columns:

- If `AccessLink` contains a valid HTTP(S) URL, `npdb` downloads directly with the HTTP backend.
- Otherwise, it falls back to git sparse checkout using `RepositoryURL` and `ImagingSessionPath`.
- If the repository exposes a `git-annex` branch, annex retrieval is enabled automatically.

### HTTP backend (automatic)

For HTTP links, `npdb` uses the `httpx` client and downloads unique URLs found in the `AccessLink` column.

### Git backend (automatic)

Downloads the dataset from a `git` repository, using sparse checkout logic to only download the files present in the NeuroBagel query result.

Instead of relying on the `AccessLink` column, the command will use several columns to reconstruct the `git` repository URL and the paths to the datasets' files :

- `RepositoryURL` : the URL of the `git` repository hosting the dataset
- `ImagingSessionPath` : the subpath to the dataset's imaging session folder in the `git` repository

>[!WARNING]
>The command will also **download all derivatives** of the dataset if present. To turn off, use the `--no-derivatives` option.

### Git-annex backend (automatic)

Downloads the dataset from a `git-annex` repository (used in combination with the git backend).

This command relies on the same columns as the `--git` mode, and sparsely clones the dataset(s) as well. However, after cloning the `git` repository, it will use the `git-annex` command line tool to download the dataset's files from the `git-annex` references contained in the `git` repository.
