# `npdb download`

- [`npdb download`](#npdb-download)
  - [Download backends](#download-backends)
    - [`--git` mode](#--git-mode)
    - [`--git-annex` mode](#--git-annex-mode)

## Download backends

By default, the command assumes the dataset is accessible via the `HTTP(S)` protocol and uses the `requests` library to download the links contained in the `AccessLink` column of the NeuroBagel query result.

### `--git` mode

Downloads the dataset from a `git` repository, using sparse checkout logic to only download the files present in the NeuroBagel query result.

Instead of relying on the `AccessLink` column, the command will use several columns to reconstruct the `git` repository URL and the paths to the datasets' files :

- `RepositoryURL` : the URL of the `git` repository hosting the dataset
- `ImagingSessionPath` : the subpath to the dataset's imaging session folder in the `git` repository

>[!WARNING]
>The command will also **download all derivatives** of the dataset if present. To turn off, use the `--no-derivatives` option.

### `--git-annex` mode

Downloads the dataset from a `git-annex` repository (used in combination with the `--git` mode).

This command relies on the same columns as the `--git` mode, and sparsely clones the dataset(s) as well. However, after cloning the `git` repository, it will use the `git-annex` command line tool to download the dataset's files from the `git-annex` references contained in the `git` repository.
