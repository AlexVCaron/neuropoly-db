# Install npdb

Two commands, no Python knowledge needed. Works on Linux and macOS (on Windows, use [WSL](https://learn.microsoft.com/windows/wsl/install)).

## 1. Install the tools npdb relies on

`npdb` downloads with **git** and **git-annex**.

- **Ubuntu/Debian:**

  ```bash
  sudo apt-get install -y git git-annex
  ```

- **macOS** (with [Homebrew](https://brew.sh)):

  ```bash
  brew install git git-annex
  ```

## 2. Install uv

[uv](https://docs.astral.sh/uv/) installs `npdb` and the right Python for you.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Close and reopen your terminal** afterwards.

## 3. Install npdb

Initialize the pinned validator source before building:

```bash
git submodule update --init --recursive -- tools/bids-validator-rust
uv build
```

Published npdb sdists contain the Rust source and do not require submodule
initialization. GitHub-generated source ZIP/tar archives omit it and are not
standalone build inputs.

## 4. Check it works

```bash
npdb --help
bids-repair --help
```

You should see the list of `npdb` commands.

The same installation also provides `bids-repair` for local BIDS validation and
repair. See the [BIDS repair guide](bids-repair.md) for native-validator setup and target versions.

> [!TIP]
> `npdb: command not found`? Run `uv tool update-shell`, then close and reopen your terminal.

## Update or remove

```bash
uv tool upgrade npdb      # get the latest version
uv tool uninstall npdb    # remove it
```

Next: [Find data](find-data.md)
