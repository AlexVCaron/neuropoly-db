import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import typer
from dotenv import load_dotenv
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from npdb.annotation.modes import AnnotationMode
from npdb.cli.display import RepoDownloadDisplay
from npdb.factories import GiteaManagerFactory

OPTION_GROUP_NAMES = {
    "input": "Input Options",
    "output": "Output Options",
    "behavior": "Behavior Options",
    "automation": "Automation Options",
    "ai": "AI Options",
    "troubleshooting": "Troubleshooting",
}


def show_help(ctx: typer.Context, value: bool):
    if value:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def help_option():
    return typer.Option(
        False,
        "--help",
        "-h",
        callback=show_help,
        help="Show this message and exit.",
        rich_help_panel=OPTION_GROUP_NAMES["troubleshooting"],
    )


npdb = typer.Typer(
    help="Conversion tools and utilities for NeuroPoly Database (BIDS)",
    context_settings={"help_option_names": ["--help", "-h"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
    epilog="Run 'npdb COMMAND --help' for more information on a command.",
)


@npdb.callback()
def main():
    return


@npdb.command()
def gitea2bagel(
    dataset: str = typer.Argument(
        ...,
        help="Dataset name on Gitea (under the datasets organization).",
    ),
    output: Path = typer.Argument(
        ...,
        help="Output directory for converted dataset.",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    verify_ssl: bool = typer.Option(
        True,
        help="Verify SSL certificates when connecting to Gitea.",
        rich_help_panel=OPTION_GROUP_NAMES["input"],
    ),
    mode: str = typer.Option(
        AnnotationMode.MANUAL.value,
        help="Annotation mode: manual|assist|auto|full-auto",
        rich_help_panel=OPTION_GROUP_NAMES["behavior"],
    ),
    phenotype_dict: Optional[Path] = typer.Option(
        None,
        help="Path to phenotype dictionary JSON for prefill.",
        exists=True,
        rich_help_panel=OPTION_GROUP_NAMES["input"],
    ),
    headless: bool = typer.Option(
        True,
        "--headless/--headed",
        help="Run browser in headless mode (automation modes).",
        rich_help_panel=OPTION_GROUP_NAMES["automation"],
    ),
    timeout: int = typer.Option(
        300,
        help="Timeout per step in seconds (automation modes).",
        rich_help_panel=OPTION_GROUP_NAMES["automation"],
    ),
    artifacts_dir: Optional[Path] = typer.Option(
        None,
        help="Directory for screenshots/traces (automation modes).",
        file_okay=False,
        dir_okay=True,
        writable=True,
        rich_help_panel=OPTION_GROUP_NAMES["automation"],
    ),
    ai_provider: Optional[str] = typer.Option(
        None,
        help="AI provider (e.g., 'ollama').",
        rich_help_panel=OPTION_GROUP_NAMES["ai"],
    ),
    ai_model: Optional[str] = typer.Option(
        None,
        help="AI model name (e.g., 'neural-chat').",
        rich_help_panel=OPTION_GROUP_NAMES["ai"],
    ),
    header_map: Optional[Path] = typer.Option(
        None,
        "--header-map",
        help="JSON file mapping desired Neurobagel headers to input variants.",
        exists=True,
        rich_help_panel=OPTION_GROUP_NAMES["input"],
    ),
    help_: bool = help_option(),
):
    """
    [bold]Convert a BIDS dataset from Gitea to Neurobagel JSON-LD format[/bold]

    This command automates annotation of phenotypic data using the selected mode:
    * [cyan]manual[/cyan]: Interactive annotation tool
    * [cyan]assist[/cyan]: Browser automation with user confirmation
    * [cyan]auto[/cyan]: Fully automated with ML-based suggestions
    * [cyan]full-auto[/cyan]: Experimental unattended mode (requires review!)
    """
    import asyncio

    from dotenv import load_dotenv

    from npdb.annotation.standardize import load_header_map, validate_header_map_keys
    from npdb.automation.mappings.solvers import load_static_mappings
    from npdb.cli.facade import DatasetConversionFacade
    from npdb.factories import AnnotationConfigFactory, GiteaManagerFactory

    try:
        mode_enum = AnnotationMode(mode)
    except ValueError:
        typer.echo(f"Error: Invalid mode '{mode}'.", err=True)
        raise typer.Exit(code=1)

    if mode_enum == AnnotationMode.MANUAL and (ai_provider or ai_model):
        typer.echo("Warning: AI options ignored in manual mode.", err=True)

    if ai_provider and not ai_model:
        typer.echo("Error: --ai-model required with --ai-provider.", err=True)
        raise typer.Exit(code=1)

    if ai_model and not ai_provider:
        typer.echo("Error: --ai-provider required with --ai-model.", err=True)
        raise typer.Exit(code=1)

    if header_map:
        try:
            hmap = load_header_map(header_map)
            static = load_static_mappings()
            valid_keys = set(static.get("mappings", {}).keys())
            validate_header_map_keys(hmap, valid_keys)
        except (ValueError, FileNotFoundError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        typer.echo(f"Error creating output directory '{output}': {e}", err=True)
        raise typer.Exit(code=1)

    if artifacts_dir:
        try:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            typer.echo(
                f"Error creating artifacts directory '{artifacts_dir}': {e}", err=True
            )
            raise typer.Exit(code=1)

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

    try:
        gitea_manager = GiteaManagerFactory.create_from_env(ssl_verify=verify_ssl)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    annotation_config = AnnotationConfigFactory.create_from_cli_args(
        mode=mode,
        headless=headless,
        timeout=timeout,
        artifacts_dir=artifacts_dir,
        ai_provider=ai_provider,
        ai_model=ai_model,
        phenotype_dictionary=phenotype_dict,
        header_map=header_map,
    )

    facade = DatasetConversionFacade(gitea_manager, annotation_config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(f"Converting {dataset}...", total=None)
        try:
            asyncio.run(facade.run(dataset, output))
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    typer.echo(f"Conversion complete! Output saved to: {output}")


def _read_download_tsv(tsv_path: Path) -> list[dict]:
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("TSV file is empty or has no header row")
        rows = list(reader)
    if not rows:
        raise ValueError("TSV file contains no data rows")
    return rows


def _fetch_url(url: str, dest: Path, timeout: int = 300) -> tuple[bool, str]:
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        return True, f"Downloaded: {dest.name}"
    except Exception as exc:
        return False, str(exc)


def _is_http_url(value: str) -> bool:
    is_http = value.startswith(("http://", "https://"))
    is_git = value.endswith(".git") or "/tree/" in value
    return is_http and not is_git


def _normalize_repo_url_for_git(repo_url: str) -> str:
    parsed = urlparse(repo_url if "://" in repo_url else f"https://{repo_url}")
    repo_path = parsed.path.rstrip("/")
    tree_idx = repo_path.find("/tree/")
    if tree_idx != -1:
        repo_path = repo_path[:tree_idx]
    if not repo_path.endswith(".git"):
        repo_path += ".git"
    return f"{parsed.scheme}://{parsed.netloc}{repo_path}"


def _repo_has_git_annex(gitea_manager, repo_url: str) -> bool:
    git_url = _normalize_repo_url_for_git(repo_url)
    cmd = (
        ["git"]
        + gitea_manager.git_http_config()
        + ["ls-remote", "--heads", git_url, "refs/heads/git-annex"]
    )

    try:
        stdout, _ = gitea_manager._run_git(
            cmd,
            env=gitea_manager.git_env(),
            context=f"probe git-annex metadata branch for '{repo_url}'",
        )
    except RuntimeError:
        return False

    return bool(stdout.strip())


def _looks_like_non_git_repo_error(message: str) -> bool:
    lowered = message.lower()
    patterns = [
        "not a git repository",
        "does not appear to be a git repository",
        "fatal: repository",
        "repository not found",
    ]
    return any(p in lowered for p in patterns)


@npdb.command("download")
def download(
    query_results: Path = typer.Argument(
        ...,
        help="Path to query results TSV file with AccessLink column.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    derivatives: bool = typer.Option(
        True,
        help="Download derivatives associated to the raw input data in each repository (git mode only).",
        rich_help_panel=OPTION_GROUP_NAMES["behavior"],
    ),
    output_dir: Path = typer.Option(
        Path.cwd(),
        "--output-dir",
        "-o",
        help="Directory to save downloaded files.",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    max_workers: int = typer.Option(
        4,
        "--max-workers",
        help="Maximum parallel HTTP downloads.",
        rich_help_panel=OPTION_GROUP_NAMES["behavior"],
    ),
    verify_ssl: bool = typer.Option(
        True,
        help="Verify SSL certificates when connecting to Gitea (git mode only).",
        rich_help_panel=OPTION_GROUP_NAMES["input"],
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print each git command before it runs (git mode only).",
        rich_help_panel=OPTION_GROUP_NAMES["troubleshooting"],
    ),
    help_: bool = help_option(),
):
    """
    [bold]Download imaging data from query results TSV[/bold]

    This command reads a TSV file containing query results and automatically
    selects the download protocol per dataset:

    * [cyan]HTTP:[/cyan] If [bold]AccessLink[/bold] is present for the dataset, download from
      link(s) directly.
    * [cyan]Git:[/cyan] Otherwise, clone from [bold]RepositoryURL[/bold] with sparse checkout.
    * [cyan]Git-annex:[/cyan] If the repository exposes a [bold]git-annex[/bold] branch,
      run annex content retrieval after git checkout.

    Git operations require [bold]NP_GITEA_APP_URL[/bold], [bold]NP_GITEA_APP_USER[/bold],
    and [bold]NP_GITEA_APP_TOKEN[/bold] environment variables.
    """
    try:
        rows = _read_download_tsv(query_results)
    except (OSError, ValueError) as exc:
        typer.echo(f"Error reading TSV: {exc}", err=True)
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_with_http: set[str] = set()
    seen_urls: set[str] = set()
    http_jobs: list[tuple[str, Path, str, str]] = []

    for row in rows:
        dataset = (row.get("DatasetName") or "unknown").strip()
        subject = (row.get("SubjectID") or "unknown").strip()
        url = (row.get("AccessLink") or "").strip()
        if not _is_http_url(url):
            continue
        datasets_with_http.add(dataset)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        filename = os.path.basename(url.split("?")[0]) or f"{subject}.bin"
        dest = output_dir / dataset / subject / filename
        http_jobs.append((url, dest, dataset, subject))

    http_failures = 0
    if http_jobs:
        typer.echo(
            f"Downloading {len(http_jobs)} file(s) via HTTP ({max_workers} workers)..."
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_fetch_url, url, dest): (dataset, subject)
                for url, dest, dataset, subject in http_jobs
            }
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "Downloading HTTP links...", total=len(futures)
                )
                for future in as_completed(futures):
                    ok, msg = future.result()
                    dataset, subject = futures[future]
                    typer.echo(
                        f"{'SUCCESS' if ok else 'FAIL'} {dataset}/{subject}: {msg}"
                    )
                    if not ok:
                        http_failures += 1
                    progress.advance(task)

        typer.echo("HTTP download phase complete.")

    subjects: list[tuple[str, str, str]] = []
    for row in rows:
        dataset = (row.get("DatasetName") or "unknown").strip()
        if dataset in datasets_with_http:
            continue
        repo_url = (row.get("RepositoryURL") or "").strip()
        imaging_path = (row.get("ImagingSessionPath") or "").strip()
        if not repo_url or not imaging_path:
            continue
        subjects.append((repo_url, imaging_path, dataset))

    git_failures = 0
    if subjects:
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

        try:
            if verbose:
                typer.echo("Initializing Gitea manager...")
            gitea_manager = GiteaManagerFactory.create_from_env(ssl_verify=verify_ssl)
            if verbose:
                typer.echo("Gitea manager initialized successfully.")
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        gitea_manager.verbose = verbose

        grouped: dict[tuple[str, str], list[str]] = {}
        for repo_url, sparse_path, dataset in subjects:
            key = (repo_url, dataset)
            if key not in grouped:
                grouped[key] = []
            if sparse_path not in grouped[key]:
                grouped[key].append(sparse_path)

        typer.echo(
            "Downloading via git auto-selection "
            f"({len(subjects)} paths across {len(grouped)} repo(s))..."
        )

        display = RepoDownloadDisplay()
        gitea_manager.add_download_observer(display)

        with Live(display, refresh_per_second=4, transient=False):
            for (repo_url, dataset), sparse_paths in grouped.items():
                use_annex = _repo_has_git_annex(gitea_manager, repo_url)
                if verbose:
                    protocol = "git + git-annex" if use_annex else "git"
                    typer.echo(f"Protocol for {dataset}: {protocol}")

                results = gitea_manager.download_subjects(
                    [(repo_url, p, dataset) for p in sparse_paths],
                    output_dir,
                    use_annex=use_annex,
                    derivatives=derivatives,
                )

                for ok, label, message in results:
                    if ok:
                        continue
                    git_failures += 1
                    if _looks_like_non_git_repo_error(message):
                        typer.echo(
                            f"FAIL {label}: RepositoryURL is not a git repository.",
                            err=True,
                        )
                    else:
                        typer.echo(f"FAIL {label}: {message}", err=True)

    if not http_jobs and not subjects:
        typer.echo(
            "Warning: No download targets found in TSV (no valid AccessLink or git imaging rows).",
            err=True,
        )
        return

    if git_failures or http_failures:
        typer.echo(
            f"Download completed with failures (HTTP: {http_failures}, git: {git_failures}).",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo("Download complete!")


standardize = typer.Typer(
    help="Standardization tools for BIDS datasets.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
npdb.add_typer(standardize, name="standardize")


@standardize.command("bids")
def standardize_bids(
    bids_dir: Path = typer.Argument(
        ...,
        help="Path to BIDS dataset root (must contain participants.tsv).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    mode: str = typer.Option(
        AnnotationMode.MANUAL.value,
        help="Annotation mode: manual|assist|auto|full-auto",
        rich_help_panel=OPTION_GROUP_NAMES["behavior"],
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print changes to terminal without writing files.",
        rich_help_panel=OPTION_GROUP_NAMES["behavior"],
    ),
    no_new_columns: bool = typer.Option(
        False,
        "--no-new-columns",
        help="Don't add missing standard columns (e.g., age, sex).",
        rich_help_panel=OPTION_GROUP_NAMES["behavior"],
    ),
    keep_annotations: bool = typer.Option(
        False,
        "--keep-annotations",
        help="Include Neurobagel Annotations block in participants.json.",
        rich_help_panel=OPTION_GROUP_NAMES["behavior"],
    ),
    phenotype_dict: Optional[Path] = typer.Option(
        None,
        help="Path to phenotype dictionary JSON for prefill.",
        exists=True,
        rich_help_panel=OPTION_GROUP_NAMES["input"],
    ),
    headless: bool = typer.Option(
        True,
        "--headless/--headed",
        help="Run browser in headless mode (automation modes).",
        rich_help_panel=OPTION_GROUP_NAMES["automation"],
    ),
    timeout: int = typer.Option(
        300,
        help="Timeout per step in seconds (automation modes).",
        rich_help_panel=OPTION_GROUP_NAMES["automation"],
    ),
    artifacts_dir: Optional[Path] = typer.Option(
        None,
        help="Directory for screenshots/traces (automation modes).",
        file_okay=False,
        dir_okay=True,
        writable=True,
        rich_help_panel=OPTION_GROUP_NAMES["automation"],
    ),
    ai_provider: Optional[str] = typer.Option(
        None,
        help="AI provider (e.g., 'ollama').",
        rich_help_panel=OPTION_GROUP_NAMES["ai"],
    ),
    ai_model: Optional[str] = typer.Option(
        None,
        help="AI model name (e.g., 'neural-chat').",
        rich_help_panel=OPTION_GROUP_NAMES["ai"],
    ),
    header_map: Optional[Path] = typer.Option(
        None,
        "--header-map",
        help="JSON file mapping desired headers to input variants.",
        exists=True,
        rich_help_panel=OPTION_GROUP_NAMES["input"],
    ),
    help_: bool = help_option(),
):
    """
    [bold]Standardize BIDS dataset participants.tsv and participants.json[/bold]

    Renames column headers to canonical BIDS names, adds missing standard
    columns, and generates a BIDS-compliant participants.json sidecar.

    Edits the dataset in-place. Use [cyan]--dry-run[/cyan] to preview changes
    without writing files.
    """
    import asyncio

    from npdb.cli.facade import BIDSStandardizationFacade
    from npdb.factories import AnnotationConfigFactory

    try:
        mode_enum = AnnotationMode(mode)
    except ValueError:
        typer.echo(f"Error: Invalid mode '{mode}'.", err=True)
        raise typer.Exit(code=1)

    if mode_enum == AnnotationMode.MANUAL and (ai_provider or ai_model):
        typer.echo("Warning: AI options ignored in manual mode.", err=True)

    if ai_provider and not ai_model:
        typer.echo("Error: --ai-model required with --ai-provider.", err=True)
        raise typer.Exit(code=1)
    if ai_model and not ai_provider:
        typer.echo("Error: --ai-provider required with --ai-model.", err=True)
        raise typer.Exit(code=1)

    participants_tsv = bids_dir / "participants.tsv"
    if not participants_tsv.exists():
        typer.echo(f"Error: participants.tsv not found in {bids_dir}.", err=True)
        raise typer.Exit(code=1)

    if dry_run:
        typer.echo("Dry-run mode: no files will be modified.\n")

    config = AnnotationConfigFactory.create_from_cli_args(
        mode=mode,
        headless=headless,
        timeout=timeout,
        artifacts_dir=artifacts_dir,
        ai_provider=ai_provider,
        ai_model=ai_model,
        phenotype_dictionary=phenotype_dict,
        dry_run=dry_run,
        keep_annotations=keep_annotations,
        header_map=header_map,
        no_new_columns=no_new_columns,
    )

    facade = BIDSStandardizationFacade(config)

    try:
        asyncio.run(facade.run(bids_dir))
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during BIDS standardization: {e}", err=True)
        raise typer.Exit(code=1)

    if dry_run:
        typer.echo("\nDry-run complete. No files were modified.")
    else:
        typer.echo(f"\nBIDS standardization complete: {bids_dir}")
