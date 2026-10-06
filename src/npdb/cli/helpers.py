import csv
import json
import re
import stat
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import httpx


def unpack_provider_archives(fetched: str | Path) -> Path:
    fetched_path = Path(fetched)
    root = fetched_path if fetched_path.is_dir() else fetched_path.parent
    candidates = (
        sorted(root.rglob("*")) if fetched_path.is_dir() else [fetched_path]
    )
    tar_suffixes = (
        ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"
    )
    archives = [
        path for path in candidates
        if path.is_file() and (
            path.name.lower().endswith(".zip")
            or path.name.lower().endswith(tar_suffixes)
        )
    ]
    for archive_path in archives:
        if archive_path.is_symlink():
            raise ValueError(f"Refusing to unpack symlink archive: {archive_path}")
        destination = archive_path.parent.resolve()

        def validate_member(name: str) -> None:
            target = (destination / name).resolve()
            if not target.is_relative_to(destination) or target == archive_path.resolve():
                raise ValueError(
                    f"Unsafe member '{name}' in archive '{archive_path}'."
                )

        if archive_path.name.lower().endswith(".zip"):
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    validate_member(member.filename)
                    if stat.S_ISLNK(member.external_attr >> 16):
                        raise ValueError(
                            f"Unsafe link '{member.filename}' in archive '{archive_path}'."
                        )
                archive.extractall(destination)
        else:
            with tarfile.open(archive_path) as archive:
                for member in archive.getmembers():
                    validate_member(member.name)
                    if not (member.isfile() or member.isdir()):
                        raise ValueError(
                            f"Unsafe member '{member.name}' in archive '{archive_path}'."
                        )
                archive.extractall(destination, filter="data")
        archive_path.unlink()
    return root


def prepare_provider_dataset(root: Path) -> Path:
    rawdata = root / "rawdata"
    derivatives = root / "derivatives"
    auxiliary_directories = {"sourcedata", "stimuli", "code", "phenotype"}
    protected_directories = auxiliary_directories | {"rawdata", "derivatives"}
    for directory in (rawdata, derivatives):
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ValueError(f"Expected a regular directory: {directory}")

    other_directories = [
        path for path in sorted(root.iterdir())
        if path.is_dir()
        and path.name not in protected_directories
        and not path.name.startswith((".", "sub-"))
    ] if not derivatives.exists() else []
    if any(path.is_symlink() for path in other_directories):
        raise ValueError(f"Cannot reorganize symlink directories in {root}.")

    raw_children = sorted(rawdata.iterdir()) if rawdata.exists() else []
    for child in raw_children:
        destination = root / child.name
        if (
            destination.exists()
            or destination.is_symlink()
            or (child.name == "derivatives" and other_directories)
        ):
            raise FileExistsError(f"Cannot flatten rawdata: destination exists: {destination}")

    if other_directories:
        derivatives.mkdir()
        for directory in other_directories:
            directory.rename(derivatives / directory.name)
    for child in raw_children:
        child.rename(root / child.name)
    if rawdata.exists():
        rawdata.rmdir()

    participants = root / "participants.tsv"
    if participants.is_symlink() or (
        participants.exists() and not participants.is_file()
    ):
        raise ValueError(f"Expected a regular participants table: {participants}")
    if not participants.exists():
        subjects: set[str] = set()
        subject_pattern = re.compile(r"^sub-([A-Za-z0-9]+)(?=_|\.|$)")
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if (
                relative.parts[0] in auxiliary_directories | {"derivatives"}
                or any(part.startswith(".") for part in relative.parts)
            ):
                continue
            if path.is_file() or path.is_dir():
                match = subject_pattern.match(path.name)
                if match:
                    subjects.add(f"sub-{match.group(1)}")
        if not subjects:
            raise ValueError(f"No BIDS subject labels found to create {participants}.")
        with participants.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["participant_id", "age", "sex"])
            writer.writerows((subject, "N/A", "N/A") for subject in sorted(subjects))
    return root


def read_tsv(tsv_path: Path) -> list[dict]:
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("TSV file is empty or has no header row")
        rows = list(reader)
    if not rows:
        raise ValueError("TSV file contains no data rows")
    return rows


def fetch_url(url: str, dest: Path, timeout: int = 300) -> tuple[bool, str]:
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


def is_http_url(value: str) -> bool:
    is_http = value.startswith(("http://", "https://"))
    is_git = value.endswith(".git") or "/tree/" in value
    return is_http and not is_git


def normalize_repo_url_for_git(repo_url: str) -> str:
    parsed = urlparse(repo_url if "://" in repo_url else f"https://{repo_url}")
    repo_path = parsed.path.rstrip("/")
    tree_idx = repo_path.find("/tree/")
    if tree_idx != -1:
        repo_path = repo_path[:tree_idx]
    if not repo_path.endswith(".git"):
        repo_path += ".git"
    return f"{parsed.scheme}://{parsed.netloc}{repo_path}"


def repo_has_git_annex(gitea_manager, repo_url: str) -> bool:
    git_url = normalize_repo_url_for_git(repo_url)
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


def looks_like_non_git_repo_error(message: str) -> bool:
    lowered = message.lower()
    patterns = [
        "not a git repository",
        "does not appear to be a git repository",
        "fatal: repository",
        "repository not found",
    ]
    return any(p in lowered for p in patterns)


def extend_bids_description(
    dataset: str, local_clone: str, url: str, access_type: str = "restricted"
):
    """
    Extend the dataset_description.json file using NeuroBagel standard.
    See : https://neurobagel.org/user_guide/dataset_description/#editable-template
    """
    desc_path = Path(local_clone) / "dataset_description.json"
    with open(desc_path, "r") as f:
        description = json.load(f)

    description["Name"] = dataset

    if not description.get("Keywords"):
        description["Keywords"] = [dataset]

    description["RepositoryURL"] = f"{url}"

    description["AccessInstructions"] = (
        "Refer to the access link provided with the repository."
    )
    description["AccessLink"] = description["RepositoryURL"]
    description["AccessType"] = access_type

    return description
