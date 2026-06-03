"""
Live display for npdb CLI commands (docker-buildx style).

Usage::

    from rich.live import Live
    from npdb.ui.display import CommandDisplay

    display = CommandDisplay()
    with Live(display, refresh_per_second=4, transient=False) as live:
        display.start_step("Cloning repository")
        # ... do work ...
        display.complete_step()

        display.start_step("Running annotation")
        # ... do work, stream output ...
        display.append_output("some line of output")
        display.complete_step()
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generator, List

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from npdb.cli.observers import DownloadObserver


class StepStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class Step:
    """A single tracked step in a :class:`CommandDisplay`."""

    name: str
    status: StepStatus = StepStatus.RUNNING
    output: List[str] = field(default_factory=list)

    def __rich__(self):
        if self.status == StepStatus.SUCCESS:
            return Text(f"\u2705 {self.name}", style="green")
        if self.status == StepStatus.RUNNING:
            body = "\n".join(self.output) if self.output else "[dim]Running\u2026[/dim]"
            return Panel(
                body,
                title=f"[yellow]\u27f3 {self.name}[/yellow]",
                border_style="yellow",
            )
        # FAILURE
        body = "\n".join(self.output) if self.output else ""
        return Panel(
            body,
            title=f"[red]\u274c {self.name}[/red]",
            border_style="red",
        )


class CommandDisplay:
    """
    Maintains an ordered list of :class:`Step` objects and renders them as a
    Rich *Group* suitable for use with :class:`rich.live.Live`.

    Typical workflow per step::

        display.start_step("Step name")
        display.append_output("line of output")   # optional, repeatable
        display.complete_step()                    # or display.fail_step()
    """

    def __init__(self) -> None:
        self._steps: List[Step] = []

    # ------------------------------------------------------------------
    # Rich protocol
    # ------------------------------------------------------------------

    def __rich__(self):
        return Group(*self._steps)

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def start_step(self, name: str) -> None:
        """Append a new step in RUNNING state."""
        self._steps.append(Step(name=name))

    def append_output(self, line: str) -> None:
        """Append an output line to the currently running step."""
        if self._steps:
            self._steps[-1].output.append(line)

    def complete_step(self) -> None:
        """Mark the last step as SUCCESS."""
        if self._steps:
            self._steps[-1].status = StepStatus.SUCCESS

    def fail_step(self, output: List[str] | None = None) -> None:
        """Mark the last step as FAILURE, optionally replacing its output."""
        if self._steps:
            self._steps[-1].status = StepStatus.FAILURE
            if output is not None:
                self._steps[-1].output = list(output)


class _LineCallbackWriter:
    """A stdout-compatible writer that feeds complete lines to a callback."""

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, callback: Callable[[str], None]) -> None:
        self._callback = callback
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._callback(line)
        return len(text)

    def flush(self) -> None:
        pass

    def fileno(self) -> int:  # needed by some libraries that call fileno()
        raise OSError("_LineCallbackWriter has no real file descriptor")


@contextmanager
def capture_stdout(callback: Callable[[str], None]) -> Generator[None, None, None]:
    """Context manager that redirects sys.stdout lines to *callback*.

    Rich's ``Live`` console is unaffected because it holds its own file
    reference captured at construction time.
    """
    original = sys.stdout
    sys.stdout = _LineCallbackWriter(callback)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = original


# ---------------------------------------------------------------------------
# Download progress display (docker-buildx style)
# ---------------------------------------------------------------------------

_MAX_FILE_LINES = 8  # max file-progress lines shown per repo panel


@dataclass
class _FileState:
    """Per-file download state tracked inside a :class:`RepoState`."""

    name: str
    bytes_done: int = 0
    bytes_total: int = 0
    done: bool = False


@dataclass
class RepoState:
    """
    Render state for a single repository being downloaded.

    While *status* is ``RUNNING`` the panel shows a step progress bar and a
    rolling list of the most-recently active files.  On completion it collapses
    to a single success/failure line — identical to the ``docker buildx build``
    output style.
    """

    name: str
    current_step: str = ""
    step_num: int = 0
    total_steps: int = 0
    status: StepStatus = StepStatus.RUNNING
    files: dict = field(default_factory=dict)  # str → _FileState
    file_order: List[str] = field(default_factory=list)

    def _step_bar(self) -> str:
        if self.total_steps <= 0:
            return self.current_step or "Starting…"
        pct = int(self.step_num * 100 / self.total_steps)
        filled = int(pct * 28 / 100)
        bar = "█" * filled + "░" * (28 - filled)
        return f"{bar}  {pct:3d}%  {self.current_step}"

    def __rich__(self):
        if self.status == StepStatus.SUCCESS:
            return Text(f"✅ {self.name}", style="green")

        lines: List[str] = [self._step_bar()]

        for key in self.file_order[-_MAX_FILE_LINES:]:
            fs: _FileState | None = self.files.get(key)
            if fs is None:
                continue
            short = os.path.basename(key) or key
            if fs.done:
                lines.append(f"  ✅ {short}")
            elif fs.bytes_total > 0:
                fpct = int(fs.bytes_done * 100 / fs.bytes_total)
                ffilled = int(fpct * 20 / 100)
                fbar = "█" * ffilled + "░" * (20 - ffilled)
                lines.append(f"  {fbar}  {fpct:3d}%  {short}")
            else:
                lines.append(f"  ➳ {short}")

        body = "\n".join(lines)

        if self.status == StepStatus.FAILURE:
            return Panel(body, title=f"[red]✗ {self.name}[/red]", border_style="red")
        return Panel(
            body, title=f"[yellow]⟳ {self.name}[/yellow]", border_style="yellow"
        )


class RepoDownloadDisplay(DownloadObserver):
    """
    Implements :class:`~npdb.cli.observers.DownloadObserver` and renders as a
    Rich *Group* suitable for use with :class:`rich.live.Live`.

    Each repository appears as a bordered yellow panel while it is being
    processed.  Once finished it collapses to a single green ✅ or red ✗ line,
    freeing terminal space for the next repository.

    Thread-safe: all mutations are protected by an internal lock so that the
    Live refresh thread and the download worker threads operate concurrently
    without data races.
    """

    def __init__(self) -> None:
        self._repos: dict[str, RepoState] = {}
        self._repo_order: List[str] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Rich protocol
    # ------------------------------------------------------------------

    def __rich__(self):
        with self._lock:
            states = [self._repos[k] for k in self._repo_order]
        return Group(*states)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _get_or_create(self, repo: str) -> RepoState:
        """Return existing :class:`RepoState` or create one.  Must hold *_lock*."""
        if repo not in self._repos:
            state = RepoState(name=repo)
            self._repos[repo] = state
            self._repo_order.append(repo)
        return self._repos[repo]

    # ------------------------------------------------------------------
    # DownloadObserver implementation
    # ------------------------------------------------------------------

    def on_repo_step(
        self, repo: str, step: str, step_num: int, total_steps: int
    ) -> None:
        with self._lock:
            state = self._get_or_create(repo)
            state.current_step = step
            state.step_num = step_num
            state.total_steps = total_steps

    def on_file_progress(
        self, repo: str, file: str, bytes_done: int, bytes_total: int
    ) -> None:
        with self._lock:
            state = self._get_or_create(repo)
            if file not in state.files:
                state.file_order.append(file)
                state.files[file] = _FileState(name=file)
            fs = state.files[file]
            fs.bytes_done = bytes_done
            fs.bytes_total = bytes_total

    def on_file_complete(self, repo: str, file: str) -> None:
        with self._lock:
            state = self._get_or_create(repo)
            if file not in state.files:
                state.file_order.append(file)
                state.files[file] = _FileState(name=file, done=True)
            else:
                state.files[file].done = True

    def on_repo_done(self, repo: str, success: bool) -> None:
        with self._lock:
            state = self._get_or_create(repo)
            state.status = StepStatus.SUCCESS if success else StepStatus.FAILURE
