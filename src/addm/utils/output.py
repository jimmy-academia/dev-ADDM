"""Centralized output for ADDM CLI.

Provides Rich-formatted console output with convenience methods.
Mode-aware: adjusts output for ondemand vs batch execution.
"""

from typing import Any, List, Optional

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table


class OutputManager:
    """Singleton Rich Console wrapper for CLI output."""

    _instance: Optional["OutputManager"] = None
    _console: Optional[Console] = None
    _quiet: bool = False
    _mode: str = "ondemand"
    _suppress_all: bool = False  # For multi-policy runs - suppress everything

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._console = Console()
        return cls._instance

    @property
    def console(self) -> Console:
        """Get underlying Rich Console."""
        return self._console

    def configure(self, quiet: bool = False, mode: str = "ondemand", suppress_all: bool = False):
        """Configure output behavior.

        Args:
            quiet: Suppress verbose output
            mode: Execution mode (ondemand or batch)
            suppress_all: Suppress ALL output (for multi-policy parallel runs)
        """
        self._quiet = quiet
        self._mode = mode
        self._suppress_all = suppress_all

    def info(self, message: str, **kwargs) -> None:
        """Print info message (blue). Suppressed in suppress_all mode."""
        if self._suppress_all:
            return
        self._console.print(f"[blue]ℹ[/blue] {message}", **kwargs)

    def success(self, message: str, **kwargs) -> None:
        """Print success message (green). Suppressed in suppress_all mode."""
        if self._suppress_all:
            return
        self._console.print(f"[green]✓[/green] {message}", **kwargs)

    def warn(self, message: str, **kwargs) -> None:
        """Print warning message (yellow). Suppressed in suppress_all mode."""
        if self._suppress_all:
            return
        self._console.print(f"[yellow]⚠[/yellow] {message}", **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Print error message (red)."""
        self._console.print(f"[red]✗[/red] {message}", **kwargs)

    def status(self, message: str, **kwargs) -> None:
        """Print status message (dim). Suppressed in quiet mode."""
        if not self._quiet:
            self._console.print(f"[dim]{message}[/dim]", **kwargs)

    def header(self, title: str, subtitle: str = "", **kwargs) -> None:
        """Print section header with optional subtitle."""
        if self._suppress_all:
            return
        self._console.print()
        self._console.rule(f"[bold]{title}[/bold]", **kwargs)
        if subtitle:
            self._console.print(f"[dim]{subtitle}[/dim]")

    def rule(self, title: str = "", **kwargs) -> None:
        """Print horizontal rule/divider."""
        if self._suppress_all:
            return
        self._console.rule(title, **kwargs)

    def print(self, message: str = "", **kwargs) -> None:
        """Plain print (delegates to Rich Console)."""
        self._console.print(message, **kwargs)

    def print_table(
        self,
        title: str,
        columns: List[str],
        rows: List[List[Any]],
        **kwargs,
    ) -> None:
        """Print formatted table."""
        if self._suppress_all:
            return
        table = Table(title=title, **kwargs)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        self._console.print(table)

    def print_config(self, config: dict[str, Any]) -> None:
        """Print run configuration as key-value pairs."""
        if self._suppress_all:
            return
        for key, value in config.items():
            self._console.print(f"  {key}: [cyan]{value}[/cyan]")

    def print_result(
        self,
        name: str,
        predicted: str,
        ground_truth: str | None,
        score: float | None = None,
        correct: bool | None = None,
    ) -> None:
        """Print a single result with verdict comparison."""
        if self._suppress_all:
            return
        mark = ""
        if correct is not None:
            mark = " [green]✓[/green]" if correct else " [red]✗[/red]"

        score_str = f" (score={score:.1f})" if score is not None else ""
        gt_str = f" | GT: {ground_truth}" if ground_truth else ""

        self._console.print(
            f"[bold]{name}[/bold]: {predicted}{score_str}{gt_str}{mark}"
        )

    def print_accuracy(self, correct: int, total: int) -> None:
        """Print accuracy summary."""
        if self._suppress_all:
            return
        accuracy = correct / total if total > 0 else 0.0
        color = "green" if accuracy >= 0.7 else "yellow" if accuracy >= 0.5 else "red"
        self._console.print()
        self._console.print(
            f"[bold]ACCURACY:[/bold] [{color}]{correct}/{total} = {accuracy:.1%}[/{color}]"
        )

    def progress(self, description: str = "Working..."):
        """Create progress context manager for long operations."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self._console,
        )

    # Mode-specific output methods

    def batch_submitted(self, batch_id: str) -> None:
        """Print batch submission confirmation."""
        self.success(f"Batch submitted: [cyan]{batch_id}[/cyan]")
        self.info("Cron job will check for completion every 5 minutes")

    def batch_status(self, batch_id: str, status: str) -> None:
        """Print batch status check result."""
        color = (
            "green"
            if status == "completed"
            else "yellow" if status in ("in_progress", "validating") else "red"
        )
        self.print(f"Batch [cyan]{batch_id}[/cyan] status: [{color}]{status}[/{color}]")

    def batch_completed(self, batch_id: str, n_results: int) -> None:
        """Print batch completion message."""
        self.success(f"Batch [cyan]{batch_id}[/cyan] completed with {n_results} results")

    def create_multi_policy_progress(self, policies: list[str]) -> "MultiPolicyProgress":
        """Create a progress tracker for multi-policy runs.

        Args:
            policies: List of policy IDs to run

        Returns:
            MultiPolicyProgress context manager
        """
        return MultiPolicyProgress(policies, self._console)


class MultiPolicyProgress:
    """Progress tracker for multi-policy experiment runs.

    Shows live progress with:
    - Overall progress bar (N/M policies complete)
    - Current running policy with spinner
    - Completed policies with status (✓/✗)
    """

    def __init__(self, policies: list[str], console: Console):
        self.policies = policies
        self.console = console
        self.total = len(policies)
        self.completed = 0
        self.results: dict[str, dict] = {}  # policy_id -> result
        self.running: set[str] = set()

        # Create progress display
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=console,
            expand=False,
        )
        self.task_id = None

    def __enter__(self):
        self.progress.start()
        self.task_id = self.progress.add_task(
            f"[cyan]Running {self.total} policies[/cyan]",
            total=self.total,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()
        return False

    def start_policy(self, policy_id: str) -> None:
        """Mark a policy as started."""
        self.running.add(policy_id)
        self._update_description()

    def complete_policy(self, policy_id: str, result: dict) -> None:
        """Mark a policy as completed."""
        self.running.discard(policy_id)
        self.results[policy_id] = result
        self.completed += 1
        self.progress.update(self.task_id, completed=self.completed)
        self._update_description()

    def _update_description(self) -> None:
        """Update progress description with current running policies."""
        if self.running:
            running_list = list(self.running)[:3]  # Show max 3
            if len(self.running) > 3:
                running_str = ", ".join(running_list) + f" (+{len(self.running) - 3})"
            else:
                running_str = ", ".join(running_list)
            self.progress.update(
                self.task_id,
                description=f"[cyan]Running:[/cyan] {running_str}",
            )
        else:
            self.progress.update(
                self.task_id,
                description=f"[cyan]Running {self.total} policies[/cyan]",
            )

    def print_summary(self) -> None:
        """Print summary table after all policies complete."""
        self.console.print()

        # Count successes and failures
        successes = 0
        failures = 0
        for result in self.results.values():
            if isinstance(result, dict) and result.get("error"):
                failures += 1
            else:
                successes += 1

        # Print summary line
        if failures == 0:
            self.console.print(f"[green]✓[/green] All {successes} policies completed successfully")
        else:
            self.console.print(
                f"[yellow]![/yellow] {successes} succeeded, {failures} failed"
            )

        # Print compact results table
        table = Table(show_header=True, header_style="bold", expand=False)
        table.add_column("Policy", style="cyan")
        table.add_column("Status", width=8)
        table.add_column("Details", style="dim")

        for policy_id in self.policies:
            result = self.results.get(policy_id, {})
            if isinstance(result, dict) and result.get("error"):
                table.add_row(
                    policy_id,
                    "[red]✗ FAIL[/red]",
                    str(result.get("error", ""))[:50],
                )
            elif isinstance(result, dict) and result.get("quota_met"):
                table.add_row(
                    policy_id,
                    "[yellow]SKIP[/yellow]",
                    "Quota met",
                )
            elif isinstance(result, dict) and result.get("phase") == "1":
                summary = result.get("seed_summary", {})
                table.add_row(
                    policy_id,
                    "[green]✓ OK[/green]",
                    f"Phase 1: kw={summary.get('keywords', '?')} fld={summary.get('fields', '?')}",
                )
            elif isinstance(result, dict):
                accuracy = result.get("accuracy", 0)
                n = result.get("n", 0)
                table.add_row(
                    policy_id,
                    "[green]✓ OK[/green]",
                    f"n={n} acc={accuracy:.1%}" if n > 0 else "OK",
                )
            else:
                table.add_row(policy_id, "[dim]?[/dim]", str(result)[:50])

        self.console.print(table)


# Global singleton instance
output = OutputManager()
