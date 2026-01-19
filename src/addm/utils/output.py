"""Centralized output for ADDM CLI.

Provides Rich-formatted console output with convenience methods.
Mode-aware: adjusts output for ondemand vs 24hrbatch execution.
"""

from typing import Any, List, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table


class OutputManager:
    """Singleton Rich Console wrapper for CLI output."""

    _instance: Optional["OutputManager"] = None
    _console: Optional[Console] = None
    _quiet: bool = False
    _mode: str = "ondemand"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._console = Console()
        return cls._instance

    @property
    def console(self) -> Console:
        """Get underlying Rich Console."""
        return self._console

    def configure(self, quiet: bool = False, mode: str = "ondemand"):
        """Configure output behavior.

        Args:
            quiet: Suppress verbose output
            mode: Execution mode (ondemand or 24hrbatch)
        """
        self._quiet = quiet
        self._mode = mode

    def info(self, message: str, **kwargs) -> None:
        """Print info message (blue)."""
        self._console.print(f"[blue]ℹ[/blue] {message}", **kwargs)

    def success(self, message: str, **kwargs) -> None:
        """Print success message (green)."""
        self._console.print(f"[green]✓[/green] {message}", **kwargs)

    def warn(self, message: str, **kwargs) -> None:
        """Print warning message (yellow)."""
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
        self._console.print()
        self._console.rule(f"[bold]{title}[/bold]", **kwargs)
        if subtitle:
            self._console.print(f"[dim]{subtitle}[/dim]")

    def rule(self, title: str = "", **kwargs) -> None:
        """Print horizontal rule/divider."""
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
        table = Table(title=title, **kwargs)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        self._console.print(table)

    def print_config(self, config: dict[str, Any]) -> None:
        """Print run configuration as key-value pairs."""
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


# Global singleton instance
output = OutputManager()
