"""Centralized output for ADDM CLI.

Provides Rich-formatted console output with convenience methods.
Mode-aware: adjusts output for ondemand vs batch execution.
"""

from typing import Any, Callable, Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


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

    def create_amos_progress(self, policies: list[str]) -> "AMOSProgressTracker":
        """Create an AMOS-specific progress tracker for multi-policy runs.

        Shows separate progress bars for each policy with internal step tracking:
        - Phase 1: OBSERVE → PLAN → ACT
        - Phase 2: X/Y reviews processed

        Args:
            policies: List of policy IDs to run

        Returns:
            AMOSProgressTracker context manager
        """
        return AMOSProgressTracker(policies, self._console)


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


class AMOSProgressTracker:
    """Progress tracker for AMOS multi-policy runs.

    Shows separate progress bars for each policy with internal step tracking:
    - Phase 1: OBSERVE → PLAN → ACT (→ FIX if needed)
    - Phase 2: X/Y reviews processed

    Features:
    - max_display=6 active bars, rest shown as "(+N queued)"
    - Overall progress footer showing "X/Y complete (Z%)"
    - Detailed 8-metric summary table after completion
    """

    MAX_DISPLAY = 6

    def __init__(self, policies: List[str], console: Console):
        self.policies = policies
        self.console = console
        self.total = len(policies)
        self.completed = 0
        self.results: Dict[str, Dict] = {}  # policy_id -> result

        # Policy states
        self.active: Dict[str, Dict] = {}  # policy_id -> {task_id, phase, step, progress, detail}
        self.queued: List[str] = []  # Policies waiting to be displayed
        self.finished: List[str] = []  # Completed policies

        # Progress display
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            expand=False,
        )
        self.live = None

    def __enter__(self):
        self.live = Live(self._build_display(), console=self.console, refresh_per_second=4)
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.live:
            self.live.stop()
        return False

    def _build_display(self) -> Group:
        """Build the composite display with progress bars and overall status."""
        # Build individual policy progress bars (create fresh each time for simplicity)
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            expand=False,
        )

        # Add task for each active policy (sorted for consistent display)
        for policy_id in sorted(self.active.keys()):
            state = self.active[policy_id]
            desc = self._format_description(policy_id, state)
            progress.add_task(desc, total=100, completed=state.get("progress", 0))

        # Build overall status text
        pct = (self.completed / self.total * 100) if self.total > 0 else 0
        status_parts = []

        if self.queued:
            status_parts.append(f"[dim](+{len(self.queued)} queued)[/dim]")

        status_parts.append(
            f"[bold]Overall:[/bold] {self.completed}/{self.total} complete ({pct:.0f}%)"
        )

        status_text = Text.from_markup("  ".join(status_parts))

        return Group(progress, status_text)

    def _format_description(self, policy_id: str, state: Dict) -> str:
        """Format policy progress description."""
        phase = state.get("phase", 1)
        step = state.get("step", "")
        detail = state.get("detail", "")

        if phase == 1:
            return f"[cyan]{policy_id}[/cyan] P1: {step or 'starting'}"
        else:
            return f"[cyan]{policy_id}[/cyan] P2: {detail or 'extracting'}"

    def start_policy(self, policy_id: str) -> None:
        """Mark a policy as started."""
        if len(self.active) < self.MAX_DISPLAY:
            # Add to active display
            self.active[policy_id] = {
                "phase": 1,
                "step": "starting",
                "progress": 0,
                "detail": "",
            }
        else:
            # Add to queue
            self.queued.append(policy_id)

        self._refresh()

    def update_policy(
        self,
        policy_id: str,
        phase: int,
        step: str,
        progress: float,
        detail: str = "",
    ) -> None:
        """Update progress for a policy.

        Args:
            policy_id: Policy identifier
            phase: 1 or 2
            step: Current step (e.g., "OBSERVE", "PLAN", "ACT" for P1, or "extracting" for P2)
            progress: Progress percentage (0-100)
            detail: Additional detail (e.g., "25/50 reviews")
        """
        if policy_id in self.active:
            self.active[policy_id].update({
                "phase": phase,
                "step": step,
                "progress": progress,
                "detail": detail,
            })
            self._refresh()

    def complete_policy(self, policy_id: str, result: Dict) -> None:
        """Mark a policy as completed."""
        self.results[policy_id] = result
        self.completed += 1
        self.finished.append(policy_id)

        # Remove from active
        if policy_id in self.active:
            del self.active[policy_id]

        # Promote from queue if available
        if self.queued:
            next_policy = self.queued.pop(0)
            self.active[next_policy] = {
                "phase": 1,
                "step": "starting",
                "progress": 0,
                "detail": "",
            }

        self._refresh()

    def get_callback(self, policy_id: str) -> Callable:
        """Get a callback function for AMOS to report progress.

        Returns:
            Callback with signature: callback(phase, step, progress, detail)
        """
        def callback(phase: int, step: str, progress: float, detail: str = "") -> None:
            self.update_policy(policy_id, phase, step, progress, detail)

        return callback

    def _refresh(self) -> None:
        """Refresh the display."""
        if self.live:
            self.live.update(self._build_display())

    def print_summary(self) -> None:
        """Print simple summary after all policies complete (same as MultiPolicyProgress)."""
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

    def print_detailed_summary(self, n: int = 0, k: int = 0, method: str = "amos") -> None:
        """Print detailed 8-metric summary table after all policies complete.

        Args:
            n: Number of samples per policy
            k: Context size (reviews per restaurant)
            method: Method name
        """
        self.console.print()

        # Count successes and failures
        successes = sum(1 for r in self.results.values() if not (isinstance(r, dict) and r.get("error")))
        failures = len(self.results) - successes

        if failures == 0:
            self.console.print(f"[green]✓[/green] All {successes} policies completed")
        else:
            self.console.print(f"[yellow]![/yellow] {successes} succeeded, {failures} failed")

        self.console.print()
        self.console.print(f"[bold]RESULTS:[/bold] N={n} samples, K={k} reviews, method={method}")
        self.console.print()

        # Group policies by topic
        topics: Dict[str, List[str]] = {}
        for policy_id in self.policies:
            # Extract topic: G1_allergy_V2 -> G1_allergy
            parts = policy_id.rsplit("_", 1)
            if len(parts) == 2:
                topic = parts[0]
            else:
                topic = policy_id
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(policy_id)

        # Print table for each topic
        for topic, topic_policies in topics.items():
            self._print_topic_table(topic, topic_policies)

        # Print column legend
        self.console.print()
        self.console.rule()
        self.console.print(
            "[dim]Columns: Acc=accuracy, AUPRC, F1=Macro F1, Ev.P=Evidence Precision,[/dim]"
        )
        self.console.print(
            "[dim]         Ev.R=Evidence Recall, Snippet=Snippet Validity, Judg=Judgement Accuracy,[/dim]"
        )
        self.console.print(
            "[dim]         Score=Score Accuracy (V2/V3 only), Cons.=Verdict Consistency[/dim]"
        )

        # Compute and print average accuracy
        accuracies = []
        for policy_id in self.policies:
            result = self.results.get(policy_id, {})
            if isinstance(result, dict) and not result.get("error"):
                acc = result.get("accuracy")
                if acc is not None:
                    accuracies.append(acc)

        if accuracies:
            avg_acc = sum(accuracies) / len(accuracies)
            self.console.print()
            self.console.print(
                f"[bold]Average accuracy across {len(accuracies)} policies:[/bold] {avg_acc:.1%}"
            )

    def _print_topic_table(self, topic: str, policies: List[str]) -> None:
        """Print metrics table for a single topic."""
        self.console.rule(f"[bold]{topic}[/bold]", align="left")

        table = Table(show_header=True, header_style="bold", expand=False)
        table.add_column("Variant", style="cyan", width=8)
        table.add_column("Acc", width=6, justify="right")
        table.add_column("AUPRC", width=7, justify="right")
        table.add_column("F1", width=6, justify="right")
        table.add_column("Ev.P", width=6, justify="right")
        table.add_column("Ev.R", width=6, justify="right")
        table.add_column("Snippet", width=8, justify="right")
        table.add_column("Judg", width=6, justify="right")
        table.add_column("Score", width=7, justify="right")
        table.add_column("Cons.", width=7, justify="right")

        for policy_id in sorted(policies):
            # Extract variant: G1_allergy_V2 -> V2
            parts = policy_id.rsplit("_", 1)
            variant = parts[1] if len(parts) == 2 else policy_id

            result = self.results.get(policy_id, {})
            if isinstance(result, dict) and result.get("error"):
                table.add_row(variant, "[red]ERROR[/red]", "-", "-", "-", "-", "-", "-", "-", "-")
                continue
            if isinstance(result, dict) and result.get("quota_met"):
                table.add_row(variant, "[yellow]SKIP[/yellow]", "-", "-", "-", "-", "-", "-", "-", "-")
                continue

            # Extract metrics
            eval_metrics = result.get("evaluation_metrics", {}) if isinstance(result, dict) else {}
            accuracy = result.get("accuracy") if isinstance(result, dict) else None

            def fmt_pct(val):
                return f"{val:.0%}" if val is not None else "-"

            table.add_row(
                variant,
                fmt_pct(accuracy),
                fmt_pct(eval_metrics.get("auprc")),
                fmt_pct(eval_metrics.get("macro_f1")),  # Note: may not exist yet
                fmt_pct(eval_metrics.get("evidence_precision")),
                fmt_pct(eval_metrics.get("evidence_recall")),
                fmt_pct(eval_metrics.get("snippet_validity")),
                fmt_pct(eval_metrics.get("judgement_accuracy")),
                fmt_pct(eval_metrics.get("score_accuracy")),
                fmt_pct(eval_metrics.get("verdict_consistency")),
            )

        self.console.print(table)
        self.console.print()


# Global singleton instance
output = OutputManager()
