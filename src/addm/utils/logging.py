"""Key result logging for ADDM experiments.

Captures important experiment outcomes and metrics for analysis.
Separate from output.py (display) and debug_logger.py (LLM capture).

NOTE: Per-sample results are written to item_logs/{sample_id}.json via ItemLogger.
Aggregated usage is in results.json.
"""

import logging


def setup_logging(verbose: bool = True) -> logging.Logger:
    """Legacy logging setup for backward compatibility."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger("addm")
