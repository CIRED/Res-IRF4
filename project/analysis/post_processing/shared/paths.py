"""Path helpers for post-processing notebooks.

These functions centralize path conventions introduced during the refactor.
"""

from pathlib import Path

POST_PROCESSING_ROOT = Path(__file__).resolve().parent.parent


def runs_root(domain: str) -> Path:
    """Return the root run folder for a logical domain."""
    return POST_PROCESSING_ROOT / domain / "runs"


def data_root(domain: str) -> Path:
    """Return the data folder for a logical domain."""
    return POST_PROCESSING_ROOT / domain / "data"


def resolve_run(domain: str, run_name: str) -> Path:
    """Resolve a run folder by domain and run name."""
    return runs_root(domain) / run_name
