# Shared data-directory path resolution: the single DATA_DIR root, plus
# resolving a short user name (e.g. "adeeb") to their raw-export folder
# (e.g. data/adeeb-2026-09-08/), used by every per-user pipeline script.

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def resolve_user_dir(user: str) -> Path:
    """
    Resolve `user` to a data/ folder. An exact existing folder name (e.g.
    "adeeb-2026-09-08") is used as-is -- an explicit override for targeting
    a specific dated snapshot (e.g. a manually held-out train/test split)
    instead of the latest. Otherwise `user` is treated as a short name (e.g.
    "adeeb") and resolved to its most recent data/<user>-<date>/ folder,
    since YYYY-MM-DD names sort chronologically as strings.
    """
    exact = DATA_DIR / user
    if exact.is_dir():
        return exact
    matches = sorted(p for p in DATA_DIR.glob(f"{user}-*") if p.is_dir())
    if not matches:
        raise SystemExit(f"No data folder found for user {user!r} (expected data/{user}-YYYY-MM-DD/)")
    return matches[-1]
