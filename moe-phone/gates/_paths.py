"""
Single source of truth for where moe-phone gate scripts read and write
result artifacts. Mirrors vaers/gates/_paths.py, whose history explains why
this module exists: scripts that construct results paths themselves drift
apart and silently read stale artifacts.

Rules (CLAUDE.md section 6.6):
  - results directories are dated and write-once in spirit: a script may add
    a NEW artifact file, but must not overwrite a different day's artifact.
  - reads resolve to the NEWEST dated directory containing the file, so an
    analysis script always sees the latest committed run and says which.
  - tests never write here; they use tmp_path.
"""
import datetime
import os
import re

GATES = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(GATES)
REPO = os.path.dirname(PROJECT)
RESULTS_ROOT = os.path.join(PROJECT, "results")
_DATED = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _dated_dirs():
    if not os.path.isdir(RESULTS_ROOT):
        return []
    return sorted(d for d in os.listdir(RESULTS_ROOT)
                  if _DATED.match(d) and os.path.isdir(os.path.join(RESULTS_ROOT, d)))


def read_path(fname):
    """Newest dated results dir containing fname. Fails loudly if absent."""
    for d in reversed(_dated_dirs()):
        p = os.path.join(RESULTS_ROOT, d, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"{fname} not found under {RESULTS_ROOT}/<date>/ — run its producing "
        f"script first (see moe-phone/README.md script index)")


def write_path(fname):
    """Today's dated results dir (created if needed) for a new artifact."""
    d = os.path.join(RESULTS_ROOT, datetime.date.today().isoformat())
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)
