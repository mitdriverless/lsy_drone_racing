"""Per-step flight log for the controls tutorial.

Every tutorial controller calls ``History.log(...)`` once per control step with at least the
fields in ``REQUIRED``. Extra fields (e.g. ``pos_error=...``, ``i_error=...``) are allowed and are
saved alongside. At the end of an episode, call ``save()`` and inspect the result with
``python scripts/plot_log.py <file>.npz``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# tutorial_utils/ -> control/ -> lsy_drone_racing/ -> repo root
LOG_DIR = Path(__file__).resolve().parents[3] / "logs"

REQUIRED = ("t", "pos", "vel", "quat", "target", "action")


class History:
    """Growing list of per-step records that can be saved as an ``.npz`` file."""

    def __init__(self):
        """Create an empty history."""
        self.reset()

    def reset(self):
        """Clear all logged steps."""
        self._rows: list[dict[str, np.ndarray]] = []

    def log(self, **fields):
        """Append one step.

        Required keyword arguments: t, pos, vel, quat, target, action. Any additional keyword
        arguments are stored too, but every step must log the same set of fields.
        """
        missing = [k for k in REQUIRED if k not in fields]
        if missing:
            raise KeyError(f"History.log() is missing required fields: {missing}")
        if self._rows and fields.keys() != self._rows[0].keys():
            raise KeyError(
                "History.log() must receive the same fields every step. "
                f"First step had {sorted(self._rows[0])}, this step has {sorted(fields)}."
            )
        self._rows.append({k: np.array(v, dtype=float, copy=True) for k, v in fields.items()})

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Return the log as a dict of stacked arrays, one row per step."""
        if not self._rows:
            return {}
        return {k: np.stack([row[k] for row in self._rows]) for k in self._rows[0]}

    def save(self, path: str | Path) -> Path | None:
        """Save the log as .npz. Relative paths go in <repo>/logs/. Does nothing if empty."""
        if not self._rows:
            return None
        path = Path(path)
        if not path.is_absolute():
            path = LOG_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.as_arrays())
        return path

    def __len__(self) -> int:
        """Number of logged steps."""
        return len(self._rows)