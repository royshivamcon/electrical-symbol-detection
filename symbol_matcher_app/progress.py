"""Tiny progress logger for the long-running SAM / analysis loops.

Emits lines like::

    12:04:31 [symbol_matcher] HQ-SAM 320pts: 75/320 (23%) · elapsed 18s · ETA 59s

to stdout (so they show up in the uvicorn console). A line is logged on the
first item, then every ``every`` items or ``every_secs`` seconds (whichever comes
first), and once more when the loop finishes.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """Shared INFO logger that always writes to stdout (uvicorn console)."""
    global _LOGGER
    if _LOGGER is None:
        lg = logging.getLogger("symbol_matcher")
        lg.setLevel(logging.INFO)
        if not lg.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
            lg.addHandler(h)
        lg.propagate = False  # don't double-print through the root logger
        _LOGGER = lg
    return _LOGGER


def _fmt_secs(s: float) -> str:
    s = int(round(max(0.0, s)))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class Progress:
    """Counts up to ``total`` and periodically logs percent done + ETA."""

    def __init__(self, total: int, label: str, every: int = 20, every_secs: float = 5.0):
        self.total = max(1, int(total))
        self.label = label
        self.every = max(1, int(every))
        self.every_secs = every_secs
        self.log = get_logger()
        self.start = time.time()
        self.last_emit = 0.0
        self.n = 0
        # Workers call ``update`` concurrently, so guard the counter + throttle.
        self._lock = threading.Lock()
        self.log.info("%s: starting — %d item(s)", label, self.total)

    def update(self, inc: int = 1, note: str = "") -> None:
        with self._lock:
            self.n += inc
            now = time.time()
            done = self.n >= self.total
            # throttle: skip unless it's an ``every``-th item, ``every_secs`` passed, or done
            if not done and (self.n % self.every) and (now - self.last_emit) < self.every_secs:
                return
            self.last_emit = now
            n, elapsed = self.n, now - self.start
        rate = n / elapsed if elapsed > 0 else 0.0
        eta = (self.total - n) / rate if rate > 0 else 0.0
        pct = 100.0 * n / self.total
        self.log.info(
            "%s: %d/%d (%.0f%%) · elapsed %s · ETA %s%s%s",
            self.label, n, self.total, pct,
            _fmt_secs(elapsed), _fmt_secs(eta),
            f" · {note}" if note else "",
            " · done" if done else "",
        )

    def done(self) -> None:
        elapsed = time.time() - self.start
        self.log.info(
            "%s: complete — %d/%d in %s (%.1f/s)",
            self.label, self.n, self.total, _fmt_secs(elapsed),
            self.n / elapsed if elapsed > 0 else 0.0,
        )
