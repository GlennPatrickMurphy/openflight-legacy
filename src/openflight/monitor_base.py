"""Shared base class for launch monitors.

Both ``RollingBufferMonitor`` and ``LegacySpeedMonitor`` share the same
public surface that the Flask server consumes (callbacks, lifecycle,
shot bookkeeping, session stats, club tracking). The hardware-specific
parts — how the radar is configured, how shots are detected — differ.

This module captures the shared bookkeeping in ``BaseMonitor`` so the
two implementations can stay thin and behave identically on the bits
the server depends on.
"""

from __future__ import annotations

import statistics
import threading
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from .launch_monitor import ClubType, Shot
from .ops243 import SpeedReading

ShotCallback = Callable[[Shot], None]
LiveCallback = Callable[[SpeedReading], None]
DiagnosticCallback = Callable[[dict], None]


class BaseMonitor(ABC):
    """Common bookkeeping and lifecycle for launch monitors.

    Subclasses implement ``connect``, ``disconnect``, ``get_radar_info``,
    and the capture loop (``_start_capture`` / ``_stop_capture``).
    They append detected shots via ``_emit_shot`` so the base class can
    track the session.
    """

    #: Mode label, surfaced in ``get_session_stats`` and shot records.
    mode: str = "base"

    def __init__(self) -> None:
        self._running: bool = False
        self._shots: List[Shot] = []
        self._current_club: ClubType = ClubType.DRIVER
        self._shot_callback: Optional[ShotCallback] = None
        self._live_callback: Optional[LiveCallback] = None
        self._diagnostic_callback: Optional[DiagnosticCallback] = None
        # Subclasses with background work store their thread here so
        # ``stop`` can join uniformly.
        self._capture_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the radar and apply any one-time configuration."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the radar. Should call ``stop()`` first."""

    @abstractmethod
    def get_radar_info(self) -> dict:
        """Return the radar's identification info (Product, Version, ...)."""

    @abstractmethod
    def _start_capture(self) -> None:
        """Begin background capture. Called by ``start`` after callbacks
        are wired. Typical implementation spawns a daemon thread that
        feeds shots into ``_emit_shot``."""

    def _stop_capture(self) -> None:
        """Stop background capture. Default joins ``_capture_thread`` if
        the subclass populated it. Override for more bespoke shutdown."""
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
            self._capture_thread = None

    # ------------------------------------------------------------------
    # Public surface used by the server
    # ------------------------------------------------------------------

    def start(
        self,
        shot_callback: Optional[ShotCallback] = None,
        live_callback: Optional[LiveCallback] = None,
        diagnostic_callback: Optional[DiagnosticCallback] = None,
    ) -> None:
        """Begin monitoring. Callbacks are invoked from the capture thread."""
        self._shot_callback = shot_callback
        self._live_callback = live_callback
        self._diagnostic_callback = diagnostic_callback
        self._running = True
        self._start_capture()

    def stop(self) -> None:
        """Stop monitoring. Safe to call multiple times."""
        self._running = False
        self._stop_capture()

    def get_shots(self) -> List[Shot]:
        """Return a *copy* of the session's detected shots."""
        return self._shots.copy()

    def clear_session(self) -> None:
        """Drop all recorded shots."""
        self._shots = []

    def set_club(self, club: ClubType) -> None:
        """Set the club for future shots."""
        self._current_club = club

    def get_session_stats(self) -> dict:
        """Return summary stats for the session. Mode-specific subclasses
        may override or extend with extra fields (e.g. spin metrics)."""
        if not self._shots:
            return {
                "shot_count": 0,
                "avg_ball_speed": 0,
                "max_ball_speed": 0,
                "min_ball_speed": 0,
                "avg_club_speed": None,
                "avg_smash_factor": None,
                "avg_carry_est": 0,
                "mode": self.mode,
            }

        ball_speeds = [s.ball_speed_mph for s in self._shots]
        club_speeds = [s.club_speed_mph for s in self._shots if s.club_speed_mph]
        smash_factors = [s.smash_factor for s in self._shots if s.smash_factor]

        return {
            "shot_count": len(self._shots),
            "avg_ball_speed": statistics.mean(ball_speeds),
            "max_ball_speed": max(ball_speeds),
            "min_ball_speed": min(ball_speeds),
            "std_dev": statistics.stdev(ball_speeds) if len(ball_speeds) > 1 else 0,
            "avg_club_speed": statistics.mean(club_speeds) if club_speeds else None,
            "avg_smash_factor": statistics.mean(smash_factors) if smash_factors else None,
            "avg_carry_est": statistics.mean([s.estimated_carry_yards for s in self._shots]),
            "mode": self.mode,
        }

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def _emit_shot(self, shot: Shot) -> None:
        """Record a shot and invoke the shot callback (if any).

        Subclasses use this instead of touching ``_shots`` /
        ``_shot_callback`` directly, so all delivery passes through one
        path.
        """
        self._shots.append(shot)
        if self._shot_callback is not None:
            self._shot_callback(shot)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseMonitor":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.disconnect()
        return False
