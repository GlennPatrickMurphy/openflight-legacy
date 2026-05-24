"""Legacy speed-streaming monitor for OPS243 firmware <1.2.3.

When the radar firmware does not support rolling buffer mode (``GC``),
we fall back to reading streaming speed values from the radar at
~150-200 Hz and detecting shots via an event-based state machine.

Pipeline:
    OPS243 → continuous SpeedReadings → ShotDetector → Shot

What we lose vs rolling buffer:
    * No spin rate (no I/Q to analyse)
    * Club speed accuracy is ~±5 mph vs ~±1 mph
    * Shot timestamp resolution is set by the radar's report rate
      (~5-7 ms) rather than the HOST_INT edge (~10 μs)

What still works:
    * K-LD7 launch angle / club path (independent of rolling buffer)
    * Camera-based ball detection
    * Carry estimation
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

from .launch_monitor import ClubType, Shot, estimate_carry_distance
from .monitor_base import BaseMonitor
from .ops243 import OPS243Radar, SpeedReading

logger = logging.getLogger("openflight.legacy_speed_monitor")


# ---------------------------------------------------------------------------
# Shot detector — pure state machine, no I/O. Easy to test.
# ---------------------------------------------------------------------------


@dataclass
class _BufferedReading:
    speed: float
    t: float


class ShotDetector:
    """Event-based shot detector for streaming speed readings.

    The OPS243 in continuous-speed mode reports the dominant moving
    target's speed at ~150-200 Hz. During a golf swing the radar sees:

        idle  →  club approach  →  *impact*  →  ball departure  →  idle

    An "event" is a run of consecutive readings at or above
    ``min_ball_speed_mph``. We collect the event, then close it when
    either:

        * No reading at/above threshold has arrived for ``silence_s``
          seconds (the ball has left the radar's field of view), or
        * The event has lasted longer than ``max_event_s`` (safety
          net for radar quirks).

    At close time:
        ball_speed = peak speed during the event
        club_speed = max reading BEFORE the peak that is below
                     ``club_max_ratio * ball_speed`` (so we don't pick
                     the ball itself as the club).

    Use:
        detector = ShotDetector(club_provider=lambda: my_current_club)
        for reading in radar_stream:
            shot = detector.feed(reading, now=time.monotonic())
            if shot is not None:
                handle_shot(shot)
    """

    def __init__(
        self,
        club_provider: Callable[[], ClubType],
        min_ball_speed_mph: float = 35.0,
        silence_s: float = 0.05,
        max_event_s: float = 0.5,
        cooldown_s: float = 1.0,
        min_readings_in_event: int = 2,
        club_max_ratio: float = 0.85,
    ) -> None:
        self._club_provider = club_provider
        self._min_ball_speed_mph = min_ball_speed_mph
        self._silence_s = silence_s
        self._max_event_s = max_event_s
        self._cooldown_s = cooldown_s
        self._min_readings_in_event = min_readings_in_event
        self._club_max_ratio = club_max_ratio

        # Event state — None when idle.
        self._event_start_t: Optional[float] = None
        self._last_above_threshold_t: Optional[float] = None
        self._event_readings: List[_BufferedReading] = []
        # Cooldown lockout end timestamp
        self._cooldown_until: float = 0.0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the state machine. Useful for tests."""
        self._event_start_t = None
        self._last_above_threshold_t = None
        self._event_readings = []
        self._cooldown_until = 0.0

    # ------------------------------------------------------------------
    def feed(self, reading: SpeedReading, now: float) -> Optional[Shot]:
        """Process a single reading. Returns a Shot if one closes this tick.

        ``now`` is a monotonic timestamp in seconds. Tests pass any
        consistent clock; production wires ``time.monotonic()``.
        """
        speed = abs(reading.speed)
        in_cooldown = now < self._cooldown_until

        above = speed >= self._min_ball_speed_mph

        # Start an event when speed crosses threshold (and we're not in cooldown).
        if above and not in_cooldown and self._event_start_t is None:
            self._event_start_t = now

        # Accumulate readings inside an active event.
        if self._event_start_t is not None:
            self._event_readings.append(_BufferedReading(speed=speed, t=now))
            if above:
                self._last_above_threshold_t = now

            # Should we close the event?
            event_age = now - self._event_start_t
            silence = (
                self._last_above_threshold_t is not None
                and (now - self._last_above_threshold_t) >= self._silence_s
            )
            timed_out = event_age >= self._max_event_s

            if silence or timed_out:
                return self._close_event(now)

        return None

    # ------------------------------------------------------------------
    def _close_event(self, now: float) -> Optional[Shot]:
        readings = self._event_readings
        # Reset state regardless of whether we emit a shot.
        self._event_start_t = None
        self._last_above_threshold_t = None
        self._event_readings = []

        # Only consider readings actually above threshold for ball-speed.
        above = [b for b in readings if b.speed >= self._min_ball_speed_mph]
        if len(above) < self._min_readings_in_event:
            logger.debug(
                "[LEGACY] Rejecting event: only %d above-threshold readings (need %d)",
                len(above), self._min_readings_in_event,
            )
            return None

        peak = max(above, key=lambda b: b.speed)
        ball_speed = peak.speed

        # Club speed = max above-threshold reading BEFORE the peak that
        # is below club_max_ratio * ball_speed. This filters out the
        # ball itself appearing in pre-impact readings (e.g. the peak
        # might be at index 3, but index 2 is also a high ball reading).
        club_ceiling = ball_speed * self._club_max_ratio
        club_candidates = [
            b.speed for b in above
            if b.t < peak.t and b.speed <= club_ceiling
        ]
        club_speed: Optional[float] = max(club_candidates) if club_candidates else None

        # Impact timestamp in epoch seconds for K-LD7 correlation.
        impact_epoch = time.time() - (now - peak.t)

        # Enter cooldown.
        self._cooldown_until = now + self._cooldown_s

        club = self._club_provider()
        carry = estimate_carry_distance(ball_speed, club)

        logger.info(
            "[LEGACY] Shot: ball=%.1f mph, club=%s mph, club_type=%s, carry=%.0f yds",
            ball_speed,
            f"{club_speed:.1f}" if club_speed is not None else "N/A",
            club.value,
            carry,
        )

        return Shot(
            ball_speed_mph=ball_speed,
            timestamp=datetime.now(),
            impact_timestamp=impact_epoch,
            club_speed_mph=club_speed,
            club=club,
            mode="streaming",
        )


# ---------------------------------------------------------------------------
# Thin monitor wrapper that runs the detector in a background thread.
# ---------------------------------------------------------------------------


class LegacySpeedMonitor(BaseMonitor):
    """Launch monitor for OPS243 firmware <1.2.3.

    Drives a ``ShotDetector`` from a background thread that reads
    ``SpeedReading`` values via ``radar.read_speed_nonblocking()``.

    The monitor exposes a synchronous ``_tick()`` for unit tests so
    the detection pipeline can be exercised without spawning threads.
    """

    mode = "streaming"

    def __init__(
        self,
        port: Optional[str] = None,
        radar: Optional[OPS243Radar] = None,
        poll_interval_s: float = 0.005,
        **detector_kwargs,
    ) -> None:
        """Args:
            port: Serial port for the radar. Ignored if ``radar`` is given.
            radar: Pre-built radar (injectable for tests). If None we
                create an ``OPS243Radar(port=port)``.
            poll_interval_s: Sleep between non-blocking reads in the
                capture thread. The radar reports at ~150-200 Hz; 5 ms
                gives us plenty of headroom without busy-looping.
            **detector_kwargs: Forwarded to ``ShotDetector``.
        """
        super().__init__()
        self.radar = radar if radar is not None else OPS243Radar(port=port)
        self._poll_interval_s = poll_interval_s
        self._detector = ShotDetector(
            club_provider=lambda: self._current_club,
            **detector_kwargs,
        )

    # ------------------------------------------------------------------
    # BaseMonitor contract
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        self.radar.connect()
        # Force the radar back to standard speed mode (CW). The user
        # may have persisted an alternate mode (object sensor on GPIO,
        # for instance — the OPS243 1.2.2 firmware's GC command does
        # that instead of rolling buffer, so any prior --setup run
        # leaves the radar emitting {"DetectedObject":...} events
        # instead of speed values).
        if hasattr(self.radar, "serial") and self.radar.serial is not None:
            try:
                self.radar.serial.write(b"GS")
                time.sleep(0.2)
                self.radar.serial.reset_input_buffer()
                logger.info("[LEGACY] Sent GS — radar set to standard speed mode")
            except Exception:  # pylint: disable=broad-except
                logger.debug("[LEGACY] GS write failed", exc_info=True)
        # Enable JSON-wrapped output. ``read_speed_nonblocking`` only
        # parses JSON lines; without OJ the radar emits plain text
        # floats which the reader silently discards.
        try:
            self.radar.enable_json_output(True)
            logger.info("[LEGACY] Enabled JSON output (OJ)")
        except Exception:  # pylint: disable=broad-except
            logger.debug("[LEGACY] enable_json_output failed", exc_info=True)
        # Best-effort mph units; ignore on older firmwares.
        try:
            from .ops243 import SpeedUnit  # pylint: disable=import-outside-toplevel
            self.radar.set_units(SpeedUnit.MPH)
        except Exception:  # pylint: disable=broad-except
            logger.debug("[LEGACY] set_units(MPH) not supported on this firmware")
        return True

    def disconnect(self) -> None:
        self.stop()
        self.radar.disconnect()

    def get_radar_info(self) -> dict:
        return self.radar.get_info()

    def _start_capture(self) -> None:
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )
        self._capture_thread.start()
        logger.info("[LEGACY] Legacy speed monitor started")

    def stop(self) -> None:
        super().stop()
        logger.info("[LEGACY] Legacy speed monitor stopped")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Background thread body: drain readings, feed the detector."""
        while self._running:
            self._tick()
            time.sleep(self._poll_interval_s)

    def _tick(self, now: Optional[float] = None) -> None:
        """One drain-and-dispatch cycle.

        Public-ish so tests can drive the pipeline synchronously
        without threading. ``now`` lets tests supply a deterministic
        monotonic clock; production passes ``None`` and ``time.monotonic()``
        is used.
        """
        if now is None:
            now = time.monotonic()
        reading = self.radar.read_speed_nonblocking()
        if reading is None:
            return
        if self._live_callback is not None:
            self._live_callback(reading)
        shot = self._detector.feed(reading, now=now)
        if shot is not None:
            self._emit_shot(shot)
