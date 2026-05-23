"""Tests for the legacy speed-streaming monitor.

This monitor is used when the OPS243 firmware is older than 1.2.3 and
does not support rolling buffer mode (GC). It reads streaming SpeedReading
values from the radar and detects shots via an event-based state machine,
producing Shot objects with ball_speed and club_speed (but no spin / I/Q).
"""

from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from openflight.launch_monitor import ClubType, Shot
from openflight.legacy_speed_monitor import LegacySpeedMonitor, ShotDetector
from openflight.ops243 import Direction, SpeedReading


def _r(speed: float, direction: Direction = Direction.OUTBOUND) -> SpeedReading:
    """Tiny helper to build a SpeedReading for tests."""
    return SpeedReading(speed=speed, direction=direction, magnitude=None, timestamp=None)


def _drive_stream(detector: ShotDetector, stream) -> List[Shot]:
    """Feed (time, speed) pairs into the detector, return all emitted shots."""
    shots: List[Shot] = []
    for t, speed in stream:
        out = detector.feed(_r(speed), now=t)
        if out is not None:
            shots.append(out)
    return shots


class TestShotDetectorBasic:
    """The detector's most fundamental contract: in → Shot or None."""

    def test_empty_stream_emits_no_shot(self):
        detector = ShotDetector(club_provider=lambda: ClubType.DRIVER)
        assert detector.feed(_r(0.0), now=0.0) is None

    def test_all_readings_below_threshold_emit_no_shot(self):
        detector = ShotDetector(club_provider=lambda: ClubType.DRIVER)
        shots = [detector.feed(_r(s), now=t * 0.01) for t, s in enumerate([5, 10, 20, 25, 30])]
        assert all(shot is None for shot in shots)

    def test_single_above_threshold_reading_does_not_emit_immediately(self):
        """One sample isn't enough — we need silence to close the event."""
        detector = ShotDetector(club_provider=lambda: ClubType.DRIVER)
        assert detector.feed(_r(100), now=0.0) is None


class TestShotDetectorDriverShot:
    """A realistic driver shot: club approach → impact → ball departure."""

    def test_driver_shot_yields_ball_and_club_speed(self):
        detector = ShotDetector(
            club_provider=lambda: ClubType.DRIVER,
            silence_s=0.05,
        )

        # Simulated swing at ~100Hz: club ramping up, ball departing, then idle.
        stream = [
            (0.000, 5),    # idle
            (0.010, 10),
            (0.020, 25),
            (0.040, 50),   # event STARTS here (50 >= 35)
            (0.060, 75),
            (0.080, 90),   # peak club speed
            (0.100, 150),  # IMPACT — ball pops in at high speed
            (0.110, 155),  # ball peak
            (0.120, 148),  # ball decelerating
            (0.140, 140),
            (0.160, 130),
            (0.200, 100),  # ball leaving field, still above 35
            (0.250, 30),   # below 35 — start of silence
            (0.310, 5),    # silence > 0.05s, event closes
            (0.320, 0),
        ]

        shots = _drive_stream(detector, stream)

        assert len(shots) == 1, f"expected exactly one shot, got {len(shots)}"
        shot = shots[0]
        assert 145 <= shot.ball_speed_mph <= 160, f"ball_speed_mph was {shot.ball_speed_mph}"
        assert shot.club_speed_mph is not None
        assert 80 <= shot.club_speed_mph <= 95, f"club_speed_mph was {shot.club_speed_mph}"
        assert shot.club == ClubType.DRIVER
        assert shot.mode == "streaming"

    def test_shot_records_impact_timestamp(self):
        """K-LD7 correlation needs an impact_timestamp (epoch seconds)."""
        detector = ShotDetector(
            club_provider=lambda: ClubType.DRIVER,
            silence_s=0.05,
        )
        stream = [
            (10.000, 90), (10.010, 145), (10.020, 152), (10.030, 148),
            (10.060, 130), (10.100, 50), (10.160, 20), (10.220, 0),
        ]
        shots = _drive_stream(detector, stream)
        assert len(shots) == 1
        assert shots[0].impact_timestamp is not None
        # Impact should be set to roughly "now in epoch", since the
        # peak reading happened recently in monotonic time. We won't
        # pin a tighter range here because it depends on wall clock.
        import time as _time
        assert abs(shots[0].impact_timestamp - _time.time()) < 5.0


class TestShotDetectorRejection:
    """The detector must not emit shots for noise, idle radar chatter, or doubles."""

    def test_low_peak_below_min_ball_speed_rejected(self):
        """A swing that never reaches min_ball_speed is not a shot."""
        detector = ShotDetector(
            club_provider=lambda: ClubType.DRIVER,
            min_ball_speed_mph=35.0,
            silence_s=0.05,
        )
        stream = [(t * 0.01, s) for t, s in enumerate([5, 10, 20, 25, 30, 28, 20, 10, 5])]
        assert _drive_stream(detector, stream) == []

    def test_second_shot_within_cooldown_suppressed(self):
        """Two impacts in rapid succession should only produce one shot."""
        detector = ShotDetector(
            club_provider=lambda: ClubType.DRIVER,
            silence_s=0.05,
            cooldown_s=1.0,
        )
        stream = [
            # First shot
            (0.000, 90), (0.010, 150), (0.020, 155), (0.030, 145),
            (0.060, 80), (0.110, 30), (0.180, 0),
            # Second "shot" arrives 0.4s after first ended — still in cooldown
            (0.400, 88), (0.410, 148), (0.420, 152), (0.440, 130),
            (0.490, 25), (0.560, 0),
        ]
        shots = _drive_stream(detector, stream)
        assert len(shots) == 1

    def test_two_shots_with_cooldown_between_both_detected(self):
        """Shots separated by more than cooldown should both be detected."""
        detector = ShotDetector(
            club_provider=lambda: ClubType.DRIVER,
            silence_s=0.05,
            cooldown_s=0.5,
        )
        stream = [
            # First shot
            (0.000, 88), (0.010, 150), (0.020, 152), (0.060, 120),
            (0.080, 80), (0.150, 20), (0.220, 0),
            # Quiet for > cooldown_s
            (0.500, 0), (0.800, 0), (1.100, 0),
            # Second shot, well past cooldown
            (1.300, 85), (1.310, 145), (1.320, 148), (1.360, 120),
            (1.380, 80), (1.450, 20), (1.520, 0),
        ]
        shots = _drive_stream(detector, stream)
        assert len(shots) == 2

    def test_isolated_single_high_reading_rejected_as_noise(self):
        """A single high reading sandwiched between zeros is likely noise."""
        detector = ShotDetector(
            club_provider=lambda: ClubType.DRIVER,
            min_readings_in_event=2,
            silence_s=0.05,
        )
        stream = [(0.000, 0), (0.010, 0), (0.020, 150), (0.030, 0),
                  (0.080, 0), (0.140, 0)]
        assert _drive_stream(detector, stream) == []


class TestShotDetectorEventClosure:
    """The event must close cleanly via silence OR a max-duration safety net."""

    def test_max_event_duration_closes_event(self):
        """If readings never drop below threshold, max_event_s closes it anyway."""
        detector = ShotDetector(
            club_provider=lambda: ClubType.DRIVER,
            silence_s=10.0,        # absurdly long silence requirement
            max_event_s=0.2,       # but max_event triggers first
        )
        # Continuous high readings — silence would never close this
        stream = [(t * 0.01, 100 + t) for t in range(50)]  # 0.00 → 0.49s, 100→149 mph
        shots = _drive_stream(detector, stream)
        assert len(shots) == 1
        # Peak is the last reading because it's the highest, but max_event
        # closes the event around t=0.2s so we won't see all 50 readings.

    def test_silence_period_closes_event_promptly(self):
        """Event should close shortly after speed drops below threshold."""
        detector = ShotDetector(
            club_provider=lambda: ClubType.DRIVER,
            silence_s=0.05,
            max_event_s=2.0,
        )
        stream = [
            (0.000, 100), (0.010, 150), (0.020, 145),  # event
            (0.030, 10),                                # below threshold
            (0.090, 10),                                # 60ms later → silence > 50ms
        ]
        shots = _drive_stream(detector, stream)
        assert len(shots) == 1


class TestShotDetectorClubSpeed:
    """Verify club speed extraction quirks."""

    def test_club_speed_omitted_when_no_pre_impact_readings(self):
        """If we only see the ball with nothing before, club_speed should be None."""
        detector = ShotDetector(
            club_provider=lambda: ClubType.DRIVER,
            silence_s=0.05,
        )
        # Ball spike with no preceding club approach
        stream = [
            (5.000, 150), (5.010, 155), (5.020, 148),  # event
            (5.030, 20), (5.090, 0),                    # silence closes
        ]
        shots = _drive_stream(detector, stream)
        assert len(shots) == 1
        assert shots[0].ball_speed_mph >= 150
        # No clean pre-impact club approach should leave club_speed None
        assert shots[0].club_speed_mph is None

    def test_pre_impact_reading_above_ratio_excluded(self):
        """A pre-impact reading too close to ball speed is likely the ball, not club."""
        detector = ShotDetector(
            club_provider=lambda: ClubType.DRIVER,
            silence_s=0.05,
            club_max_ratio=0.85,
        )
        # 140 is too high vs ball 150 (>85% = >127.5)
        # Real club readings are 90, 100
        stream = [
            (0.000, 90), (0.010, 100), (0.020, 140),  # 140 will be excluded
            (0.030, 150),  # ball peak
            (0.040, 145), (0.080, 80), (0.140, 20), (0.200, 0),
        ]
        shots = _drive_stream(detector, stream)
        assert len(shots) == 1
        assert shots[0].ball_speed_mph >= 145
        assert shots[0].club_speed_mph is not None
        # Club speed must be below the ratio cutoff
        assert shots[0].club_speed_mph <= shots[0].ball_speed_mph * 0.85


class TestShotDetectorClubProvider:
    """The shot is tagged with whichever club was selected at impact."""

    def test_current_club_used_for_shot(self):
        current_club = {"club": ClubType.DRIVER}
        detector = ShotDetector(
            club_provider=lambda: current_club["club"],
            silence_s=0.05,
            cooldown_s=0.3,
        )

        stream1 = [(0.0, 90), (0.010, 150), (0.060, 80),
                   (0.130, 20), (0.200, 0)]
        shots = _drive_stream(detector, stream1)
        assert len(shots) == 1
        assert shots[0].club == ClubType.DRIVER

        # Switch club, then second shot
        current_club["club"] = ClubType.IRON_7
        # Quiet for cooldown
        for t in [0.5, 1.0, 1.5]:
            detector.feed(_r(0.0), now=t)
        stream2 = [(2.0, 70), (2.010, 100), (2.060, 80),
                   (2.130, 20), (2.200, 0)]
        shots2 = _drive_stream(detector, stream2)
        assert len(shots2) == 1
        assert shots2[0].club == ClubType.IRON_7


class TestShotDetectorDirection:
    """Direction enum should not exclude shots (golfer setup varies)."""

    def test_inbound_readings_still_produce_shot(self):
        detector = ShotDetector(
            club_provider=lambda: ClubType.DRIVER,
            silence_s=0.05,
        )
        stream = [(0.0, 90), (0.010, 150), (0.020, 152), (0.060, 120),
                  (0.130, 20), (0.200, 0)]
        shots: List[Shot] = []
        for t, s in stream:
            out = detector.feed(_r(s, Direction.INBOUND), now=t)
            if out is not None:
                shots.append(out)
        assert len(shots) == 1
        assert shots[0].ball_speed_mph >= 150


# ============================================================================
# LegacySpeedMonitor integration tests (with a mocked OPS243Radar)
# ============================================================================


class FakeRadar:
    """A drop-in OPS243Radar fake that lets tests inject SpeedReading sequences."""

    def __init__(self, readings: List[Optional[SpeedReading]]):
        self._queue = list(readings)
        self.connected = False
        self.disconnected = False

    def connect(self):
        self.connected = True
        return True

    def disconnect(self):
        self.disconnected = True

    def get_info(self) -> dict:
        return {"Product": "OPS243-A", "Version": "1.2.2"}

    def read_speed_nonblocking(self) -> Optional[SpeedReading]:
        if not self._queue:
            return None
        return self._queue.pop(0)

    def set_units(self, *_args, **_kwargs):
        pass


class TestLegacySpeedMonitorLifecycle:
    """Connect / start / stop / get_radar_info contract matches BaseMonitor."""

    def test_connect_and_disconnect(self):
        radar = FakeRadar([])
        monitor = LegacySpeedMonitor(radar=radar)
        assert monitor.connect() is True
        assert radar.connected
        monitor.disconnect()
        assert radar.disconnected

    def test_get_radar_info_delegates_to_radar(self):
        radar = FakeRadar([])
        monitor = LegacySpeedMonitor(radar=radar)
        monitor.connect()
        info = monitor.get_radar_info()
        assert info["Product"] == "OPS243-A"

    def test_set_club_persists(self):
        radar = FakeRadar([])
        monitor = LegacySpeedMonitor(radar=radar)
        monitor.set_club(ClubType.IRON_7)
        assert monitor._current_club == ClubType.IRON_7


class TestLegacySpeedMonitorShotEmission:
    """End-to-end: feed readings via radar fake, monitor emits Shot via callback."""

    def test_realistic_swing_emits_shot_via_callback(self):
        # A complete event ending in silence so the detector closes naturally
        readings = [
            _r(s)
            for s in [0, 5, 10, 25, 50, 75, 90, 150, 155, 148, 140, 130, 100,
                      20, 0, 0, 0, 0]
        ]
        radar = FakeRadar(readings)
        monitor = LegacySpeedMonitor(radar=radar, silence_s=0.05)
        monitor.connect()

        captured: List[Shot] = []
        monitor.start(shot_callback=lambda shot: captured.append(shot))

        # Drain the radar synchronously, advancing a fake clock 10ms per tick
        for i in range(len(readings) + 5):
            monitor._tick(now=i * 0.010)

        monitor.stop()
        assert len(captured) == 1
        assert captured[0].ball_speed_mph >= 145
        assert captured[0].mode == "streaming"

    def test_live_callback_invoked_for_every_reading(self):
        """Live readings stream through even when no shot is detected."""
        readings = [_r(s) for s in [0, 5, 10, 20]]
        radar = FakeRadar(readings)
        monitor = LegacySpeedMonitor(radar=radar)
        monitor.connect()

        seen: List[SpeedReading] = []
        monitor.start(live_callback=lambda r: seen.append(r))
        for i in range(len(readings) + 2):
            monitor._tick(now=i * 0.010)
        monitor.stop()
        assert len(seen) == len(readings)


class TestLegacySpeedMonitorSessionStats:
    """Inherited bookkeeping should still work for the legacy monitor."""

    def test_empty_stats(self):
        radar = FakeRadar([])
        monitor = LegacySpeedMonitor(radar=radar)
        stats = monitor.get_session_stats()
        assert stats["shot_count"] == 0
        assert stats["mode"] == "streaming"

    def test_get_shots_returns_copy(self):
        radar = FakeRadar([])
        monitor = LegacySpeedMonitor(radar=radar)
        monitor._shots.append(MagicMock(spec=Shot))
        shots = monitor.get_shots()
        shots.clear()
        # Original list should not be mutated by the caller
        assert len(monitor._shots) == 1
