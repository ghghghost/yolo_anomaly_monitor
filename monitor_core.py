"""Framework-independent state logic for the YOLO work-zone monitor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class MonitorStatus(str, Enum):
    WAITING = "WAITING"
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    NOT_DETECTED = "NOT_DETECTED"
    RECOVERED = "RECOVERED"


@dataclass(frozen=True)
class Detection:
    box: Tuple[int, int, int, int]
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass(frozen=True)
class Zone:
    left: int
    top: int
    right: int
    bottom: int

    def contains(self, point: Tuple[int, int]) -> bool:
        x, y = point
        return self.left <= x <= self.right and self.top <= y <= self.bottom


def centered_zone(width: int, height: int, ratio: float = 0.5) -> Zone:
    """Return a centered rectangular zone occupying ``ratio`` of each dimension."""
    ratio = min(1.0, max(0.1, ratio))
    margin_x = int(width * (1.0 - ratio) / 2.0)
    margin_y = int(height * (1.0 - ratio) / 2.0)
    return Zone(margin_x, margin_y, width - margin_x, height - margin_y)


class AnomalyStateMachine:
    """Debounce detections so a single bad frame cannot trigger an alarm."""

    def __init__(self, consecutive_frames: int = 5, recovered_frames: int = 12):
        self.consecutive_frames = max(1, consecutive_frames)
        self.recovered_frames = max(1, recovered_frames)
        self.reset()

    def reset(self) -> None:
        self.status = MonitorStatus.WAITING
        self._outside_count = 0
        self._missing_count = 0
        self._recovered_count = 0

    def update(self, detection: Optional[Detection], zone: Zone) -> MonitorStatus:
        previous = self.status

        if detection is None:
            self._missing_count += 1
            self._outside_count = 0
            self._recovered_count = 0
            if self._missing_count >= self.consecutive_frames:
                self.status = MonitorStatus.NOT_DETECTED
            return self.status

        self._missing_count = 0
        if zone.contains(detection.center):
            self._outside_count = 0
            if previous in (MonitorStatus.WARNING, MonitorStatus.NOT_DETECTED):
                self.status = MonitorStatus.RECOVERED
                self._recovered_count = 1
            elif previous == MonitorStatus.RECOVERED:
                self._recovered_count += 1
                if self._recovered_count >= self.recovered_frames:
                    self.status = MonitorStatus.NORMAL
            else:
                self.status = MonitorStatus.NORMAL
            return self.status

        self._outside_count += 1
        self._recovered_count = 0
        if self._outside_count >= self.consecutive_frames:
            self.status = MonitorStatus.WARNING
        return self.status

