import unittest

from monitor_core import AnomalyStateMachine, Detection, MonitorStatus, centered_zone


class MonitorCoreTests(unittest.TestCase):
    def setUp(self):
        self.zone = centered_zone(640, 480, 0.5)
        self.inside = Detection((280, 200, 360, 280), 0.9)
        self.outside = Detection((0, 0, 80, 80), 0.9)

    def test_centered_zone(self):
        self.assertEqual((self.zone.left, self.zone.top, self.zone.right, self.zone.bottom),
                         (160, 120, 480, 360))

    def test_normal_when_inside(self):
        state = AnomalyStateMachine(consecutive_frames=5)
        self.assertEqual(state.update(self.inside, self.zone), MonitorStatus.NORMAL)

    def test_warning_requires_consecutive_frames(self):
        state = AnomalyStateMachine(consecutive_frames=5)
        for _ in range(4):
            self.assertNotEqual(state.update(self.outside, self.zone), MonitorStatus.WARNING)
        self.assertEqual(state.update(self.outside, self.zone), MonitorStatus.WARNING)

    def test_missing_requires_consecutive_frames(self):
        state = AnomalyStateMachine(consecutive_frames=3)
        state.update(self.inside, self.zone)
        self.assertEqual(state.update(None, self.zone), MonitorStatus.NORMAL)
        self.assertEqual(state.update(None, self.zone), MonitorStatus.NORMAL)
        self.assertEqual(state.update(None, self.zone), MonitorStatus.NOT_DETECTED)

    def test_recovers_from_warning(self):
        state = AnomalyStateMachine(consecutive_frames=2, recovered_frames=2)
        state.update(self.outside, self.zone)
        self.assertEqual(state.update(self.outside, self.zone), MonitorStatus.WARNING)
        self.assertEqual(state.update(self.inside, self.zone), MonitorStatus.RECOVERED)
        self.assertEqual(state.update(self.inside, self.zone), MonitorStatus.NORMAL)


if __name__ == "__main__":
    unittest.main()

