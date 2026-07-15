import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from 控制代码.SafePosition import build_safe_lift_path


class SafePositionTests(unittest.TestCase):
    def test_build_safe_lift_path_lifts_vertically_before_horizontal_move(self):
        path = build_safe_lift_path([120, 130, 5], safe_point=[100, 100, 50])

        self.assertEqual(path, [[120.0, 130.0, 50.0], [100.0, 100.0, 50.0]])

    def test_build_safe_lift_path_does_not_lower_before_safe_move(self):
        path = build_safe_lift_path([120, 130, 60], safe_point=[100, 100, 50])

        self.assertEqual(path, [[100.0, 100.0, 50.0]])


if __name__ == "__main__":
    unittest.main()
