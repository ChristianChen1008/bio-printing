import unittest

from 控制代码.PathProgram import build_rectangle_circle_program


class PathProgramValidationTests(unittest.TestCase):
    def test_rectangle_circle_retract_speed_remains_integer(self):
        program = build_rectangle_circle_program()
        retract_steps = [step for step in program.steps if step.step_type == "retract"]

        self.assertEqual(len(retract_steps), 1)
        self.assertIsInstance(retract_steps[0].params["speed"], int)


if __name__ == "__main__":
    unittest.main()
