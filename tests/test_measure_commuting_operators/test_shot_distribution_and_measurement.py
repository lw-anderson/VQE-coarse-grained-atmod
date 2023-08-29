from unittest import TestCase

from measure_commuting_operators.shot_distribution_and_measurements import shot_distribution_and_measurements


class TestShotDistributionAndMeasurement(TestCase):
    def test_expected_shot_distribution(self):
        grouped_pairs = [[((0, 0), 3.), ((1, 1), 4.)], [((3, 3), 4.)]]
        shots_and_measurements = shot_distribution_and_measurements(grouped_pairs, total_shots=100, use=True)
        self.assertEqual(len(shots_and_measurements), 2)
        self.assertEqual(sum(shots for (shots, _) in shots_and_measurements), 100)
        self.assertEqual(shots_and_measurements[0][0], shots_and_measurements[1][0])
        self.assertEqual(shots_and_measurements[0][1], (1, 1))
        self.assertEqual(shots_and_measurements[1][1], (3, 3))
