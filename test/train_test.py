import train
import unittest


class TrainTest(unittest.TestCase):

    def test_load_training_set(self):
        c1, c2, y = train.get_sic_training_data('../test/train.json')
        self.assertEqual(3, len(c1))
        self.assertEqual(3, len(c2))
        self.assertEqual(3, len(y))
        self.assertEqual(75 * 75, len(c1[0]))
        self.assertEqual(75 * 75, len(c2[0]))
        self.assertEqual(0, y[0])