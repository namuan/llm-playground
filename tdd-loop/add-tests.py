import unittest

from add import add_numbers


class TestAddNumbersFunction(unittest.TestCase):
    def test_positive_numbers(self):
        self.assertEqual(add_numbers(5, 7), 12)

    def test_negative_numbers(self):
        self.assertEqual(add_numbers(-3, -9), -12)

    def test_zero(self):
        self.assertEqual(add_numbers(0, 10), 10)
        self.assertEqual(add_numbers(1, 0), 1)

    def test_mixed_positive_and_negative(self):
        self.assertEqual(add_numbers(-2, 5), 3)
        self.assertEqual(add_numbers(3, -4), -1)


if __name__ == "__main__":
    unittest.main()
