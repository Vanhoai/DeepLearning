import sys
import os

src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src)

import unittest
import numpy as np


class TestNeuralNetworkV3(unittest.TestCase):
    pass  # Placeholder for future tests


if __name__ == "__main__":
    unittest.main()
