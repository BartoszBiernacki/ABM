import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from disease_spread_model.config import Directories

from disease_spread_model.model.my_math_utils import get_S_and_exponents_for_sym_hist
from disease_spread_model.model.my_math_utils import int_from_hist


class TestMath(unittest.TestCase):
    
    def test_hist_preparation(self):
        """Test histogram preparation values.
        
        BINS=5 -> exponents=[0, 1, 2, 1, 0] --heights-> [1, 2, 4, 2, 1] --sum-> S=10
        """
       
        S, exponents = get_S_and_exponents_for_sym_hist(bins=5)
        
        self.assertEqual(S, 10,
                         'Preparation of `S` for histogram failed!')
        
        self.assertTrue(np.allclose(exponents, [0, 1, 2, 1, 0]),
                        'Preparation of `exponents` for histogram failed!')

    def test_hist_preparation_ValueError(self):
        """Test histogram preparation values.

        Num of bins must be >0 and even integer.
        """
        
        self.assertRaises(ValueError, get_S_and_exponents_for_sym_hist, -5)
        self.assertRaises(ValueError, get_S_and_exponents_for_sym_hist, -4)
        self.assertRaises(ValueError, get_S_and_exponents_for_sym_hist, -3.5)
        self.assertRaises(ValueError, get_S_and_exponents_for_sym_hist, -0.5)
        self.assertRaises(ValueError, get_S_and_exponents_for_sym_hist, 0)
        self.assertRaises(ValueError, get_S_and_exponents_for_sym_hist, 0.5)
        self.assertRaises(ValueError, get_S_and_exponents_for_sym_hist, 3.5)
        self.assertRaises(ValueError, get_S_and_exponents_for_sym_hist, 4)
        
    def test_get_numbers_from_my_hist_distribution(self):
        MEAN = 5
        BINS = 7
        EXPONENTS = np.array([0, 1, 2, 3, 2, 1, 0])
        S = sum([1, 2, 4, 8, 4, 2, 1])

        sample_size = 2000000
        vals = [int_from_hist(MEAN, BINS, S, EXPONENTS) for _ in range(sample_size)]

        range_passed = True
        counts_passed = True

        labels, counts = np.unique(vals, return_counts=True)

        if np.array_equal(labels, EXPONENTS + MEAN):
            range_passed = False

        for exponent, count in zip(EXPONENTS, counts):
            expected = ((2**exponent) * sample_size) / S
            if not np.allclose(expected, count, rtol=0.05):
                counts_passed = False

        plt.bar(labels, counts, align='center')
        plt.gca().set_xticks(labels)

        directory = f'{Directories.TEST_PLOT_DIR}histogram distribution test/'
        Path(directory).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{directory}mean={MEAN}_bins={BINS}.pdf")

        self.assertTrue(range_passed,
                        'Histogram bins are shifted!')

        self.assertTrue(counts_passed,
                        'Histogram counts does not math theory (5% tolerance)!')


if __name__ == '__main__':
    unittest.main()
    