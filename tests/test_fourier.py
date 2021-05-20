#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as nt
from booz_xform.fourier_series import DoubleFourierSeries


RANDOM_GENERATOR = np.random.default_rng(seed=42)


class TestDoubleFourierSeries(unittest.TestCase):
    R"""
        Test case:
            S = 3 cos(2 \theta - 3 \zeta) + 4 sin( 6 \theta - 2 \zeta)
    """


    def explicit_fourier_series(self, zeta, theta):
        out = 3 * np.cos(2 * theta - 3 * zeta) \
            + 4 * np.sin(6 * theta - 2 * zeta)

        return out


    def get_fourier_series(self):
        m = np.array([2, 6])
        n = np.array([3, 2])

        cos_mn_ampl = np.array([3, 0])
        sin_mn_ampl = np.array([0, 4])

        return DoubleFourierSeries(m, n, cos_mn_ampl, sin_mn_ampl)


    def test_call(self):

        fs = self.get_fourier_series()

        theta = 2*np.pi * RANDOM_GENERATOR.random(10)
        zeta = 2*np.pi * RANDOM_GENERATOR.random(10)

        expected_result = self.explicit_fourier_series(zeta, theta)

        actual = fs(zeta, theta)

        nt.assert_allclose(actual, expected_result, atol=1e-10, rtol=1e-10)






if __name__ == '__main__':
    unittest.main()
