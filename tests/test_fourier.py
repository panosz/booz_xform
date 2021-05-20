#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as nt
from booz_xform.fourier_series import DoubleFourierSeries


RANDOM_GENERATOR = np.random.default_rng(seed=42)
THETA = 2*np.pi * RANDOM_GENERATOR.random(10)
ZETA = 2*np.pi * RANDOM_GENERATOR.random(10)



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

    def explicit_fourier_series_deriv_0_1(self, zeta, theta):
        """
        The first derivative wrt theta
        """
        out = 24*np.cos(6*theta - 2*zeta) \
            - 6*np.sin(2*theta - 3*zeta)
        return out


    def explicit_fourier_series_deriv_1_0(self, zeta, theta):
        """
        The first derivative wrt zeta
        """
        out = -8*np.cos(6*theta - 2*zeta) \
            + 9*np.sin(2*theta - 3*zeta)
        return out

    def explicit_fourier_series_deriv_1_1(self, zeta, theta):
        R"""
        d^2 / d\zeta d\theta
        """
        out = 18*np.cos(2*theta - 3*zeta) \
            + 48*np.sin(6*theta - 2*zeta)
        return out


    def explicit_fourier_series_deriv_2_0(self, zeta, theta):
        R"""
        d^2 / d\zeta^2
        """
        out = -27*np.cos(2*theta - 3*zeta) \
            - 16*np.sin(6*theta - 2*zeta)
        return out

    def explicit_fourier_series_deriv_0_2(self, zeta, theta):
        R"""
        d^2 / d\theta^2
        """
        out = -12*np.cos(2*theta - 3*zeta) - 144*np.sin(6*theta - 2*zeta)
        return out

    def explicit_fourier_series_deriv_1_2(self, zeta, theta):
        R"""
        d^3 / d\zeta d\theta^2
        """
        out = 288*np.cos(6*theta - 2*zeta) - 36*np.sin(2*theta - 3*zeta)
        return out

    def explicit_fourier_series_deriv_3_0(self, zeta, theta):
        R"""
        d^3 / d\zeta^3
        """
        out = 32*np.cos(6*theta - 2*zeta) - 81*np.sin(2*theta - 3*zeta)
        return out

    def explicit_fourier_series_deriv_2_2(self, zeta, theta):
        R"""
        d^4 / d\zeta^2 d\theta^2
        """
        out = 108*np.cos(2*theta - 3*zeta) + 576*np.sin(6*theta - 2*zeta)
        return out

    def explicit_fourier_series_deriv_3_1(self, zeta, theta):
        R"""
        d^4 / d\zeta^3 d\theta^1
        """
        out = -162*np.cos(2*theta - 3*zeta) - 192*np.sin(6*theta - 2*zeta)
        return out

    def explicit_fourier_series_deriv_5_7(self, zeta, theta):
        R"""
        d^4 / d\zeta^3 d\theta^1
        """
        out = -93312*np.cos(2*theta - 3*zeta) \
            - 35831808*np.sin(6*theta - 2*zeta)
        return out


    def test_call(self):

        fs = self.get_fourier_series()

        expected_result = self.explicit_fourier_series(ZETA, THETA)

        actual = fs(ZETA, THETA)

        nt.assert_allclose(actual, expected_result, atol=1e-10, rtol=1e-10)


    def test_deriv_0_1(self):
        fs = self.get_fourier_series()

        expected_result = self.explicit_fourier_series_deriv_0_1(ZETA, THETA)

        actual = fs.calculate_deriv(ZETA, THETA, phi_order=0, theta_order=1)

        nt.assert_allclose(actual, expected_result, atol=1e-10, rtol=1e-10)


    def test_deriv_1_0(self):
        fs = self.get_fourier_series()

        expected_result = self.explicit_fourier_series_deriv_1_0(ZETA, THETA)

        actual = fs.calculate_deriv(ZETA, THETA, phi_order=1, theta_order=0)

        nt.assert_allclose(actual, expected_result, atol=1e-10, rtol=1e-10)


    def test_deriv_1_1(self):
        fs = self.get_fourier_series()

        expected_result = self.explicit_fourier_series_deriv_1_1(ZETA, THETA)

        actual = fs.calculate_deriv(ZETA, THETA, phi_order=1, theta_order=1)

        nt.assert_allclose(actual, expected_result, atol=1e-10, rtol=1e-10)


    def test_deriv_2_0(self):
        fs = self.get_fourier_series()

        expected_result = self.explicit_fourier_series_deriv_2_0(ZETA, THETA)

        actual = fs.calculate_deriv(ZETA, THETA, phi_order=2, theta_order=0)

        nt.assert_allclose(actual, expected_result, atol=1e-10, rtol=1e-10)


    def test_deriv_0_2(self):
        fs = self.get_fourier_series()

        expected_result = self.explicit_fourier_series_deriv_0_2(ZETA, THETA)

        actual = fs.calculate_deriv(ZETA, THETA, phi_order=0, theta_order=2)

        nt.assert_allclose(actual, expected_result, atol=1e-10, rtol=1e-10)

    def test_deriv_1_2(self):
        fs = self.get_fourier_series()

        expected_result = self.explicit_fourier_series_deriv_1_2(ZETA, THETA)

        actual = fs.calculate_deriv(ZETA, THETA, phi_order=1, theta_order=2)

        nt.assert_allclose(actual, expected_result, atol=1e-10, rtol=1e-10)

    def test_deriv_3_0(self):
        fs = self.get_fourier_series()

        expected_result = self.explicit_fourier_series_deriv_3_0(ZETA, THETA)

        actual = fs.calculate_deriv(ZETA, THETA, phi_order=3, theta_order=0)

        nt.assert_allclose(actual, expected_result, atol=1e-10, rtol=1e-10)

    def test_deriv_2_2(self):
        fs = self.get_fourier_series()

        expected_result = self.explicit_fourier_series_deriv_2_2(ZETA, THETA)

        actual = fs.calculate_deriv(ZETA, THETA, phi_order=2, theta_order=2)

        nt.assert_allclose(actual, expected_result, atol=1e-10, rtol=1e-10)

    def test_deriv_3_1(self):
        fs = self.get_fourier_series()

        expected_result = self.explicit_fourier_series_deriv_3_1(ZETA, THETA)

        actual = fs.calculate_deriv(ZETA, THETA, phi_order=3, theta_order=1)

        nt.assert_allclose(actual, expected_result, atol=1e-10, rtol=1e-10)

    def test_deriv_5_7(self):
        fs = self.get_fourier_series()

        expected_result = self.explicit_fourier_series_deriv_5_7(ZETA, THETA)

        actual = fs.calculate_deriv(ZETA, THETA, phi_order=5, theta_order=7)

        nt.assert_allclose(actual, expected_result, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
