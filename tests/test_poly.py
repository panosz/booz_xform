#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as nt
from booz_xform.polynomials import MultiPoly


RANDOM_GENERATOR = np.random.default_rng(seed=42)
X = 10 * RANDOM_GENERATOR.random(50) - 5



class TestMultiPoly(unittest.TestCase):
    R"""
        Test case: p = [
                x**2 - 2*x - 3,
                3*x**3 - x**2 + 1
                ]

    """

    def explicit_poly1(self, x):
        return x**2 - 2*x - 3

    def explicit_poly2(self, x):
        return 3*x**3 - x**2 + 1

    def explicit_polys(self, x):
        p1 = self.explicit_poly1(x)
        p2 = self.explicit_poly2(x)
        return np.stack([p1, p2], axis=0)

    def explicit_poly1_deriv_1(self, x):
        return 2*x - 2

    def explicit_poly2_deriv_1(self, x):
        return 9*x**2 - 2 * x

    def explicit_polys_deriv_1(self, x):
        p1 = self.explicit_poly1_deriv_1(x)
        p2 = self.explicit_poly2_deriv_1(x)
        return np.stack([p1, p2], axis=0)

    def explicit_poly1_deriv_2(self, x):
        return np.full_like(x, fill_value=2)

    def explicit_poly2_deriv_2(self, x):
        return 18*x - 2

    def explicit_polys_deriv_2(self, x):
        p1 = self.explicit_poly1_deriv_2(x)
        p2 = self.explicit_poly2_deriv_2(x)
        return np.stack([p1, p2], axis=0)

    def get_multi_poly(self):
        return MultiPoly([[-3, -2, 1, 0], [1, 0, -1, 3]])

    def test_single_poly_assingement(self):
        p1 = MultiPoly([-3, -2, 1])
        nt.assert_allclose(p1(X), self.explicit_poly1(X))

        p2 = MultiPoly([1, 0, -1, 3])
        nt.assert_allclose(p2(X), self.explicit_poly2(X))

    def test_multi_poly_assingement(self):
        p = self.get_multi_poly()
        nt.assert_allclose(p(X), self.explicit_polys(X))

    def test_multi_poly_deriv_1(self):
        p1 = self.get_multi_poly().deriv(1)
        nt.assert_allclose(p1(X), self.explicit_polys_deriv_1(X))

        p2 = self.get_multi_poly().deriv(2)
        nt.assert_allclose(p2(X), self.explicit_polys_deriv_2(X))

if __name__ == '__main__':
    unittest.main()
