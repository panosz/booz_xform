#!/usr/bin/env python3
"""
This module is used to test the ToroidalModel class.
Testing is performed indirectly, by comparing the calculation of the
magnetic field as performed by Booz_xform in two different waves:

1) Calculate on the stored flux surfaces directly by evaluating the
   Fourier series.

2) First return a ToroidalModel for the magnetic field and calculate
    its output at the flux value corresponding to the flux surface.

Copyright © 2021 Panagiotis Zestanakis
"""
from pathlib import Path
import unittest
import numpy as np
import numpy.testing as nt
import booz_xform as bx

CUR_DIR = Path(__file__).absolute().parent

NTHETA = 10
NPHI = 10

J_SURFACE = -1


class BaseTestToroidalModelByMeansOfBField():

    def __init__(self, *args, **kwargs):
        INPUT_FILE = str(CUR_DIR / self.input_file_name)
        self.booz = bx.Booz_xform()
        self.booz.verbose = 0
        self.booz.read_wout(INPUT_FILE)
        self.booz.run()

        theta1d = np.linspace(0, 2 * np.pi, NTHETA)
        phi1d = np.linspace(0, 2 * np.pi / self.booz.nfp, NPHI)
        self.phi, self.theta = np.meshgrid(phi1d, theta1d)
        super().__init__(*args, **kwargs)


    def test_B_model_calculation(self):
        B_on_surface = self.booz.calculate_modB_boozer_on_surface(
            js=J_SURFACE,
            phi=self.phi,
            theta=self.theta
        )

        B_model = self.booz.mod_B_model()

        psi_b = self.booz.psi_b[J_SURFACE]

        B_modelled = B_model.calculate_on_surface(psi_b, self.theta, self.phi)

        nt.assert_allclose(B_modelled, B_on_surface, atol=5e-5, rtol=5e-5)

    def test_call_and_calculate_on_surface_compatibility(self):

        B_model = self.booz.mod_B_model()

        psi_b = self.booz.psi_b[J_SURFACE]

        phi_rav = self.phi.ravel()
        theta_rav = self.theta.ravel()

        s_flux_full = np.full_like(phi_rav, fill_value=psi_b)


        B_on_surface = B_model.calculate_on_surface(psi_b, self.theta, self.phi)

        B_call_on_surface = B_model(s_flux_full, theta_rav, phi_rav).reshape(self.phi.shape)

        nt.assert_allclose(B_on_surface, B_call_on_surface)


    def test_call_scalar_and_vector_consistency(self):

        B_model = self.booz.mod_B_model()
        psi_b = self.booz.psi_b[J_SURFACE]

        phi = self.phi[0, 5]
        theta = self.theta[0, 8]

        B_scalar = B_model(psi_b, theta, phi)

        B_vector = B_model(np.repeat(psi_b, 3),
                           np.repeat(theta, 3),
                           np.repeat(phi, 3),
                           )

        nt.assert_allclose(np.repeat(B_scalar, 3), B_vector)


    def test_theta_deriv(self):

        dB_dtheta_on_surf = self.booz.calculate_modB_boozer_deriv_on_surface(
            js=J_SURFACE,
            phi=self.phi,
            theta=self.theta,
            phi_order=0,
            theta_order=1)


        dB_dtheta_model = self.booz.mod_B_model().deriv(theta_order=1)

        psi_b = self.booz.psi_b[J_SURFACE]

        dB_dtheta_modelled = dB_dtheta_model.calculate_on_surface(psi_b,
                                                                  self.theta,
                                                                  self.phi)

        nt.assert_allclose(dB_dtheta_modelled,
                           dB_dtheta_on_surf,
                           atol=5e-5,
                           rtol=5e-5)


    def test_chaining_derivatives_in_theta(self):

        psi_b = self.booz.psi_b[J_SURFACE]

        dB2_dtheta2_model1 = self.booz.mod_B_model().deriv(theta_order=2)
        dB2_dtheta2_model2 = self.booz.mod_B_model().deriv(
            theta_order=1).deriv(theta_order=1)

        nt.assert_allclose(dB2_dtheta2_model1.calculate_on_surface(psi_b,
                                                                   self.theta,
                                                                   self.phi),
                           dB2_dtheta2_model2.calculate_on_surface(psi_b,
                                                                   self.theta,
                                                                   self.phi),
                           )


    def test_chaining_derivatives_mixed(self):

        psi_b = self.booz.psi_b[J_SURFACE]

        dB3_dall_model1 = self.booz.mod_B_model().deriv(r_order=1,
                                                        theta_order=1,
                                                        phi_order=1,
                                                        )
        dB3_dall_model2 = self.booz.mod_B_model().deriv(
            r_order=1).deriv(theta_order=1).deriv(phi_order=1)

        nt.assert_allclose(dB3_dall_model1.calculate_on_surface(psi_b,
                                                                self.theta,
                                                                self.phi),
                           dB3_dall_model2.calculate_on_surface(psi_b,
                                                                self.theta,
                                                                self.phi),
                           )


class TestWithLandreman(BaseTestToroidalModelByMeansOfBField,
                        unittest.TestCase):
    input_file_name = './test_files/wout_LandremanSenguptaPlunk_section5p3.nc'


class TestWithUpDownAssymTokamak(BaseTestToroidalModelByMeansOfBField,
                                 unittest.TestCase):
    input_file_name = './test_files/wout_up_down_asymmetric_tokamak.nc'


class TestWithCircularTokamak(BaseTestToroidalModelByMeansOfBField,
                              unittest.TestCase):
    input_file_name = './test_files/wout_circular_tokamak.nc'


class TestWithLi383(BaseTestToroidalModelByMeansOfBField,
                    unittest.TestCase):
    input_file_name = './test_files/wout_li383_1.4m.nc'


if __name__ == '__main__':

    unittest.main()
