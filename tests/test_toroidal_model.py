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

cur_dir = Path(__file__).absolute().parent

input_file_name = 'test_files/wout_LandremanSenguptaPlunk_section5p3.nc'

INPUT_FILE = str(cur_dir / input_file_name)

NTHETA = 50
NPHI = 90

J_SURFACE = -1

BOOZ = bx.Booz_xform()
BOOZ.verbose = 0
BOOZ.read_wout(INPUT_FILE)
BOOZ.run()

THETA1D = np.linspace(0, 2 * np.pi, NTHETA)
PHI1D = np.linspace(0, 2 * np.pi / BOOZ.nfp, NPHI)
PHI, THETA = np.meshgrid(PHI1D, THETA1D)


class TestToroidalModelByMeansOfBoozMagneticField(unittest.TestCase):
    def test_B_model_calculation(self):
        B_on_surface = BOOZ.calculate_modB_boozer_on_surface(js=J_SURFACE,
                                                             phi=PHI,
                                                             theta=THETA)

        B_model = BOOZ.mod_B_model()

        s_flux = BOOZ.s_in[J_SURFACE]

        B_modelled = B_model(s_flux, THETA, PHI)

        nt.assert_allclose(B_modelled, B_on_surface, atol=1e-5, rtol=1e-5)


    def test_theta_deriv(self):

        dB_dtheta_on_surface = BOOZ.calculate_modB_boozer_deriv_on_surface(
            js=J_SURFACE,
            phi=PHI,
            theta=THETA,
            phi_order=0,
            theta_order=1)


        dB_dtheta_model = BOOZ.mod_B_model().deriv(theta_order=1)

        s_flux = BOOZ.s_in[J_SURFACE]

        dB_dtheta_modelled = dB_dtheta_model(s_flux, THETA, PHI)

        nt.assert_allclose(dB_dtheta_modelled,
                           dB_dtheta_on_surface,
                           atol=1e-5,
                           rtol=1e-5)


    def test_chaining_derivatives_in_theta(self):

        s_flux = BOOZ.s_in[J_SURFACE]

        dB2_dtheta2_model1 = BOOZ.mod_B_model().deriv(theta_order=2)
        dB2_dtheta2_model2 = BOOZ.mod_B_model().deriv(
            theta_order=1).deriv(theta_order=1)

        nt.assert_allclose(dB2_dtheta2_model1(s_flux, THETA, PHI),
                           dB2_dtheta2_model2(s_flux, THETA, PHI),
                           )


    def test_chaining_derivatives_mixed(self):

        s_flux = BOOZ.s_in[J_SURFACE]

        dB3_dall_model1 = BOOZ.mod_B_model().deriv(s_order=1,
                                                   theta_order=1,
                                                   phi_order=1,
                                                   )
        dB3_dall_model2 = BOOZ.mod_B_model().deriv(
            s_order=1).deriv(theta_order=1).deriv(phi_order=1)

        nt.assert_allclose(dB3_dall_model1(s_flux, THETA, PHI),
                           dB3_dall_model2(s_flux, THETA, PHI),
                           )


if __name__ == '__main__':

    unittest.main()
