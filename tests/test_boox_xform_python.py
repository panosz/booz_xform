#!/usr/bin/env python3
"""
This module is used to test the overriding methods of the Booz_xform class.
Copyright © 2021 Panagiotis Zestanakis
"""
from pathlib import Path
import unittest
import numpy as np
import numpy.testing as nt
import netCDF4 as nc
import booz_xform as bx

CUR_DIR = Path(__file__).absolute().parent


class BaseTestBoozXform():

    def __init__(self, *args, **kwargs):
        self.INPUT_FILE = str(CUR_DIR / self.input_file_name)
        self.booz = bx.Booz_xform()
        self.booz.verbose = 0
        self.booz.read_wout(self.INPUT_FILE)
        self.booz.run()

        super().__init__(*args, **kwargs)


    def test_consistent_psi_in(self):
        """
        Test that the psi_in attribute is consistent with the s_in attribute
        """
        psi_in = self.booz.psi_in
        psi_lcfs = self.booz.psi_lcfs

        psi_in_computed = self.booz.s_in * psi_lcfs

        nt.assert_allclose(psi_in, psi_in_computed, atol=1e-10, rtol=1e-10)


    def test_consistent_psi_p(self):
        """
        Test that the psi_p model is consistent with the
        psi_p samples that are saved in the `wout` file.

        Note:
        -----

        Due to some COCOS convention the psi_p model and the vmec samples
        differ by a sing inversion.
        """

        ds = nc.Dataset(self.INPUT_FILE)
        psi_p_sampled = ds.variables['chi'][:].data

        n_full_grid = len(self.booz.s_in)+1
        s_full_grid = np.linspace(0, 1, num=n_full_grid)
        psi_full_grid = s_full_grid * self.booz.psi_lcfs

        psi_p_modelled = self.booz.psi_p(psi_full_grid)

        nt.assert_allclose(psi_p_modelled, - psi_p_sampled, atol=1e-5, rtol=1e-3)



class TestWithLandreman(BaseTestBoozXform,
                        unittest.TestCase):
    input_file_name = './test_files/wout_LandremanSenguptaPlunk_section5p3.nc'


class TestWithUpDownAssymTokamak(BaseTestBoozXform,
                                 unittest.TestCase):
    input_file_name = './test_files/wout_up_down_asymmetric_tokamak.nc'


class TestWithCircularTokamak(BaseTestBoozXform,
                              unittest.TestCase):
    input_file_name = './test_files/wout_circular_tokamak.nc'


class TestWithLi383(BaseTestBoozXform,
                    unittest.TestCase):
    input_file_name = './test_files/wout_li383_1.4m.nc'


if __name__ == '__main__':

    unittest.main()
