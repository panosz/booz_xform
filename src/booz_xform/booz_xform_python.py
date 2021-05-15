import numpy as np
from ._booz_xform import Booz_xform as Booz_xform_cpp
from .fourier_series import _calculate_fourier_series, DoubleFourierSeries, ToroidalModel


class Booz_xform(Booz_xform_cpp):

    def mod_B_model(self):
        if self.asym:
            bmns_b = self.bmns_b
        else:
            bmns_b = None

        return ToroidalModel(self.s_in,
                             self.xm_b,
                             self.xn_b,
                             self.bmnc_b,
                             bmns_b,
                             deg=15
                             )

    def calculate_modB_boozer_on_surface(self, js, phi, theta):
        """
        Calculates :math:`|B|` on a surface in Boozer poloidal and toroidal
        angles.
        Args:
          js (int): The index among the output surfaces.
          phi (array-like): The toroidal angle values.
          theta (array-like): The poloidal angle values.
        """

        phi = np.asanyarray(phi, dtype=np.float)
        theta = np.asanyarray(theta, dtype=np.float)

        cos_ampl = self.bmnc_b[:, js]

        if self.asym:
            sin_ampl = self.bmns_b[:, js]
        else:
            sin_ampl = None


        fs = DoubleFourierSeries(self.xm_b,
                                 self.xn_b,
                                 cos_ampl,
                                 sin_ampl,
                                 )
        return fs(phi, theta)

