import numpy as np
from ._booz_xform import Booz_xform as Booz_xform_cpp
from .fourier_series import _calculate_fourier_series, DoubleFourierSeries, ToroidalModel


class Booz_xform(Booz_xform_cpp):

    def extrapolate_on_axis_bmnc_b(self):
        non_zero = (self.xm_b != 1)
        s0, s1 = self.s_b[:2]

        y0 = self.bmnc_b[:, 0]
        y1 = self.bmnc_b[:, 1]
        out = np.zeros_like(y0)
        out[non_zero] = (s1*y0[non_zero] - s0*y1[non_zero])/(s1 - s0)

        return out

    def extrapolate_on_axis_bmns_b(self):
        non_zero = (self.xm_b != 1)
        s0, s1 = self.s_b[:2]

        y0 = self.bmns_b[:, 0]
        y1 = self.bmns_b[:, 1]
        out = np.zeros_like(y0)
        out[non_zero] = (s1*y0[non_zero] - s0*y1[non_zero])/(s1 - s0)
        return out


    def mod_B_model(self):
        if self.asym:
            sin_ampls = self.bmns_b
            sin_ampls_on_axis = self.extrapolate_on_axis_bmns_b()
        else:
            sin_ampls = 0
            sin_ampls_on_axis = None

        cos_ampls = self.bmnc_b
        cos_ampls_on_axis = self.extrapolate_on_axis_bmnc_b()

        return ToroidalModel.fit_fixed_on_axis(self.s_b,
                                               self.xm_b,
                                               self.xn_b,
                                               cos_ampls,
                                               cos_ampls_on_axis,
                                               sin_ampls,
                                               sin_ampls_on_axis,
                                               deg=15
                                               )

    def calculate_modB_boozer_on_surface(self, js, theta, phi):
        """
        Calculates :math:`|B|` on a surface in Boozer poloidal and toroidal
        angles.
        Args:
          js (int): The index among the output surfaces.
          phi (array-like): The toroidal angle values.
          theta (array-like): The poloidal angle values.
        """

        phi = np.asanyarray(phi, dtype=float)
        theta = np.asanyarray(theta, dtype=float)

        cos_ampl = self.bmnc_b[:, js]

        if self.asym:
            sin_ampl = self.bmns_b[:, js]
        else:
            sin_ampl = 0


        fs = DoubleFourierSeries(self.xm_b,
                                 self.xn_b,
                                 cos_ampl,
                                 sin_ampl,
                                 )
        return fs(theta, phi)


    def calculate_modB_boozer_deriv_on_surface(self, js, theta, phi, theta_order, phi_order):
        """
        Calculates :math:`|B|` on a surface in Boozer poloidal and toroidal
        angles.
        Args:
          js (int): The index among the output surfaces.
          theta (array-like): The poloidal angle values.
          phi (array-like): The toroidal angle values.
          theta_order (int, positive): The order of the theta derivative
          phi_order (int, positive): The order of the phi derivative
        """

        phi = np.asanyarray(phi, dtype=float)
        theta = np.asanyarray(theta, dtype=float)

        cos_ampl = self.bmnc_b[:, js]

        if self.asym:
            sin_ampl = self.bmns_b[:, js]
        else:
            sin_ampl = 0


        fs = DoubleFourierSeries(self.xm_b,
                                 self.xn_b,
                                 cos_ampl,
                                 sin_ampl,
                                 )
        return fs.calculate_deriv(theta, phi, theta_order, phi_order)

