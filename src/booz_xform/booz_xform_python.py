import numpy as np
import netCDF4 as nc
from ._booz_xform import Booz_xform as Booz_xform_cpp
from .fourier_series import _calculate_fourier_series, DoubleFourierSeries, ToroidalModel
from .polynomials import Poly

class _FluxModelBuilder():
    """
    Factory for polynomial functions of flux quantities sampled on
    the half grid.

    Parameters:
    -----------
    s_b: array,
        The values of the boozer flux surfaces (half-grid)
    """

    def __init__(self, s_b):
        self.s_b = s_b

    def _extrapolate_flux_quantity_on_axis_half_grid(self, f_half):
        """
        Given a flux quantity series `f_half` which is sampled on the half
        grid, extrapolate it on the magnetic axis.
        """
        s0, s1 = self.s_b[:2]
        y0, y1 = f_half[:2]
        f_on_axis = (s1*y0 - s0*y1)/(s1 - s0)
        return f_on_axis

    def build(self, f_half, deg):
        """
        Given a flux quantity series `f_half` which is sampled on the half
        grid, return a polynomial fit for `f_half`

        Parameters:
        -----------
        f_half: array, shape(N)
            the flux quantity on the half grid

        deg: int, positive
            The desired degree of the polynomial model.
            This is not always equal to the actual degree of the returned
            model. See "Notes", for more information.

        Notes:
        ------
        If `deg` > `N`, then a model of degree= `N` is returned.
        """
        deg = min(deg, self.s_b.size)
        x = np.concatenate([[0], self.s_b])
        y_on_axis = self._extrapolate_flux_quantity_on_axis_half_grid(f_half)
        y = np.concatenate([[y_on_axis], f_half])
        return Poly.fit(x, y, deg=deg)


def _read_toroidal_flux_full_wout(wout_file):
    """
    Reads the toroidal flux (full grid) from a wout file.
    """
    ds = nc.Dataset(wout_file)
    return ds.variables['phi'][:].data


class Booz_xform(Booz_xform_cpp):

    degrees_for_flux_polynomials = {
        "g": 4,
        "I": 4,
        "iota": 4,
    }


    def read_wout(self, wout_file):
        wout_file = str(wout_file)
        super().read_wout(wout_file)

        psi_f = _read_toroidal_flux_full_wout(wout_file)

        psi_half = 0.5 * (psi_f[:-1] + psi_f[1:])

        self.psi_in = psi_half

        self.psi_lcfs = psi_f[-1]


    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)
        self.psi_b = self.s_b * self.psi_lcfs
        self._build_flux_models()

    def _build_boozer_flux_models(self):

        if self.s_b.size > 1:
            builder = _FluxModelBuilder(self.psi_b)

            self.g = builder.build(
                self.Boozer_G,
                deg=self.degrees_for_flux_polynomials["g"]
            )

            self.I = builder.build(
                self.Boozer_I,
                deg=self.degrees_for_flux_polynomials["I"],
            )

    def _build_input_flux_models(self):
        if self.s_in.size > 1:
            builder = _FluxModelBuilder(self.psi_in)

            self.iota_m = builder.build(
                self.iota,
                deg=self.degrees_for_flux_polynomials["iota"],
            )

    def _build_psi_p_model(self):
        R"""
        Calculate a psi_p model by integrating iota_m:

        psi_p = \int iota dpsi

        NOTE that there is an implicit COCOS choice hidden here.
        For more, see test files.
        """
        self.psi_p = self.iota_m.integ()


    def _build_flux_models(self):
        self._build_input_flux_models()
        self._build_boozer_flux_models()
        self._build_psi_p_model()


    def q(self, s):
        return 1/self.iota_m(s)


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

        return ToroidalModel.fit_fixed_on_axis(self.psi_b,
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

