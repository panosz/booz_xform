"""
Fourier series models
"""
import numpy as np
from .polynomials import PolyCollection


def is_scalar_and_zero(x):
    return np.size(x) == 1 and x == 0


def _calculate_fourier_series(m, n, cos_mn_ampl, sin_mn_ampl, phi, theta):
    """
    Calculates a double Fourier series at the poloidal angles `theta` and
    toroidal angles `phi`.

    Parameters:
    -----------
        m: array, shape(Nmn,)
            The poloidal harmonic numbers

        n: array, shape(Nmn,)
            The toroidal harmonic numbers

        cos_mn_ampl, sin_mn_ampl: array, shape(Nmn,)
            The Fourier amplitudes

        phi: array,
            The toroidal positions

        theta: array,
            The poloidal positions

    Returns:
    --------
        out: array, shape(phi.shape)
            The series calculated at the given positions


    Notes:
    ------
        `phi` and `theta` must have equal shape

    """

    out = np.zeros_like(phi)

    for jmn in range(len(m)):

        m_i = m[jmn]
        n_i = n[jmn]
        angle = m_i * theta - n_i * phi

        if not is_scalar_and_zero(cos_mn_ampl):
            out += cos_mn_ampl[jmn] * np.cos(angle)
        if not is_scalar_and_zero(sin_mn_ampl):
            out += sin_mn_ampl[jmn] * np.sin(angle)

    return out


def _calculate_fourier_series_deriv(m,
                                    n,
                                    cos_mn_ampl,
                                    sin_mn_ampl,
                                    phi,
                                    theta,
                                    phi_order,
                                    theta_order,
                                    ):
    """
    Calculates the derivative of a double Fourier series at the poloidal angles
    `theta` and toroidal angles `phi`.

    Parameters:
    -----------
        m: array, shape(Nmn,)
            The poloidal harmonic numbers

        n: array, shape(Nmn,)
            The toroidal harmonic numbers

        cos_mn_ampl, sin_mn_ampl: array, shape(Nmn,)
            The Fourier amplitudes

        phi: array,
            The toroidal positions

        theta: array,
            The poloidal positions

        phi_order, theta_order: int, positive
            The order of the derivative

    Returns:
    --------
        out: array, shape(phi.shape)
            The derivative of the series calculated at the given positions


    Notes:
    ------
        `phi` and `theta` must have equal shape

    """

    r, s = int(phi_order), int(theta_order)
    tot = r + s

    preampl = m**s * (-n)**r

    d_cos_ampl = preampl * cos_mn_ampl
    d_sin_ampl = preampl * sin_mn_ampl

    if tot % 4 == 0:
        new_cos_ampl = d_cos_ampl
        new_sin_ampl = d_sin_ampl

    elif tot % 4 == 1:
        new_cos_ampl = d_sin_ampl
        new_sin_ampl = - d_cos_ampl

    elif tot % 4 == 2:
        new_cos_ampl = - d_cos_ampl
        new_sin_ampl = - d_sin_ampl

    else:  # tot % 4 == 3
        new_cos_ampl = - d_sin_ampl
        new_sin_ampl = d_cos_ampl

    return _calculate_fourier_series(m,
                                     n,
                                     new_cos_ampl,
                                     new_sin_ampl,
                                     phi,
                                     theta)


class DoubleFourierSeries():
    """
    Represents a Fourier series on a 2-torus

    Parameters:
    -----------
    m: array-like, shape = (MNterms,)
        The poloidal harmonics

    n: array-like, shape = (MNterms,)
        The toroidal harmonics

    cos_amplitudes: array-like, shape = (MNterms,)
        The cosine amplitudes

    sin_amplitudes: array-like, shape = (MNterms) or None
        The sin amplitudes.
        If None, then the class models a symmetric quantity
        with only cosine terms.
    """

    def __init__(self, m, n, cos_amplitudes, sin_amplitudes=None):
        self.m = np.array(m)
        self.n = np.array(n)
        self.cos_amplitudes = np.array(cos_amplitudes)

        if sin_amplitudes is not None:
            self.sin_amplitudes = np.array(sin_amplitudes)

        else:
            self.sin_amplitudes = None

    def __call__(self, phi, theta):
        return _calculate_fourier_series(self.m,
                                         self.n,
                                         self.cos_amplitudes,
                                         self.sin_amplitudes,
                                         phi,
                                         theta,
                                         )

    def calculate_deriv(self, phi, theta, phi_order, theta_order):
        return _calculate_fourier_series_deriv(self.m,
                                               self.n,
                                               self.cos_amplitudes,
                                               self.sin_amplitudes,
                                               phi,
                                               theta,
                                               phi_order,
                                               theta_order
                                               )


class AlwaysReturnsZero:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return 0

    @classmethod
    def deriv(cls, *args, **kwargs):
        return cls()



class ToroidalModel():
    """
    Represents a toroidal quantity as a collection of double fourier series in
    embedeed tori.

    Parameters:
    -----------
    m: array-like
    n:
    """


    def __init__(self, m, n, cos_ampls_model, sin_ampls_model):

        self.m = np.array(m)
        self.n = np.array(n)
        self.cos_ampls_model = cos_ampls_model
        self.sin_ampls_model = sin_ampls_model


    @classmethod
    def fit(cls, s_in, m, n, cos_amplitudes, sin_amplitudes=0, deg=7):
        m = np.array(m)
        n = np.array(n)
        cos_ampls_model = PolyCollection.fit(s_in, cos_amplitudes, deg)

        if not is_scalar_and_zero(sin_amplitudes):
            sin_ampls_model = PolyCollection.fit(s_in, sin_amplitudes, deg)
        else:
            sin_ampls_model = AlwaysReturnsZero()

        return cls(m, n, cos_ampls_model, sin_ampls_model)

    def __call__(self, s_in, phi, theta):
        cos_ampls = self.cos_ampls_model(s_in)
        sin_ampls = self.sin_ampls_model(s_in)
        return _calculate_fourier_series(self.m,
                                         self.n,
                                         cos_ampls,
                                         sin_ampls,
                                         phi,
                                         theta,
                                         )

    #TODO: deriv should return a new model

    #  def deriv(self, s_in, phi, theta, s_order, phi_order, theta_order):
        #  c_deriv_ampls = self.cos_ampls_model.deriv(s_order)(s_in)
        #  s_deriv_ampls = self.sin_ampls_model.deriv(s_order)(s_in)

        #  breakpoint()

        #  return _calculate_fourier_series_deriv(self.m,
                                               #  self.n,
                                               #  c_deriv_ampls,
                                               #  s_deriv_ampls,
                                               #  phi,
                                               #  theta,
                                               #  phi_order,
                                               #  theta_order,
                                               #  )






