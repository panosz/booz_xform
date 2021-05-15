"""
Fourier series models
"""
import numpy as np
from .polynomials import PolyCollection


def _calculate_fourier_series(m, n, cos_mn_ampl, sin_mn_ampl, phi, theta):

    out = np.zeros_like(phi)

    for jmn in range(len(m)):

        m_i = m[jmn]
        n_i = n[jmn]
        angle = m_i * theta - n_i * phi

        out += cos_mn_ampl[jmn] * np.cos(angle)
        if sin_mn_ampl is not None:
            out += sin_mn_ampl[jmn] * np.sin(angle)

    return out


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


class ToroidalModel():
    """
    Represents a toroidal quantity as a collection of double fourier series in
    embedeed tori.
    """

    def _return_none(self, *args, **kwargs):
        return None

    def __init__(self, s_in, m, n, cos_amplitudes, sin_amplitudes=None, deg=7):

        self.m = np.array(m)
        self.n = np.array(n)
        breakpoint()
        self.cos_ampls_model = PolyCollection.fit(s_in, cos_amplitudes, deg)

        if sin_amplitudes is not None:
            self.cos_ampls_model = PolyCollection.fit(s_in, sin_amplitudes, deg)
        else:
            self.sin_ampls_model = self._return_none

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



