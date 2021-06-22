"""
Fourier series models
Copyright © 2021 Panagiotis Zestanakis
"""
from copy import copy

import numpy as np
from .polynomials import MultiPoly as PolyModel


def is_scalar_and_zero(x):
    return np.size(x) == 1 and x == 0


def _calculate_fourier_series(m, n, cos_mn_ampl, sin_mn_ampl, theta, phi):
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

        theta: array,
            The poloidal positions

        phi: array,
            The toroidal positions

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
                                    theta,
                                    phi,
                                    theta_order,
                                    phi_order,
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

        theta: array,
            The poloidal positions

        phi: array,
            The toroidal positions

        theta_order, phi_order: int, positive
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
                                     theta,
                                     phi,
                                     )


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

    def __call__(self, theta, phi):
        return _calculate_fourier_series(self.m,
                                         self.n,
                                         self.cos_amplitudes,
                                         self.sin_amplitudes,
                                         theta,
                                         phi,
                                         )

    def calculate_deriv(self, theta, phi, theta_order, phi_order):
        return _calculate_fourier_series_deriv(self.m,
                                               self.n,
                                               self.cos_amplitudes,
                                               self.sin_amplitudes,
                                               theta,
                                               phi,
                                               theta_order,
                                               phi_order,
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
    m: array-like, shape = (MNterms,)
        The poloidal harmonics

    n: array-like, shape = (MNterms,)
        The toroidal harmonics
    cos_ampls_model: callable,
        The model for the cos amplitudes. See Notes

    sin_ampls_model: callable,
        The model for the sin amplitudes. See Notes

    theta_phi_deriv_orders: optional
        For internal use only. Should not be set by the user.
        The orders of the fourier derivative calculated in _call_
        Default is None.


    Notes:
    ------
    The models `cos_ampls_model` and `sin_ampls_model` are callables that
    return the cos and sin amplitudes at the given flux surface.  For a scalar
    input s_scalar `cos_ampls_model(s_scalar)` should return an array with
    shape = (MNterms,).

    """

    class _FluxSurfaceDerivOrders:
        def __init__(self, theta, phi):
            self.theta = theta
            self.phi = phi

        def __repr__(self):
            return f'({self.theta}, {self.phi})'

        def __add__(self, other):
            if other is None:
                return copy(self)

            cls = type(self)

            if isinstance(other, cls):
                theta = self.theta + other.theta
                phi = self.phi + other.phi
                return cls(theta, phi)

            return NotImplemented

        def __radd__(self, other):
            return self.__add__(other)

    def __init__(self, m, n, cos_ampls_model, sin_ampls_model, theta_phi_deriv_orders=None):

        self.m = np.array(m)
        self.n = np.array(n)
        self.cos_ampls_model = cos_ampls_model
        self.sin_ampls_model = sin_ampls_model
        self._theta_phi_deriv_orders = theta_phi_deriv_orders


    @classmethod
    def fit(cls, r, m, n, cos_amplitudes, sin_amplitudes=0, deg=7):
        m = np.array(m)
        n = np.array(n)
        cos_ampls_model = PolyModel.fit(r, cos_amplitudes, deg)

        if not is_scalar_and_zero(sin_amplitudes):
            sin_ampls_model = PolyModel.fit(r, sin_amplitudes, deg)
        else:
            sin_ampls_model = AlwaysReturnsZero()

        return cls(m, n, cos_ampls_model, sin_ampls_model)

    @classmethod
    def fit_fixed_on_axis(cls,
                          r,
                          m,
                          n,
                          cos_amplitudes,
                          cos_ampls_on_axis,
                          sin_amplitudes=0,
                          sin_ampls_on_axis=None,
                          deg=7):
        m = np.array(m)
        n = np.array(n)
        cos_ampls_model = PolyModel.fit_fixed_constant_term(r, cos_amplitudes, cos_ampls_on_axis, deg)

        if not is_scalar_and_zero(sin_amplitudes):
            sin_ampls_model = PolyModel.fit_fixed_constant_term(r, sin_amplitudes, sin_ampls_on_axis, deg)
        else:
            sin_ampls_model = AlwaysReturnsZero()

        return cls(m, n, cos_ampls_model, sin_ampls_model)

    def calculate_on_surface(self, r, theta, phi):
        """
        Calculate the model on a single surface `r` at multiple toroidal and
        poloidal angles.

        Parameters:
        -----------
        r: float,
            The surface

        theta, phi: array, shape=(N) or shape=(N, M)
            The toroidal and poloidal angles. `theta` and `phi` must have the same shape.

        Returns:
        --------

        out: array, shape=(N) or shape = (N, M)
            The model calculated at the given positions.
        """

        cos_ampls = self.cos_ampls_model(r)
        sin_ampls = self.sin_ampls_model(r)

        if self._theta_phi_deriv_orders is None:
            return _calculate_fourier_series(self.m,
                                             self.n,
                                             cos_ampls,
                                             sin_ampls,
                                             theta,
                                             phi,
                                             )

        else:
            theta_order = self._theta_phi_deriv_orders.theta
            phi_order = self._theta_phi_deriv_orders.phi
            return _calculate_fourier_series_deriv(self.m,
                                                   self.n,
                                                   cos_ampls,
                                                   sin_ampls,
                                                   theta,
                                                   phi,
                                                   theta_order,
                                                   phi_order,
                                                   )

    def _calculate_next_deriv_order(self, theta_order, phi_order):
        if theta_order > 0 or phi_order > 0:
            extra_deriv = self._FluxSurfaceDerivOrders(theta_order,
                                                       phi_order
                                                       )
        else:
            extra_deriv = None

        if self._theta_phi_deriv_orders is None and extra_deriv is None:
            return None

        return self._theta_phi_deriv_orders + extra_deriv


    def deriv(self, r_order=0, theta_order=0, phi_order=0):
        cos_deriv = self.cos_ampls_model.deriv(r_order)
        sin_deriv = self.sin_ampls_model.deriv(r_order)

        theta_phi_deriv_orders = self._calculate_next_deriv_order(theta_order,
                                                                  phi_order)

        return ToroidalModel(self.m,
                             self.n,
                             cos_deriv,
                             sin_deriv,
                             theta_phi_deriv_orders,
                             )







