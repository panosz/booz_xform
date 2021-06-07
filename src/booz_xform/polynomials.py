"""
Polynomial utilities
Copyright © 2021 Panagiotis Zestanakis

Part of this code is created by modicifation of the numpy library.
Copyright (c) 2005-2021, NumPy Developers.
"""
import warnings
import numpy as np
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial import polyutils as pu
from numpy.polynomial.polynomial import polyval, polyfit, polyder, polyvander


class PolyCollection:
    """
    A collection of 1D polynomials
    """
    def __init__(self, polys):
        self.polys = polys

    def __call__(self, x_i):
        """
        calclulate the collection at `xi`

        Returns:
        out, array shape = (Npoly,) or (Npoly, Nx_i)
            For scalar input, the output is a vector of size `Npoly`.
            For vector input, the output is a 2D array of shape (Npoly, Nx_i)
        """
        x_i = np.ravel(x_i)
        if np.size(x_i) == 0:
            return np.array([])

        if np.size(x_i) == 1:
            return np.concatenate([p(x_i) for p in self])

        return np.vstack([p(x_i) for p in self])

    def __iter__(self):
        return iter(self.polys)

    def deriv(self, m=1):
        polys = [p.deriv(m) for p in self]
        return type(self)(polys)

    @classmethod
    def fit(cls, x_i, y_i, deg, **kwargs):
        """
        Return a collection of M polynomials.


        Parameters:
        -----------

        x_i: array like, shape (M,)
            x-coordinates of the M sample points

        y_i: array like, shape (K, M)
            y-coordinates of the sample points. Several data sets of sample
            points sharing the same x-coordinates can be fitted at once by
            passing in a 2D-array that contains one dataset per row.

        deg : int
            Degree of the fitting polynomials.

        **kwargs:
            options to be passed to the np.Polynomial.fit method.
            Refer to numpy documentation for more details.

        """
        polys = [Poly.fit(x_i, y, deg, **kwargs) for y in y_i]
        return cls(polys)



class MultiPoly():
    window = np.array([-1, 1])
    domain = np.array([-1, 1])

    def __init__(self, coef, domain=None, window=None):
        self.coef = np.array(coef)
        if domain is not None:
            domain = np.array(domain)
            if len(domain) != 2:
                raise ValueError("Domain has wrong number of elements.")
            self.domain = domain

        if window is not None:
            window = np.array(window)
            if len(window) != 2:
                raise ValueError("Window has wrong number of elements.")
            self.window = window

    def __call__(self, arg):
        off, scl = pu.mapparms(self.domain, self.window)
        arg = off + scl*arg
        return polyval(arg, self.coef.T)

    def __iter__(self):
        return iter(self.coef)

    def __len__(self):
        return len(self.coef)

    def __repr__(self):
        coef = repr(self.coef)[6:-1]
        domain = repr(self.domain)[6:-1]
        window = repr(self.window)[6:-1]
        name = self.__class__.__name__
        return f"{name}({coef}, domain={domain}, window={window})"


    def deriv(self, m=1):
        """Differentiate.

        Return a MultiPoly that is the derivative of the current MultiPoly.

        Parameters:
        -----------
        m : non-negative int
            Find the derivative of order `m`.

        Returns
        --------
        new_poly :
            A new polynomial representing the derivative. The domain is the same
            as the domain of the differentiated polynomial.

        """
        off, scl = pu.mapparms(self.domain, self.window)
        coef = polyder(self.coef, m, scl, axis=1)
        return self.__class__(coef, self.domain, self.window)

    @classmethod
    def fit(cls, x, y, deg):
        """Least squares fit to data.

        Return a series instance that is the least squares fit to the data
        `y` sampled at `x`. The domain of the returned instance can be
        specified and this will often result in a superior fit with less
        chance of ill conditioning.

        Parameters
        ----------
        x : array_like, shape (`M`,)
            x-coordinates of the `M` sample points ``(x[i], y[i])``.

        y: array like, shape (`K`, `M`)
            y-coordinates of the sample points. Several data sets of sample
            points sharing the same x-coordinates can be fitted at once by
            passing in a 2D-array that contains one dataset per row.

        deg : int or 1-D array_like
            Degree(s) of the fitting polynomials. If `deg` is a single integer
            all terms up to and including the `deg`'th term are included in the
            fit.

        Returns
        -------
        new_series : series
            A series that represents the least squares fit to the data and
            has the domain and window specified in the call. If the
            coefficients for the unscaled and unshifted basis polynomials are
            of interest, do ``new_series.convert().coef``.

        """

        domain = pu.getdomain(x)
        window = cls.window

        xnew = pu.mapdomain(x, domain, window)
        y = np.asarray(y)

        coef = polyfit(xnew, y.T, deg).T
        return cls(coef, domain=domain, window=window)

    @classmethod
    def fit_fixed_constant_term(cls, x, y, c0, deg):
        """
        Least squares fit to data. But keep the constant terms fixed.
        Return a series instance that is the least squares fit to the data `y`
        sampled at `x`.

        Unlike `fit`, here no domain mapping is performed. Consequently, fixing
        the constant terms is equivalent to fixing the value of the returned
        polynomial at `x=0`

        Parameters
        ----------
        x : array_like, shape (`M`,)
            x-coordinates of the `M` sample points ``(x[i], y[i])``.

        y: array like, shape (`K`, `M`)
            y-coordinates of the sample points. Several data sets of sample
            points sharing the same x-coordinates can be fitted at once by
            passing in a 2D-array that contains one dataset per row.

        c0: float or array_like, shape (`K`).
            The constant coefficients of each polynomial series.

        deg : int or 1-D array_like
            Degree(s) of the fitting polynomials. If `deg` is a single integer
            all terms up to and including the `deg`'th term are included in the
            fit.
        """

        domain = [-1, 1]
        window = [-1, 1]

        c0 = np.asarray(c0)
        y = np.asarray(y)

        coefs = polyfit_fit_constant_term(x, y.T, c0, deg)

        return cls(coefs.T, domain=domain, window=window)


def _restricted_vander(x, deg):
    return polyvander(x, deg)[..., 1:]


def polyfit_fit_constant_term(x, y, c0, deg):
    """
    modified from numpy.polynomial.polyutils._fit
    Helper function used to implement the ``<type>fit`` functions.

    Calculate the polynomial coefficients by fitting to data, but keep the
    constant terms fixed.

    Parameters
    ----------
    x : array_like, shape (`M`,)
        x-coordinates of the `M` sample (data) points ``(x[i], y[i])``.
    y : array_like, shape (`M`,) or (`M`, `K`)
        y-coordinates of the sample points.  Several sets of sample points
        sharing the same x-coordinates can be (independently) fit with one
        call to `polyfit` by passing in for `y` a 2-D array that contains
        one data set per column.
    c0: float or array_like, shape (`K`).
        The constant coefficients of each polynomial series.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`'th term are included in the
        fit. For NumPy versions >= 1.11.0 a list of integers specifying the
        degrees of the terms to include may be used instead.

    Returns
    -------
    coef : ndarray, shape (`deg` + 1,) or (`deg` + 1, `K`)
        Polynomial coefficients ordered from low to high.  If `y` was 2-D,
        the coefficients in column `k` of `coef` represent the polynomial
        fit to the data in `y`'s `k`-th column.

    """
    x = np.asarray(x) + 0.0
    c0 = np.asarray(c0) + 0.0
    y = np.asarray(y) + 0.0
    deg = np.asarray(deg)

    # check arguments.
    if deg.size > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
        raise TypeError("deg must be an int")
    if deg.min() < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if len(x) != len(y):
        raise TypeError("expected x and y to have same length")
    if (y.ndim == 1 and c0.size != 1) or (y.ndim > 1 and y.shape[-1] != c0.size):
        raise TypeError("incompatible sizes of y and c0")

    lmax = deg
    order = lmax
    van = _restricted_vander(x, lmax)

    # set up the least squares matrices in transposed form
    lhs = van.T
    rhs = (y - c0).T

    rcond = len(x)*np.finfo(x.dtype).eps

    # Determine the norms of the design matrix columns.
    if issubclass(lhs.dtype.type, np.complexfloating):
        scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
    else:
        scl = np.sqrt(np.square(lhs).sum(1))
    scl[scl == 0] = 1

    # Solve the least squares problem.
    c, resids, rank, s = np.linalg.lstsq(lhs.T/scl, rhs.T, rcond)
    c = (c.T/scl).T

    # warn on rank reduction
    if rank != order:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg, pu.RankWarning, stacklevel=2)

    if c.ndim > 1:
        c = np.vstack((c0[np.newaxis, ...], c))
    else:
        c = np.append(c0, c)

    return c



if __name__ == "__main__":
    c = np.array([[0,1,2],[0,3,4]])
    p = MultiPoly(c)

