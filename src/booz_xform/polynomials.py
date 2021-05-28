"""
Polynomial utilities
Copyright © 2021 Panagiotis Zestanakis

Part of this code is created by modicifation of the numpy library.
Copyright (c) 2005-2021, NumPy Developers.
"""

import numpy as np
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial import polyutils as pu
from numpy.polynomial.polynomial import polyval, polyfit, polyder


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

    #  @classmethod
    #  def fit(cls, x, y, deg, domain=None, rcond=None, full=False, w=None,
        #  window=None):
        #  """Least squares fit to data.

        #  Return a series instance that is the least squares fit to the data
        #  `y` sampled at `x`. The domain of the returned instance can be
        #  specified and this will often result in a superior fit with less
        #  chance of ill conditioning.

        #  Parameters
        #  ----------
        #  x : array_like, shape (M,)
            #  x-coordinates of the M sample points ``(x[i], y[i])``.
        #  y : array_like, shape (M,)
            #  y-coordinates of the M sample points ``(x[i], y[i])``.
        #  deg : int or 1-D array_like
            #  Degree(s) of the fitting polynomials. If `deg` is a single integer
            #  all terms up to and including the `deg`'th term are included in the
            #  fit. For NumPy versions >= 1.11.0 a list of integers specifying the
            #  degrees of the terms to include may be used instead.
        #  domain : {None, [beg, end], []}, optional
            #  Domain to use for the returned series. If ``None``,
            #  then a minimal domain that covers the points `x` is chosen.  If
            #  ``[]`` the class domain is used. The default value was the
            #  class domain in NumPy 1.4 and ``None`` in later versions.
            #  The ``[]`` option was added in numpy 1.5.0.
        #  rcond : float, optional
            #  Relative condition number of the fit. Singular values smaller
            #  than this relative to the largest singular value will be
            #  ignored. The default value is len(x)*eps, where eps is the
            #  relative precision of the float type, about 2e-16 in most
            #  cases.
        #  full : bool, optional
            #  Switch determining nature of return value. When it is False
            #  (the default) just the coefficients are returned, when True
            #  diagnostic information from the singular value decomposition is
            #  also returned.
        #  w : array_like, shape (M,), optional
            #  Weights. If not None the contribution of each point
            #  ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the
            #  weights are chosen so that the errors of the products
            #  ``w[i]*y[i]`` all have the same variance.  The default value is
            #  None.

            #  .. versionadded:: 1.5.0
        #  window : {[beg, end]}, optional
            #  Window to use for the returned series. The default
            #  value is the default class domain

            #  .. versionadded:: 1.6.0

        #  Returns
        #  -------
        #  new_series : series
            #  A series that represents the least squares fit to the data and
            #  has the domain and window specified in the call. If the
            #  coefficients for the unscaled and unshifted basis polynomials are
            #  of interest, do ``new_series.convert().coef``.

        #  [resid, rank, sv, rcond] : list
            #  These values are only returned if `full` = True

            #  resid -- sum of squared residuals of the least squares fit
            #  rank -- the numerical rank of the scaled Vandermonde matrix
            #  sv -- singular values of the scaled Vandermonde matrix
            #  rcond -- value of `rcond`.

            #  For more details, see `linalg.lstsq`.

        #  """
        #  if domain is None:
            #  domain = pu.getdomain(x)
        #  elif type(domain) is list and len(domain) == 0:
            #  domain = cls.domain

        #  if window is None:
            #  window = cls.window

        #  xnew = pu.mapdomain(x, domain, window)
        #  res = polyfit(xnew, y, deg, w=w, rcond=rcond, full=full)
        #  if full:
            #  [coef, status] = res
            #  return cls(coef, domain=domain, window=window), status
        #  else:
            #  coef = res
            #  return cls(coef, domain=domain, window=window)

if __name__ == "__main__":
    c = np.array([[0,1,2],[0,3,4]])
    p = MultiPoly(c)

