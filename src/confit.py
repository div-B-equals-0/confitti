"""Fit conic section curves to data."""

import numpy as np
import lmfit
from scipy.stats import circmean


def residual(pars, x, y, eps=None):
    # unpack parameters: extract .value attribute for each parameter
    parvals = pars.valuesdict()
    x0 = parvals["x0"]
    y0 = parvals["y0"]
    r0 = parvals["r0"]
    theta0 = parvals["theta0"]
    cth0 = np.cos(np.deg2rad(theta0))
    sth0 = np.sin(np.deg2rad(theta0))
    eccentricity = parvals["eccentricity"]
    # radius from focus
    r = np.hypot(x - x0, y - y0)
    # distance from directrix
    d = (x - x0) * cth0 + (y - y0) * sth0 - 2 * r0
    # return the residuals from the conic section equation: r = e * d
    return (r - eccentricity * d) / (1 if eps is None else eps)


def init_conic_from_xy(xdata, ydata):
    """Initialize a conic section curve from discrete (x, y) data points."""
    # Focus is initialized to be median position of the data points
    x0 = np.median(xdata)
    y0 = np.median(ydata)
    # Scale is initialized to be bottom quartile radius of data points
    # from the focus
    r = np.hypot(xdata - x0, ydata - y0)
    th = np.arctan2(ydata - y0, xdata - x0)
    r0 = np.percentile(r, 25)
    # Angle is initialized to be the circular mean of angles of those
    # data points with radius from focus less than r0
    theta0 = np.rad2deg(circmean(th[r < r0]))

    # Note that theta0 is in degrees, not radians
    # Eccentricity is initialized to be 1.0, which is a parabola

    # Return a dict, which can be unpacked as keyword arguments to
    # lmfit.create_params
    return {
        "x0": x0,
        "y0": y0,
        "r0": r0,
        "theta0": theta0,
        "eccentricity": 1.0,
    }


def fit_conic_to_xy(xdata, ydata, eps_data=None, only_parabola=True):
    """Fit a conic section curve to discrete (x, y) data points."""
    # create a set of Parameters with initial values
    params = lmfit.create_params(**init_conic_from_xy(xdata, ydata))
    # Set limits on parameters
    params["r0"].set(min=0.0)
    params["theta0"].set(min=0.0, max=360.0)
    params["eccentricity"].set(min=0.0)
    if only_parabola:
        params["eccentricity"].set(vary=False)

    # do the fit
    result = lmfit.minimize(
        residual, params, args=(xdata, ydata, eps_data), method="leastsq"
    )
    return result
