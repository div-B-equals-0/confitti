"""Fit conic section curves to data."""

import numpy as np
import lmfit
from scipy.stats import circmean

DEBUG = False


def residual(pars, x, y, eps=None):
    """
    Objective function for minimizer: residual difference between
    radius from focus and (eccentricty times) distance from directrix
    for each data point.
    """
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
    # distance from directrix (positive for points on same side as the
    # focus)
    d = 2 * r0 - (x - x0) * cth0 - (y - y0) * sth0
    if DEBUG:
        print(f"r = {r}\nd = {d}\ne d = {eccentricity * d}")
    # return the residuals from the conic section equation: r = e * d
    return (r - eccentricity * d) / (1 if eps is None else eps)


def init_conic_from_xy(xdata, ydata):
    """Initialize a conic section curve from discrete (x, y) data points."""
    # Check that the input data is valid
    assert len(xdata) == len(ydata)
    assert len(xdata) > 4  # Need at least 5 points to fit a conic
    # Focus is initialized to be median position of the data points
    x0 = np.median(xdata)
    y0 = np.median(ydata)
    # Scale is initialized to be average radius of the closest 5 points
    r = np.hypot(xdata - x0, ydata - y0)
    th = np.arctan2(ydata - y0, xdata - x0)
    closest_points = np.argsort(r)[:5]
    r0 = np.mean(r[closest_points])
    # Angle is initialized to be the circular mean of angles of those
    # same closest points
    theta0 = np.rad2deg(circmean(th[closest_points]))

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

    # Create Minimizer object
    minner = lmfit.Minimizer(
        residual, params, fcn_args=(xdata, ydata), fcn_kws={"eps": eps_data}
    )
    # do the fit
    result = minner.minimize(method="leastsq")
    return result


class XYconic:
    """Cartesian corrdinate curve of conic section."""

    def __init__(self, x0, y0, r0, theta0, eccentricity):
        self.x0 = x0
        self.y0 = y0
        self.r0 = r0
        self.theta0 = theta0
        theta0_rad = np.deg2rad(theta0)
        self.eccentricity = eccentricity
        theta_pts = np.linspace(-np.pi, np.pi, 200)
        r_pts = (
            self.r0
            * (1 + self.eccentricity)
            / (1 + self.eccentricity * np.cos(theta_pts))
        )
        self.x_pts = self.x0 + r_pts * np.cos(theta0_rad + theta_pts)
        self.y_pts = self.y0 + r_pts * np.sin(theta0_rad + theta_pts)
        self.x_apex = self.x0 + self.r0 * np.cos(theta0_rad)
        self.y_apex = self.y0 + self.r0 * np.sin(theta0_rad)
        self.x_mirror = self.x0 + 2 * self.r0 * np.cos(theta0_rad)
        self.y_mirror = self.y0 + 2 * self.r0 * np.sin(theta0_rad)

    def __repr__(self):
        return (
            f"Conic(x0={self.x0}, y0={self.y0}, r0={self.r0}, "
            f"theta0={self.theta0}, eccentricity={self.eccentricity})"
        )

    def __str__(self):
        return (
            f"Conic section curve with focus at ({self.x0}, {self.y0}), "
            f"scale factor {self.r0}, angle {self.theta0}, and eccentricity "
            f"{self.eccentricity}."
        )
