# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Check that fits work in all four quadrants
# We had a bug report from Vanessa that it did not work in the third and fourth quadrants

# ## Imports
#
# We need to explictly add the path to the library since we haven't installed it yet.

import time

start_time = time.time()
from pathlib import Path

import confitti
import numpy as np
import lmfit
from matplotlib import pyplot as plt
import seaborn as sns

# Check that we have the right module

# +
# confitti?
# -

# ## Where to save the figures

figpath = Path.cwd().parent / "figs"
saveprefix = "demo05"

# ## Test data
#
# I use the same pattern of 7 points as the basis, but we are going to rotate it through different angles to make sure that the fits still work

xpts0, ypts0 = np.array([1, 2, 3, 4, 5, 6, 7]), np.array([0, 4, 6, 7, 6, 4, 0])
ypts0 += xpts0
xpts0 *= 3


# ## Function to rotate the points
#
# Rotate anticlockwise through an angle theta in degrees


def rotate(x, y, theta):
    c = np.cos(np.deg2rad(theta))
    s = np.sin(np.deg2rad(theta))
    xx = x * c - y * s
    yy = x * s + y * c
    return xx, yy


rotate(1, 1, -90)

# Plot the points for 6 different orientations

fig, ax = plt.subplots()
for theta in np.arange(6) * 60:
    xpts, ypts = rotate(xpts0, ypts0, theta)
    ax.scatter(xpts, ypts)
ax.set_aspect("equal")
...;

fig.savefig(figpath / f"{saveprefix}-points.pdf", bbox_inches="tight")

# ## Initial guess at conic parameters
#
# This is done automatically inside the fitting function, but we will do it explicitly so we can see what it is doing

# I have added some debug print statements inside of `init_conic_from_xy()` so we can see what is going on. This led to the realization that the problem was with the use of the median for finding the initia focus position. I have switched it to the mean, which seems to have solved the problem.

theta = 210
xpts, ypts = rotate(xpts0, ypts0, theta)
confitti.DEBUG = True
initial_conic = confitti.init_conic_from_xy(xpts, ypts)
initial_conic

# Look at the residuals for this initial guess, which is
# $$
# r - e d
# $$
# where $r$ is the radius of each point from the focus and $d$ is the distance of each point from the directrix.
#
# We turn on DEBUG so that the residual function will print out the individual vectors, $r$, $d$, and $e \times d$.

initial_params = lmfit.create_params(**initial_conic)
confitti.residual(initial_params, xpts, ypts)

# The residuals are all negative, meaning points are inside the conic. This suggests that `r0` is overestimated.

# Turn the `DEBUG` flag back off.

confitti.DEBUG = False

init_xy = confitti.XYconic(**initial_conic)
print(init_xy)

# +
fig, ax = plt.subplots()
ax.scatter(xpts, ypts, color="k")

c = "C0"
ax.plot(init_xy.x_pts, init_xy.y_pts, color=c)
ax.scatter(init_xy.x0, init_xy.y0, marker="+", color=c)
ax.plot([init_xy.x0, init_xy.x_mirror], [init_xy.y0, init_xy.y_mirror], color=c)

ax.set_aspect("equal")
margin = 8
ax.set(
    xlim=[xpts.min() - margin, xpts.max() + margin],
    ylim=[ypts.min() - margin, ypts.max() + margin],
)
...;
# -
# ## Do the fitting for a range of rotations of the origina data
#
# We make a dictionary with keys of the rotation angles that holds the data points and the two fits (parabola and general conic)

results = {}
for theta in (-3.0 + np.arange(12) * 30):
    xpts, ypts = rotate(xpts0, ypts0, theta)
    result_p = confitti.fit_conic_to_xy(xpts, ypts, allow_negative_theta=True, only_parabola=True)
    result_e = confitti.fit_conic_to_xy(xpts, ypts, allow_negative_theta=True, only_parabola=False)
    results[theta] = {
        "x": xpts,
        "y": ypts,
        "pfit": result_p,
        "efit": result_e,
    }

# Note that the argument `allow_negative_theta` is true by default. It is recommended that this option should always be left turned on. It is just included here for testing purposes. If it is set to false, then one of the fits fails because the angle gets trapped at 0.0. 

#
# Look at the residuals:

[result["efit"].residual for result in results.values()]

# Those all look the same, which is good.
#
# Extract all the parameters as a dataframe

import pandas as pd

df = pd.DataFrame(
    {"angle": angle, **result["efit"].params.valuesdict()}
    for angle, result in results.items()
)
df.style.format(precision=3)

# Now look at the fitted angle minus the data rotation angle

fig, ax = plt.subplots()
ax.scatter(df["angle"], (df["theta0"] - df["angle"]) % 360)
ax.set(ylim=[0, 180])

# This is constant, which is what we hoped. This show that the fit is the same for all the different data rotation angle.

# ## Plotting the best fit onto the data

bestp_xy = confitti.XYconic(**result_p.params.valuesdict())
print(bestp_xy)

beste_xy = confitti.XYconic(**result_e.params.valuesdict())
print(beste_xy)

init_xy = confitti.XYconic(**initial_conic)
print(init_xy)

fig, ax = plt.subplots()
for theta, result in results.items():
    beste_xy = confitti.XYconic(**result["efit"].params.valuesdict())
    ax.plot(beste_xy.x_pts, beste_xy.y_pts)
    ax.scatter(result["x"], result["y"], marker=".", color="k")
ax.set_aspect("equal")
_limit = 100
ax.set(xlim=[-_limit, _limit], ylim=[-_limit, _limit])
...;

fig.savefig(figpath / f"{saveprefix}-best-fits.pdf", bbox_inches="tight")

# ## Execution time for notebook

print(f"--- {time.time() - start_time} seconds ---")
