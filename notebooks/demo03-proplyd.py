# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Fit conic to real data from proplyd arcs
#
# We will use the same data that we used in Tarango Yong & Henney (2018) to demonstrate the circle-fit algorithm.
#
#

# ## Imports

import time

start_time = time.time()
import sys
from pathlib import Path

sys.path.append("../src")
import confit
import numpy as np
import lmfit
from matplotlib import pyplot as plt
import seaborn as sns
import regions as rg
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

sns.set_context("notebook", font_scale=1.2)

# ## Set up the arc data

datapath = Path.cwd().parent / "data"


# ### Read arc points in celestial coordinates from DS9-format regions file
#
# This function is copied over from the circle-fit project with some updates to reflect more recent API changes

def read_arc_data_ds9(filename, pt_star="o", pt_arc="x"):
    """
    Return the sky coordinates of a star (single point of type
    `pt_star`) and arc (multiple points of type: `pt_arc`), which are
    read from the DS9 region file `filename`
    """
    regions = rg.Regions.read(filename)

    try:
        star, = [x for x in regions if x.visual['point'] == pt_star]
    except IndexError:
        sys.exit("One and only one 'circle' region is required")
    points = [x for x in regions if x.visual['point'] == pt_arc]
    return star, points



star, points = read_arc_data_ds9(datapath / "new-069-601-ridge.reg")

star.center

# ### Convert to Cartesian x, y pixel coordinates
#
# We use a WCS transformation to put the arc in simple x, y coordinates so we do not need to worry about any astro stuff for a while. We could get the WCS from a fits image header, but instead we will just construct a grid centered on the star with 0.1 arcsec pixels.
#

w = WCS(naxis=2)
w.wcs.crpix = [0, 0]
w.wcs.cdelt = np.array([-0.1, 0.1]) / 3600
w.wcs.crval = [star.center.ra.deg, star.center.dec.deg]
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
w

xpts, ypts = SkyCoord([_.center for _ in points]).to_pixel(w)

# ## Plot the points

fig, ax = plt.subplots()
ax.scatter(xpts, ypts)
ax.set_aspect("equal")
...;

# ## Fit the arc

result_p = confit.fit_conic_to_xy(xpts, ypts, only_parabola=True)
result_e = confit.fit_conic_to_xy(xpts, ypts, only_parabola=False)

result_e

beste_xy = confit.XYconic(**result_e.params.valuesdict())
print(beste_xy)
bestp_xy = confit.XYconic(**result_p.params.valuesdict())
print(bestp_xy)

# +
fig, ax = plt.subplots()
ax.scatter(xpts, ypts)

for xy, c in [[bestp_xy, "orange"], [beste_xy, "m"]]:
    ax.plot(xy.x_pts, xy.y_pts, color=c)
    ax.scatter(xy.x0, xy.y0, marker="+", color=c)
    ax.plot([xy.x0, xy.x_mirror], [xy.y0, xy.y_mirror], color=c)

ax.axhline(0, lw=0.5, c="k")
ax.axvline(0, lw=0.5, c="k")
ax.set_aspect("equal")
margin = 8
ax.set(
    xlim=[xpts.min() - margin, xpts.max() + margin],
    ylim=[ypts.min() - margin, ypts.max() + margin],
)
...;
# -

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(result_p.residual, "-o")
ax.plot(result_e.residual, "-o")
ax.axhline(0, color="k", lw=0.5)
ax.set(
    xlabel="data point #",
    ylabel=r"residual: $r - e \times d$",
)
...;


