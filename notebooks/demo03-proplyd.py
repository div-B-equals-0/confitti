# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
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
from pathlib import Path
import sys
import confitti
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
figpath = Path.cwd().parent / "figs"
saveprefix = "demo03"


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
        (star,) = [x for x in regions if x.visual["marker"] == pt_star]
    except IndexError:
        sys.exit("One and only one 'circle' region is required")
    points = [x for x in regions if x.visual["marker"] == pt_arc]
    return star, points


filename = datapath / "new-069-601-ridge.reg"
regions = rg.Regions.read(filename)

regions[0].meta

regions[0].visual

regions[1].visual

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

xpts, ypts = SkyCoord([point.center for point in points]).to_pixel(w)

# ## Plot the points

fig, ax = plt.subplots()
ax.scatter(xpts, ypts)
ax.set_aspect("equal")
...;

# ## Fit the arc

result_p = confitti.fit_conic_to_xy(xpts, ypts, only_parabola=True)
result_e = confitti.fit_conic_to_xy(xpts, ypts, only_parabola=False)

result_e

beste_xy = confitti.XYconic(**result_e.params.valuesdict())
print(beste_xy)
bestp_xy = confitti.XYconic(**result_p.params.valuesdict())
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

fig.savefig(figpath / f"{saveprefix}-best-fits.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(result_p.residual, "-o")
ax.plot(result_e.residual, "-o")
ax.axhline(0, color="k", lw=0.5)
ax.set(
    xlabel="data point #",
    ylabel=r"residual: $r - e \times d$",
)
...


fig.savefig(figpath / f"{saveprefix}-residuals.pdf", bbox_inches="tight")

# ## Calculate posterior probability of parameters with emcee
#
#

emcee_kws = dict(
    steps=5000,
    burn=1000,
    thin=20,
    is_weighted=False,
    progress=False,
    workers=16,
    nan_policy="omit",
)
emcee_params = result_e.params.copy()
emcee_params.add("__lnsigma", value=np.log(0.1), min=np.log(0.001), max=np.log(1.0))

result_emcee = lmfit.minimize(
    confitti.residual,
    args=(xpts, ypts),
    method="emcee",
    params=emcee_params,
    **emcee_kws,
)

result_emcee

plt.plot(result_emcee.acceptance_fraction, "o")
plt.xlabel("walker")
plt.ylabel("acceptance fraction")
plt.show()

# +
import corner

emcee_plot = corner.corner(
    result_emcee.flatchain,
    labels=result_emcee.var_names,
    truths=list(result_emcee.params.valuesdict().values()),
)
# -

emcee_plot.savefig(figpath / f"{saveprefix}-corner.pdf", bbox_inches="tight")

best_xy = confitti.XYconic(**result_e.params.valuesdict())
chain_pars = result_emcee.flatchain.drop(columns="__lnsigma").to_dict(orient="records")
chain_xy = [confitti.XYconic(**row) for row in chain_pars[7::200]]

len(chain_xy)

import matplotlib as mpl

cmap = mpl.cm.rainbow

eparam = result_emcee.params["eccentricity"]
emin, emax = eparam.value - 2 * eparam.stderr, eparam.value + 2 * eparam.stderr
norm = mpl.colors.Normalize(vmin=emin, vmax=emax)
norm(1.0)

# +
fig, axes = plt.subplots(1, 2, figsize=(12, 8))

for ax in axes:
    c = "orange"
    ax.plot(best_xy.x_pts, best_xy.y_pts, color=c)
    ax.scatter(best_xy.x0, best_xy.y0, marker="+", color=c)
    ax.plot([best_xy.x0, best_xy.x_mirror], [best_xy.y0, best_xy.y_mirror], color=c)

    c = "m"
    alpha = 0.1
    for xy in chain_xy:
        c = cmap(norm(xy.eccentricity))
        ax.plot(xy.x_pts, xy.y_pts, color=c, alpha=alpha)
        ax.scatter(xy.x0, xy.y0, marker="+", color=c, alpha=alpha)
        ax.plot([xy.x0, xy.x_mirror], [xy.y0, xy.y_mirror], color=c, alpha=alpha)
    ax.scatter(xpts, ypts, zorder=1000)
    ax.axhline(0, lw=0.5, c="k")
    ax.axvline(0, lw=0.5, c="k")
    ax.set_aspect("equal")

margin = 100
axes[0].set(
    xlim=[xpts.min() - margin, xpts.max() + margin],
    ylim=[ypts.min() - margin, ypts.max() + margin],
)
margin = 10
axes[1].set(
    xlim=[xpts.min() - margin, xpts.max() + margin],
    ylim=[ypts.min() - margin, ypts.max() + margin],
)

fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes[1],
    orientation="horizontal",
    label="eccentricity",
)
...
# -

fig.savefig(figpath / f"{saveprefix}-emcee-samples.pdf", bbox_inches="tight")

# ## Execution time for notebook

print(f"--- {time.time() - start_time} seconds ---")
