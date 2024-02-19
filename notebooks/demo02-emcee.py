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

# # Demonstrate MCMC exploration of posterior distribution of conic parameters

# ## Imports
#
# We need to explictly add the path to the library since we haven't installed it yet.

import sys

sys.path.append("../src")
import confit
import numpy as np
import lmfit
from matplotlib import pyplot as plt
import seaborn as sns

# ## Test data
#
# Symmetric arrangement of 7 points, which I then stretch and distort to make it more interesting.  Using fewer than 7 points is not recommended, although it is possible to ge spectacularly small residuals that way!

xpts, ypts = np.array([1, 2, 3, 4, 5, 6, 7]), np.array([0, 4, 6, 7, 6, 4, 0])
ypts += xpts
xpts *= 3


# ## Do the fitting
#
# Fit of a general conic with `only_parabola=False` so that the eccentricity is allowed to vary.

result_p = confit.fit_conic_to_xy(xpts, ypts, only_parabola=True)
result_e = confit.fit_conic_to_xy(xpts, ypts, only_parabola=False)

# Look at the results

result_e

#
# There are some significant correlations between parameters, which can be better studied via MCMC, which we will do next

# ## Calculate posterior probability of parameters with emcee
#
#

emcee_kws = dict(
    steps=1000, burn=300, thin=20, is_weighted=False, progress=False, workers=16, nan_policy="omit",
)
emcee_params = result_e.params.copy()

result_emcee = lmfit.minimize(confit.residual, args=(xpts, ypts), method='emcee', params=emcee_params, **emcee_kws)

result_emcee

plt.plot(result_emcee.acceptance_fraction, 'o')
plt.xlabel('walker')
plt.ylabel('acceptance fraction')
plt.show()

# +
import corner

emcee_plot = corner.corner(result_emcee.flatchain, labels=result_emcee.var_names,
                           truths=list(result_emcee.params.valuesdict().values()))
# -

result_emcee_p = lmfit.minimize(confit.residual, args=(xpts, ypts), method='emcee', params=result_p.params.copy(), **emcee_kws)

result_emcee_p 

plt.plot(result_emcee_p.acceptance_fraction, 'o')
plt.xlabel('walker')
plt.ylabel('acceptance fraction')
plt.show()

#truths = [result_emcee_p.params.valuesdict()[name] for name in result_emcee_p.var_names]
truths = [result_p.params.valuesdict()[name] for name in result_p.var_names] + [result_emcee_p.params.valuesdict()["__lnsigma"]]
emcee_plot_p = corner.corner(
    result_emcee_p.flatchain, labels=result_emcee_p.var_names, truths=truths,
)



# ## Plotting the best fit onto the data

best_xy = confit.XYconic(**result_p.params.valuesdict())
print(best_xy)

# Get a list of dicts with the conic parameters from the MC chain

chain_pars = result_emcee_p.flatchain.drop(columns="__lnsigma").to_dict(orient="records")
len(chain_pars)

# Take every 35th row so we have 100 samples in total and get the xy curves for them all

chain_xy = [confit.XYconic(**row, eccentricity=1.0) for row in chain_pars[10::17]]

fig, axes = plt.subplots(1, 2, figsize=(12, 8))
for ax in axes:
    ax.scatter(xpts, ypts)

    c = "orange"
    ax.plot(best_xy.x_pts, best_xy.y_pts, color=c)
    ax.scatter(best_xy.x0, best_xy.y0, marker="+", color=c)
    ax.plot([best_xy.x0, best_xy.x_mirror], [best_xy.y0, best_xy.y_mirror], color=c)

    c = "m"
    alpha = 0.05
    for xy in chain_xy:
        ax.plot(xy.x_pts, xy.y_pts, color=c, alpha=alpha)
        ax.scatter(xy.x0, xy.y0, marker="+", color="k", alpha=alpha)
        ax.plot([xy.x0, xy.x_mirror], [xy.y0, xy.y_mirror], color="k", alpha=alpha)

    ax.set_aspect("equal")
margin = 150
axes[0].set(
    xlim=[xpts.min() - margin, xpts.max() + margin],
    ylim=[ypts.min() - margin, ypts.max() + margin],
)
margin = 5
axes[1].set(
    xlim=[xpts.min() - margin, xpts.max() + margin],
    ylim=[ypts.min() - margin, ypts.max() + margin],
)
...;

best_xy = confit.XYconic(**result_e.params.valuesdict())
chain_pars = result_emcee.flatchain.drop(columns="__lnsigma").to_dict(orient="records")
chain_xy = [confit.XYconic(**row) for row in chain_pars[1::10]]

# +
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
for ax in axes:
    c = "orange"
    ax.plot(best_xy.x_pts, best_xy.y_pts, color=c)
    ax.scatter(best_xy.x0, best_xy.y0, marker="+", color=c)
    ax.plot([best_xy.x0, best_xy.x_mirror], [best_xy.y0, best_xy.y_mirror], color=c)

    c = "m"
    alpha = 0.05
    for xy in chain_xy:
        ax.plot(xy.x_pts, xy.y_pts, color=c, alpha=alpha)
        ax.scatter(xy.x0, xy.y0, marker="+", color="k", alpha=alpha)
        ax.plot([xy.x0, xy.x_mirror], [xy.y0, xy.y_mirror], color="k", alpha=alpha)
    ax.scatter(xpts, ypts, zorder=1000)
    ax.set_aspect("equal")
    
margin = 50
axes[0].set(
    xlim=[xpts.min() - margin, xpts.max() + margin],
    ylim=[ypts.min() - margin, ypts.max() + margin],
)
margin = 5
axes[1].set(
    xlim=[xpts.min() - margin, xpts.max() + margin],
    ylim=[ypts.min() - margin, ypts.max() + margin],
)
...;
# -



# ## Try and put limits on parameters to avoid the "unreasonable" global minima
#
# In the parabola case, we have a whole bunch of supposedly valid fits that have small value of `r0` (less than 1) coupled with large values of `x0` and `y0` (more than 30) and `theta0` angles around 30 deg. In the figure above, they can be seen to all be well separated from the "good" fits.  So if we put bounds on `r0` we could possibly eliminate them. 

new_params = result_p.params.copy()
rscale = new_params["r0"].value
new_params["r0"].set(min=rscale/2, max=rscale*2)
new_params

result_emcee_pp = lmfit.minimize(confit.residual, args=(xpts, ypts), method='emcee', params=new_params, **emcee_kws)

result_emcee_pp

truths = [result_emcee_pp.params.valuesdict()[name] for name in result_emcee_pp.var_names]
emcee_plot_p = corner.corner(
    result_emcee_pp.flatchain, labels=result_emcee_pp.var_names, truths=truths, bins=50,
)

best_xy = confit.XYconic(**result_p.params.valuesdict())
chain_pars = result_emcee_pp.flatchain.drop(columns="__lnsigma").to_dict(orient="records")
chain_xy = [confit.XYconic(**row, eccentricity=1.0) for row in chain_pars[1::10]]

# +
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
for ax in axes:
    c = "orange"
    ax.plot(best_xy.x_pts, best_xy.y_pts, color=c)
    ax.scatter(best_xy.x0, best_xy.y0, marker="+", color=c)
    ax.plot([best_xy.x0, best_xy.x_mirror], [best_xy.y0, best_xy.y_mirror], color=c)

    c = "m"
    alpha = 0.01
    for xy in chain_xy:
        ax.plot(xy.x_pts, xy.y_pts, color=c, alpha=alpha)
        ax.scatter(xy.x0, xy.y0, marker="+", color="k", alpha=alpha)
        ax.plot([xy.x0, xy.x_mirror], [xy.y0, xy.y_mirror], color="k", alpha=alpha)
    ax.scatter(xpts, ypts, zorder=1000)
    ax.set_aspect("equal")
    
margin = 50
axes[0].set(
    xlim=[xpts.min() - margin, xpts.max() + margin],
    ylim=[ypts.min() - margin, ypts.max() + margin],
)
margin = 5
axes[1].set(
    xlim=[xpts.min() - margin, xpts.max() + margin],
    ylim=[ypts.min() - margin, ypts.max() + margin],
)
...;

# +
# corner.corner?
