# 🎊 Confitti 🎊 - for Conics Fitting
Fit conic sections (ellipse, parabola, hyperbola) to a set of points

This is the successor project to [circle-fit](https://github.com/div-B-equals-0/circle-fit)

## Installation
```bash
pip install confitti
```
or
```bash
uv install confitti
```
will install the package plus the required dependencies (numpy, scipy, lmfit). 

Optional dependencies, which are used in the example notebooks, 
may be pip-installed or uv-installed separately: 
  * emcee for MCMC sampling
  * matplotlib, seaborn, and corner for plotting
  * astropy and regions for dealing with celestial coordinates
  
## Usage
See the example jupyter notebooks in the [notebooks](./notebooks) directory. For example, 
  * [demo01-basic.ipynb](./notebooks/demo01-basic.ipynb) demonstrates basic usage: finding the best-fit parabola (or general conic) to a set of (x, y) points
  * [demo02-emcee.ipynb](./notebooks/demo02-emcee.ipynb) explores uncertainty in the parameters of the best-fit curve by means of mcmc
  * [demo03-proplyd.ipynb](./notebooks/demo03-proplyd.ipynb) is an example application to real astronomical data (HST image of a bow shock in the Orion Nebula)
  
## Prior art
Some of the literature on the topic of fitting conic sections to points is described [here](./docs/prior-art.org). 
