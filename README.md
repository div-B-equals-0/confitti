# 🎊 Confitti 🎊 - for Conics Fitting
Fit conic sections (ellipse, parabola, hyperbola) to a set of points

See the [presentation](https://github.com/dawg-at-irya/conics-talk)

[![](https://github.com/dawg-at-irya/conics-talk/blob/main/slides/001.jpeg?raw=true)](https://github.com/dawg-at-irya/conics-talk#individual-slides "Slides from presentation about fitting conic sections to points")

## Installation
```bash
pip install confitti
```
or
```bash
uv pip install confitti
```
will install the package plus the required dependencies (numpy, scipy, lmfit). 

Optional dependencies, which are used in the example notebooks, 
may be pip-installed separately: 
  * emcee for MCMC sampling
  * matplotlib, seaborn, and corner for plotting
  * astropy and regions for dealing with celestial coordinates
  
## Usage
See the example jupyter notebooks in the [notebooks][] directory. For example, 
  * [demo01-basic.ipynb][] demonstrates basic usage: finding the best-fit parabola (or general conic) to a set of (x, y) points
  * [demo02-emcee.ipynb][] explores uncertainty in the parameters of the best-fit curve by means of mcmc
  * [demo03-proplyd.ipynb][] is an example application to real astronomical data (HST image of a bow shock in the Orion Nebula)
  
## Prior art
This is the successor project to [circle-fit](https://github.com/div-B-equals-0/circle-fit)

Some of the literature on the topic of fitting conic sections to points is described [here](https://github.com/div-B-equals-0/confitti/tree/main/docs/prior-art.org). 


[demo01-basic.ipynb]: https://github.com/div-B-equals-0/confitti/tree/main/notebooks/demo01-basic.ipynb
[demo02-emcee.ipynb]: https://github.com/div-B-equals-0/confitti/tree/main/notebooks/demo02-emcee.ipynb
[demo03-proplyd.ipynb]: https://github.com/div-B-equals-0/confitti/tree/main/notebooks/demo03-propyd.ipynb
[notebooks]: https://github.com/div-B-equals-0/confitti/tree/main/notebooks


