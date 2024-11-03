# Changelog
## v0.2.1 (2024-11-03)

### Fix
- Fix bug in `confitti.ConicFitResult` when the model has a parameter that does not vary. For instance, eccentricity for the parabola fits.  The uncertainty (`.uvar` property) for such parameters is now set to zero.

## v0.2.0 (2024-10-16)

### New Features
- New class `confitti.ConicFitResult` to store the results of a conic fit. 
- Fit parameters can now be saved to and loaded from files in YAML or JSON format with 
 `confitti.ConicFitResult` methods `.write()` and `.read()`.
 
### Documentation
- Example notebooks `notebooks/demo03-proplyd.ipynb` and `notebooks/demo04-other-proplyd.ipynb` now demonstrate saving the fit results to a file.
- New example notebook `notebooks/demo06-read-save-files.ipynb` demonstrates loading previously saved fit results.

## v0.1.4 (2024-10-15)

### Fix
- Allow negative orientation angles in `fit_conic_to_xy()` (turn off with `allow_negative_theta=False`)

## v0.1.3 (2024-04-03)

### Documentation

- Fix README links to work in both github and pyPI

## v0.1.2 (2024-04-03)

### Documentation

- Fix error in README.md

## v0.1.1 (2024-04-03)

Not sure what happened to this

## v0.1.0 (2024-04-03)

Initial version of the package
