PIPE
====

.. image:: https://github.com/alphapsa/PIPE/actions/workflows/ci_tests.yml/badge.svg
   :target: https://github.com/alphapsa/PIPE/actions/workflows/ci_tests.yml

.. image:: https://readthedocs.org/projects/pipe-cheops/badge/?version=latest
   :target: https://pipe-cheops.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

PIPE (PSF Imagette Photometric Extraction) is a photometric extraction package for
CHEOPS that is complementing the official Data Reduction Pipeline (DRP). While the
DRP is using aperture photometry on the 200x200 pixel subarrays provided by CHEOPS,
PIPE is using PSF photometry and can also extract photometry from the D=60 pixel
imagettes sometimes provided for brighter stars. Imagettes have the advantage of
shorter cadence than the subarrays, improving the time resolution.

Why should you care about PIPE when we already have the DRP? There are a couple
of advantages in using PIPE:

* The shorter cadence of imagettes increases the time resolution of the
  photometry. This can be of a distinct advantage for some applications,
  e.g. when stellar activity such as flares need to be time resolved, or
  when ingress/egress is desired to be better resolved for timing purposes.

* PSF photometry is less influenced by background contaminant stars. PIPE can
  extract disentangled photometry from background stars (of separations greater
  than arcsec)

* For faint targets PSF extraction makes better use of the signal by weighting
  it over the PSF, so that higher S/N photometry can be extracted.

* The shorter cadence and producing an accurate PSF model makes it easier to
  find and correct for cosmic rays.

* PSF photometry can properly deal with hot or bad pixels by either giving them
  a lower weight or masking them entirely in the fitting process. This becomes
  increasingly important as the detector ages in space.

Are there reasons then to not always use PIPE when extracting photometry? Yes,
because accurate PSF photometry requires accurate PSF modelling. The PSF in turn
depends on

* the location on the detector
* the spectral energy distribution of the target
* the temperature of the CHEOPS telescope tube
* the motion blur introduced by pointing jitter

These challenges to produce an accurate PSF are addressed in PIPE but also
mean that PIPE is not robust enough to be run blindly, as it can fail if
assumptions about the data are wrong. In principle one could attempt to
anticipate all ways assumptions can fail, but in practice we are quite far
from achieving that. A poorly determined PSF could result in photometric
performance that is worse than that provided by the DRP. Thus, PIPE needs to
be run in a supervised mode and is not a replacement for the DRP.

Find the PIPE manual here: https://github.com/alphapsa/PIPE/blob/ad47132ca5afaaf512c9da81759fa5e11c6f3ac1/docs/pipe/PIPE_manual.pdf

Original author: Alexis Brandeker <alexis@astro.su.se>, Stockholm University.

License
-------

This project is Copyright (c) Alexis Brandeker and licensed under
the terms of the GNU GPL v3+ license. This package is based upon
the `Openastronomy packaging guide <https://github.com/OpenAstronomy/packaging-guide>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.
