*************************
PSF Photometry for CHEOPS
*************************

Table of contents
^^^^^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 1

  pipe/installation.rst
  pipe/gettingstarted.rst
  pipe/api.rst

Background
^^^^^^^^^^

PIPE is a photometric extraction package for CHEOPS that is complementing
the official Data Reduction Pipeline (DRP). While the DRP is using aperture
photometry on the D=200 pixel subarrays provided by CHEOPS, PIPE is using
PSF photometry on the D=60 pixel imagettes, often provided with a shorter cadence
than the subarrays.

Why should you care about PIPE when we already have the DRP? There are a couple
of advantages in using PIPE:

* The shorter cadence of imagattes increases the time resolution of the
  photometry. This can be of a distinct advantage for some applications,
  e.g. when stellar activity such as flares need to be time resolved, or
  when ingress/egress is desired to be better resolved for timing purposes.

* For faint targets PSF extraction makes better use of the signal by weighting
  it properly over the PSF, so that higher S/N photometry can be extracted.

* The shorter cadence and producing an accurate PSF model makes it easier to
  find and correct for cosmic rays.

* PSF photometry can properly deal with hot or bad pixels by either giving them
  a lower weight or masking them entirely in the fitting process. This may become
  critical as the detector ages in space, making it increasingly harder to find
  "good" regions on the detector without bad pixels.

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

Original author: Alexis Brandeker <alexis@astro.su.se>, Stockholm University.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
