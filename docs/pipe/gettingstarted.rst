Getting started
---------------

This is an example reduction process using PIPE to analyse the In-Orbit
Commissioning observations of the transit of KELT-11, which are available for
download on `DACE <http://dace.unige.ch>`_.

.. note::

    Due to some python quirks and
    that PIPE uses the multiprocessing module, this script cannot be run in
    an interactive python session but has to be executed independently as
    ``python reduce_Kelt11.py``.

Default is to use up to system CPU cores-1 number of threads. Some parts
are not parallelised yet, in particular the smearing correction part
that can be a bottleneck.

.. warning::

    PIPE is not optimised for RAM usage. Rule of thumb is that 6 MB per
    subarray frame is used, so ~8 GB for the 1500 frames of the Kelt-11
    visit. This likely means that about 16 GB is recommended. I have only
    tried on systems with much more RAM than that, so no guarantees.

The photometric extraction for subarrays and imagettes of the Kelt-11
data in this example takes about 1 minute on a 2015 MacBook Pro. Producing
PSF models takes additional 90 min. Normally previous PSF models from stars of
similar spectral type and located on the same part of the detector can be
used, so no need to derive new PSF models for every visit.

First, we import the PIPE objects that we'll need access to:

.. code-block:: python

    from pipe import PipeParam, PipeControl

We next initialise a parameter object. Here we have put the data from DACE in
the ``data_root/Kelt-11/101/`` directory

.. code-block:: python

    pps = PipeParam('Kelt-11', '101')

Next we set the sub-array range to extract a light curve for only a fraction
of the full observation. This is mostly used for testing with shorter execution
time. The format is (e.g.) ``pps.sa_range = (10, 20)`` for starting frame 10
and end frame 20. Imagettes are limited to the same epoch range
(so higher frame number range).

.. code-block:: python

    pps.sa_range = (10, 20)

Next is the number of the PSF principal component library "eigenlib"-
file that has been previously produced. In this case the filename
is "eigenlib_815_281_70_0.pkl" where the last number is the running
number.

.. code-block:: python

    pps.psflib = 0


We can optionally use dark subtraction, but for the purposes of this tutorial
we'll turn off this feature:

.. code-block:: python

    pps.darksub = False

In particular for faint objects, flux has been shown to correlate well
with determined background. The reason is likely a neglected non-linearity
of the detector at low exposure levels. This tweak corrects for the low-
exposure linearity and strongly reduces the flux-background correlation.
When we have the non-linearity better characterised, this will go into
the regular non-linearity correction.

.. code-block:: python

    pps.non_lin_tweak = True

The pipe_control object contains high-level methods

.. code-block:: python

    pc = PipeControl(pps)

Process the data using 10 principal PSF components (how many are available
depends on the library). How many components that is optimal to use
varies with circumstances. Too few and the varying PSF is not fit. Too
many and noise is fitted. Rule of thumb: ``klip=1`` to ``5`` for faint targets
without imagettes, ``klip=10`` for bright targets.

.. code-block:: python

    pps.klip = 10
    pc.process_eigen()

Output data is put in the output directory. ``residuals_sa.fits`` is
a fits-cube and contains residuals between fitted PSF and data, and can
be used to inspect if there are systematics left. ``maskcube_sa.fits``
displays what pixels were masked during the fitting process.

.. note::

    The next step need only to be taken if a new set of PSF functions is
    to be produced. It takes about 40 min per iteration, for 3 iterations.

Produce a set of PSFs from the observations (default is one PSF per CHEOPS
orbit). These PSFs are used for deriving principal components for PSF
variability. PSFs from several visits can be combined for the principal
component analysis, but they should be centered on the same location on
the detector (since the PSF varies with location).

.. code-block:: python

    pc.make_psf_lib()
