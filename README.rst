cbc-population-distributions: CBC Population Distributions from GWTC-4
======================================================================

.. image:: https://readthedocs.org/projects/cbc-population-distributions/badge/?version=latest
   :target: https://cbc-population-distributions.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

A pipeline to sample compact binary coalescence (CBC) populations based on the
Gravitational-Wave Transient Catalog 4 (GWTC-4) models: a Broken Power Law with
two Gaussian peaks for the component mass distribution, and a Power Law for the
redshift distribution.

We provide large-scale sampling (up to one million binaries), which can be used
with the `observing-scenarios-simulations <https://github.com/lpsinger/observing-scenarios-simulations>`_
pipeline to predict upcoming observing campaigns of the LIGO–Virgo–KAGRA (LVK) collaboration
and International Gravitational-Wave Network (IGWN).


Getting Started
---------------

To reproduce the distribution, we provide a lightweight architecture based on
``uv``, where a single ``Makefile`` command executes the entire process.
See the `documentation <https://cbc-population-distributions.readthedocs.io/en/latest/>`_
for a detailed guide.

We also provide Jupyter notebooks and a code base to visualize the distribution
results, together with comprehensive documentation of the underlying models.
