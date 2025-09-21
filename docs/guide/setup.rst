.. _setup:

============
Installation
============

This guide explains how to install the required components to run the Makefile and
draw the GWTC-4 distribution. It is recommended to use `uv <https://docs.astral.sh/uv/>`_
to create and manage the virtual environment, which avoids dependency conflicts.

.. note::

   The model describes the primary black hole mass distribution as a broken power law
   between minimum and maximum masses, with two Gaussian peaks at
   :math:`\sim 10~M_\odot` and :math:`\sim 35~M_\odot`.
   The file includes the posterior values for mass, spin, and merger-rate hyperparameters.

   Using these hyperparameters, the model can be used to generate synthetic :term:`CBC` distributions
   under the GWTC-4 Broken Power Law + Two Peaks population model. Here, we draw a distribution of one million
   samples to be used with the `observing-scenarios pipeline <https://github.com/lpsinger/observing-scenarios-simulations/>`_
   together with ``ligo.skymap`` to simulate observing campaigns for upcoming runs.


.. dropdown:: Requirements

   You will need the following:

   .. button-link:: https://colmtalbot.github.io/gwpopulation/
      :color: info
      :shadow:

      gwpopulation

   :bdg-warning:`Python >= 3.11`


.. dropdown:: Environment setup with uv

   .. code-block:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh
      uv sync



=========================
Read the hyperparams file
=========================

Below we provide a small script to read the BP2 result file with **bilby**
and extract the MAP (or ML) sample.

:download:`bp2p_hyperparams <../../scripts/bp2p_hyperparams.py>`

.. literalinclude:: ../../scripts/bp2p_hyperparams.py
   :language: python
   :linenos:


================
Run the Pipeline
================

.. dropdown:: Running the pipeline

   Two equivalent options are available - choose one:

   .. tab-set::

      .. tab-item:: Using uv directly

         Run commands without activating the environment explicitly:

         .. code-block:: bash

            uv run make

      .. tab-item:: Activating the uv environment

         Activate the ``.venv`` created by uv and run the pipeline:

         .. code-block:: bash

            source .venv/bin/activate
            make
