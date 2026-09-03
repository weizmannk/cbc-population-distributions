.. _setup:

============
Installation
============

This guide explains how to install the required components to run the Makefile and
draw the GWTC-5.0 distribution. It is recommended to use `uv <https://docs.astral.sh/uv/>`_
to create and manage the virtual environment, which avoids dependency conflicts.

.. note::

   The model describes neutron stars and the primary black hole mass distribution as a broken power law
   between minimum and maximum masses, with two Gaussian peaks at
   :math:`\sim 30~M_\odot` and :math:`\sim 9~M_\odot` (posterior median; see
   :ref:`hyperparams` for the full table).
   The file includes the posterior values for mass, spin, and merger-rate hyperparameters.

   Using these hyperparameters, the model can be used to generate synthetic :term:`CBC` distributions
   under the GWTC-5.0 FullPop population model. Here, we draw a distribution of one million
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

Below we provide a small script to read the FullPop (GWTC-5.0) result file with **bilby**
and extract the MAP and posterior-median samples.

:download:`get_hyperparams <../../scripts/hyperparams/get_hyperparams.py>`

.. dropdown:: Show script
   :icon: code
   :animate: fade-in

   .. literalinclude:: ../../scripts/hyperparams/get_hyperparams.py
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
