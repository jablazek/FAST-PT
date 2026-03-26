.. _installation:

Installation
===========

Dependencies
-----------

FAST-PT requires the following Python packages:

* numpy >= 1.17
* scipy >= 1.2
* matplotlib >= 3.0

Optional dependencies for generating power spectra:

* CAMB
* classy (python wrapper for CLASS)

These optional dependencies are required to use the power spectrum generation features but are not needed for the core FAST-PT functionality. Required dependencies will be installed automatically when you install FAST-PT.

Installation Methods
-------------------

FAST-PT can be installed using pip:

.. code-block:: bash

   pip install fast-pt

Or with conda:

.. code-block:: bash

   conda install fast-pt

For developer installation:

.. code-block:: bash

   git clone https://github.com/jablazek/FAST-PT.git
   cd FAST-PT
   pip install -e .