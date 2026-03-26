.. _quickstart:

Quick Start Guide
===============

Using FAST-PT is straightforward. Here's a simple example to get started:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from fastpt import FASTPT, FPTHandler

   #Define a k range
   k = np.logspace(1e-4, 1, 1000)

   # Initialize FASTPT
   fpt = FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5*len(k)))
   handler = FPTHandler(fpt)

   # Use the handler to generate a power spectrum
   P = handler.generate_power_spectra()

   # Calculate an individual term using the handler
   P_1loop = handler.get("P_1loop", P=P)

   # Store default parameters
   handler.update_default_params(P=P, P_window=np.array([0.2, 0.2]), C_window=0.75)

   # Use the stored parameters in a calculation
   tt_result = handler.run("IA_tt")

   # Or get the result directly
   tt_direct = fpt.IA_tt(P=P, P_window=np.array([0.2, 0.2]), C_window=0.75)

   # Plot the results
   handler.plot(data=tt_result)

NOTE: This code uses the `FPTHandler` class to simplify the process of generating power spectra and calculating specific terms. However, the handler is not needed for any `FAST-PT` calculations. It simply provides a convenient interface for common tasks and integration with relevant libraries like `matplotlib`, `CAMB`, and `CLASS`.