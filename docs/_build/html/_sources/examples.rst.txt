.. _examples:

Examples
=======

More examples can be found `here <https://github.com/jablazek/FAST-PT/tree/master/examples>`_.

One-loop Matter Power Spectrum
----------------------------

.. code-block:: python

   from fastpt import FASTPT
   import numpy as np
   import matplotlib.pyplot as plt

   # Load data
   data = np.loadtxt('Pk_test.dat')
   k = data[:, 0]
   P = data[:, 1]

   # Initialize FASTPT
   fpt = FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5*len(k)))

   # Calculate corrections
   P_1loop, P_components = fpt.one_loop_dd(P, C_window=0.75)

   # Plot
   plt.figure(figsize=(10, 7))
   plt.loglog(k, P, label='Linear P(k)')
   plt.loglog(k, P_1loop, label='1-loop P(k)')
   plt.xlabel('k [h/Mpc]')
   plt.ylabel('P(k) [(Mpc/h)³]')
   plt.legend()
   plt.tight_layout()
   plt.show()

Using the FPTHandler
-----------------

.. code-block:: python

   import numpy as np
   from fastpt import FASTPT, FPTHandler

   # Initialize with default parameters
   k_values = np.logspace(-3, 1, 1000)

   fastpt_instance = FASTPT(k_values)
   handler = FPTHandler(fastpt_instance, P_window=np.array([0.2, 0.2]), C_window=0.75)

   # Generate and store a power spectrum
   P = handler.generate_power_spectra()
   handler.update_default_params(P=P)

   # Get the 1-loop power spectrum, using the default parameters
   result = handler.get("P_1loop")

   #Plot the results
   handler.plot(data=result)

   # Save the results and your parameters
   handler.save_output(result, "one_loop_dd")
   handler.save_params("params.npz")