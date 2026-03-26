import numpy as np
from fastpt import FASTPT, FPTHandler

# Initialize with default parameters
k_values = np.loadtxt("Pk_test.dat")[:, 0]

fastpt_instance = FASTPT(k_values, low_extrap=-5, high_extrap=3)
handler = FPTHandler(fastpt_instance, P_window=np.array([0.2, 0.2]), C_window=0.75)
P = np.loadtxt("Pk_test.dat")[:, 1]
# Generate and store a power spectrum
try: 
    P_camb = handler.generate_power_spectra(method='camb')
except Exception as e:
    if ("is not installed" in str(e)):
        print("Camb is not installed to generate power spectra, using preloaded power spectrum instead.")
        P = np.loadtxt("Pk_test.dat")[:, 1]
    else:
        print("An error occurred while generating power spectra:", e)

handler.update_default_params(P=P)

# Get the 1-loop power spectrum, using the default parameters
result = handler.get("P_1loop")
# ^^ This is equivalent to: calling fastpt_instance.one_loop_dd(P, P_window=np.array([0.2,0.2]), C_window=0.75)[0]

#Plot the results
handler.plot(data=result, title="P_1loop")

# # Save the results and your parameters
handler.output_dir = "output"
handler.save_output(result, "one_loop_dd")
handler.save_params("params.npz")