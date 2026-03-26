import numpy as np
from fastpt import FASTPT


d=np.loadtxt('Pk_test.dat')
k=d[:,0]; P=d[:,1]
C_window=.75
n_pad=int(0.5*len(k))
to_do=['all']
fpt=FASTPT(k,to_do=to_do,low_extrap=-5,high_extrap=3,n_pad=n_pad)

# Base one-loop terms
P_dd = fpt.one_loop_dd(P, C_window=C_window)[0]  # Need [0] for P_22 + P_13
P_kPol = fpt.kPol(P, C_window=C_window)
P_OV = fpt.OV(P, C_window=C_window)

# Bias terms
P_loop_bias = fpt.one_loop_dd_bias(P, C_window=C_window)
P_bias_b3nl = fpt.one_loop_dd_bias_b3nl(P, C_window=C_window)
P_bias_lpt_NL = fpt.one_loop_dd_bias_lpt_NL(P, C_window=C_window)

# CLEFT terms - requires cleft_z1 and z2
#P_cleft_Q_R = fpt.cleft_Q_R(P, C_window=C_window)
# ^^ Throws error because cleft_z1 and cleft_z2 are not defined

# Intrinsic Alignment terms
P_IA_tt = fpt.IA_tt(P, C_window=C_window)
P_IA_ta = fpt.IA_ta(P, C_window=C_window)
P_IA_mix = fpt.IA_mix(P, C_window=C_window)
P_IA_ct = fpt.IA_ct(P, C_window=C_window)
P_gI_ct = fpt.gI_ct(P, C_window=C_window)
# P_IA_gb2 = fpt.IA_gb2(P, C_window=C_window)
P_gI_ta = fpt.gI_ta(P, C_window=C_window)
P_gI_tt = fpt.gI_tt(P, C_window=C_window)
P_IA_der = fpt.IA_der(P, C_window=C_window)

# RSD terms - note P_RSD requires f=1.0 parameter
P_RSD = fpt.RSD_components(P, 1.0, C_window=C_window)
P_RSD_ABsum_components = fpt.RSD_ABsum_components(P, 1.0, C_window=C_window)
P_RSD_ABsum_mu = fpt.RSD_ABsum_mu(P, 1.0, 1.0, C_window=C_window)

# IR resummation
P_IRres = fpt.IRres(P, C_window=C_window)


names = {
    'k': k,
    'P_dd': P_dd,
    'P_OV': P_OV,
    'P_RSD_ABsum_mu': P_RSD_ABsum_mu,
    'P_IRres': P_IRres,
    'P_kPol': P_kPol,
    'PIA_tt': P_IA_tt,
    'P_IA_ta': P_IA_ta,
    'P_IA_mix': P_IA_mix,
    'P_IA_ct': P_IA_ct,
    'P_gI_ct': P_gI_ct,
    'P_gI_ta': P_gI_ta,
    'P_gI_tt': P_gI_tt,
    'P_IA_der': P_IA_der,
    'P_RSD': P_RSD,
    'P_RSD_ABsum_components': P_RSD_ABsum_components,
}

for name, arr in names.items():
    try: 
        np.savetxt(f'{name}_benchmark.txt', np.transpose(arr), header=f'{name}')
    except AttributeError:
        print(f"Error saving {name} array")
        print(AttributeError.with_traceback())

inhomogeneous_arrays = {
    'P_loop_bias': list(P_loop_bias), #Have to cast to list since tuple is immutable
    'P_bias_b3nl': list(P_bias_b3nl),
    'P_bias_lpt_NL': list(P_bias_lpt_NL),
}

for name, arr in inhomogeneous_arrays.items():
    for i, element in enumerate(arr):
        if isinstance(element, float): #sig4 is of type float, converting it to np array
            new_array = np.zeros(len(arr[i-1])) #should be of length 3000, not hardcoded incase it changes
            new_array[0] = element
            arr[i] = new_array
    np.savetxt(f'{name}_benchmark.txt', np.transpose(arr), header=f'{name}')