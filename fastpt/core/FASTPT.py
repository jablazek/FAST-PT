'''
	FASTPT is a numerical algorithm to calculate
	1-loop contributions to the matter power spectrum
	and other integrals of a similar type.
	The method is presented in papers arXiv:1603.04826 and arXiv:1609.05978
	Please cite these papers if you are using FASTPT in your research.

	Joseph E. McEwen (c) 2016
	mcewen.24@osu.edu

	Xiao Fang
	fang.307@osu.edu

	Jonathan A. Blazek
	blazek.35@osu.edu


	FFFFFFFF    A           SSSSSSSSS   TTTTTTTTTTTTTT             PPPPPPPPP    TTTTTTTTTTTT
	FF     	   A A         SS                 TT                   PP      PP        TT
	FF        A   A        SS                 TT                   PP      PP        TT
	FFFFF    AAAAAAA        SSSSSSSS          TT       ==========  PPPPPPPPP         TT
	FF      AA     AA              SS         TT                   PP                TT
	FF     AA       AA             SS         TT                   PP                TT
	FF    AA         AA    SSSSSSSSS          TT                   PP                TT


	The FASTPT class is the workhorse of the FASTPT algorithm.
	This class calculates integrals of the form:

	\int \frac{d^3q}{(2 \pi)^3} K(q,k-q) P(q) P(|k-q|)

	\int \frac{d^3q_1}{(2 \pi)^3} K(\hat{q_1} \dot \hat{q_2},\hat{q_1} \dot \hat{k}, \hat{q_2} \dot \hat{k}, q_1, q_2) P(q_1) P(|k-q_1|)

'''
from __future__ import division
from __future__ import print_function

from ..info import __version__

import numpy as np
from numpy import exp, log, cos, sin, pi
from ..utils.fastpt_extr import p_window, c_window
from ..utils.matter_power_spt import P_13_reg, Y1_reg_NL, Y2_reg_NL
from ..utils.initialize_params import scalar_stuff, tensor_stuff
from ..IA.IA_tt import IA_tt
from ..IA.IA_ABD import IA_A, IA_DEE, IA_DBB, P_IA_B
from ..IA.IA_ta import IA_deltaE1, P_IA_deltaE2, IA_0E0E, IA_0B0B
from ..IA.IA_gb2 import IA_gb2_F2, IA_gb2_fe, IA_gb2_he, P_IA_13S2F2
from ..IA.IA_gb2 import IA_gb2_S2F2, IA_gb2_S2fe, IA_gb2_S2he
from ..IA.IA_ct import IA_tij_feG2, IA_tij_heG2, IA_tij_F2F2, IA_tij_G2G2, IA_tij_F2G2, P_IA_13G, P_IA_13F, IA_tij_F2G2reg
from ..IA.IA_ctbias import IA_gb2_F2, IA_gb2_G2, IA_gb2_S2F2, IA_gb2_S2G2
from ..utils.OV import OV
from ..utils.kPol import kPol
from ..rsd.RSD import RSDA, RSDB
from ..rsd.RSD_ItypeII import P_Ap1 as RSD_P_Ap1, P_Ap3 as RSD_P_Ap3, P_Ap5 as RSD_P_Ap5
from ..utils.P_extend import k_extend
from . import FASTPT_simple as fastpt_simple
from .CacheManager import CacheManager
from scipy.signal import fftconvolve
from numpy.fft import ifft, irfft, rfft


def cached_property(method):
    """Decorator to cache property values"""
    cache_name = f'_{method.__name__}'
    
    def getter(instance):
        if not hasattr(instance, cache_name):
            setattr(instance, cache_name, method(instance))
        return getattr(instance, cache_name)
    
    return property(getter)

class FASTPT:
    """
    FASTPT is a numerical algorithm to calculate 1-loop contributions to the matter power spectrum and other integrals of a similar type. 
    The method is presented in papers arXiv:1603.04826 and arXiv:1609.05978. Please cite these papers if you are using FASTPT in your research.

    Parameters
        ----------
        k : array_like
            The input k-grid (wavenumbers) in 1/Mpc. Must be logarithmically spaced
            with equal spacing in log(k) and contain an even number of elements.
        
        nu : float, optional
            Deprecated. Previously used to specify power-law biasing for FFT calculations. Now determined automatically.
        
        to_do : list of str, optional
            List of requirements for precalculation of matrices. Starting in v4, terms are calculated as needed
            without specifying them here, but pre-computing matrices remains an option and can save time on the 
            initial run of functions. 'All' or 'everything' will initialize all terms.
        
        param_mat : array_like, optional
            Custom parameter matrix for extensions (advanced usage).
        
        low_extrap : float, optional
            If provided, extrapolate the power spectrum to lower k values 
            down to 10^(low_extrap). Helps with edge effects. Typical value: -5.
        
        high_extrap : float, optional
            If provided, extrapolate the power spectrum to higher k values 
            up to 10^(high_extrap). Helps with edge effects. Typical value: 3.
            Must be greater than low_extrap if both are provided.
        
        n_pad : int, optional
            Number of zeros to pad the array with on both ends. 
            Helps reduce edge effects in FFT calculations. If None, defaults to
            half the length of the input k array.
            
        verbose : bool, optional
            If True, prints additional information during calculations.
        
        simple : bool, optional
            If True, uses the older, simplified FASTPT interface. Will be deprecated.

        max_cache_size_mb : int, optional
            Maximum size of the internal cache in megabytes. Default is 500 MB.

        dump_cache : bool, optional
            If True, the cache is cleared when a new power spectrum is provided to a Fast-PT function to avoid building an unnecessarily large cache.
        
        Notes
        -----
        The input k array must be strictly increasing, equally logarithmically spaced, and
        contain an even number of elements.
        
        Using extrapolation (low_extrap/high_extrap) and padding (n_pad) is 
        recommended to reduce numerical artifacts from the FFT-based algorithm.

    """

    def __init__(self, k, nu=None, to_do=None, param_mat=None, low_extrap=None, high_extrap=None, n_pad=None,
                verbose=False, simple=False, max_cache_size_mb=500, dump_cache=True):
        
        if (k is None or len(k) == 0):
            raise ValueError('You must provide an input k array.')
        
        if nu: print("Warning: nu is no longer needed for FAST-PT initialization. It will default to -2, if another value is needed it will be calculated internally.")
        

        # if no to_do list is given, default to fastpt_simple SPT case
        if (simple):
            if (verbose):
                print(
                    'Note: You are using an earlier call structure for FASTPT. Your code will still run correctly, calling FASTPT_simple. See user manual.')
            if (nu is None):  # give a warning if nu=None that a default value is being used.
                print('WARNING: No value for nu is given. FASTPT_simple is being called with a default of nu=-2')
                nu = -2  # this is the default value for P22+P13 and bias calculation
            print("WARNING: No to_do list is given therefore calling FASTPT_simple. FASTPT_simple will soon be DEPRECATED.")
            self.pt_simple = fastpt_simple.FASTPT(k, nu, param_mat=param_mat, low_extrap=low_extrap,
                                                  high_extrap=high_extrap, n_pad=n_pad, verbose=verbose)
            return None
        # Exit initialization here, since fastpt_simple performs the various checks on the k grid and does extrapolation.
        self.cache = CacheManager(max_size_mb=max_cache_size_mb, dump_cache=dump_cache)
        self.max_cache_size_mb = max_cache_size_mb #Stored for save instance method in handler
        self.dump_cache = dump_cache #Stored for save instance method in handler
        self.X_registry = {} #Stores the names of X terms to be used as an efficient unique identifier in hash keys
        self.__k_original = k
        self.extrap = False
        if (low_extrap is not None or high_extrap is not None):
            if (high_extrap < low_extrap):
                raise ValueError('high_extrap must be greater than low_extrap')
            self.EK = k_extend(k, low_extrap, high_extrap)
            k = self.EK.extrap_k()
            self.extrap = True

        self.low_extrap = low_extrap
        self.high_extrap = high_extrap
        self.__k_extrap = k #K extrapolation not padded

        
        # check for log spacing
        # print('Initializing k-grid quantities...')
        dk = np.diff(np.log(k))
        # dk_test=np.ones_like(dk)*dk[0]
        delta_L = (log(k[-1]) - log(k[0])) / (k.size - 1)
        dk_test = np.ones_like(dk) * delta_L

        log_sample_test = 'ERROR! FASTPT will not work if your in put (k,Pk) values are not sampled evenly in log space!'
        np.testing.assert_array_almost_equal(dk, dk_test, decimal=4, err_msg=log_sample_test, verbose=False)

        if (verbose):
            print(f'the minumum and maximum inputed log10(k) are: {np.min(np.log10(k))} and {np.max(np.log10(k))}')
            print(f'the grid spacing Delta log (k) is, {(log(np.max(k)) - log(np.min(k))) / (k.size - 1)}')
            print(f'number of input k points are, {k.size}')
            print(f'the power spectrum is extraplated to log10(k_min)={low_extrap}')
            print(f'the power spectrum is extraplated to log10(k_max)={high_extrap}')
            print(f'the power spectrum has {n_pad} zeros added to both ends of the power spectrum')


        # print(self.k_extrap.size, 'k size')
        # size of input array must be an even number
        if (k.size % 2 != 0):
            raise ValueError('Input array must contain an even number of elements.')
        # can we just force the extrapolation to add an element if we need one more? how do we prevent the extrapolation from giving us an odd number of elements? is that hard coded into extrap? or just trim the lowest k value if there is an odd numebr and no extrapolation is requested.

        if n_pad is None:
            n_pad = int(0.5 * len(k))
            if verbose:
                print(f"WARNING: N_pad is recommended but none has been provided, defaulting to 0.5*len(k) = {n_pad}.")
        self.n_pad = n_pad
        if (n_pad > 0):
            # Make sure n_pad is an integer
            if not isinstance(n_pad, int):
                n_pad = int(n_pad)
            self.n_pad = n_pad
            self.id_pad = np.arange(k.size) + n_pad
            d_logk = delta_L
            k_pad = np.log(k[0]) - np.arange(1, n_pad + 1) * d_logk
            k_pad = np.exp(k_pad)
            k_left = k_pad[::-1]

            k_pad = np.log(k[-1]) + np.arange(1, n_pad + 1) * d_logk
            k_right = np.exp(k_pad)
            k = np.hstack((k_left, k, k_right))
            n_pad_check = int(np.log(2) / delta_L) + 1
            if (n_pad < n_pad_check):
                print('*** Warning ***')
                print(f'You should consider increasing your zero padding to at least {n_pad_check}')
                print('to ensure that the minimum k_output is > 2k_min in the FASTPT universe.')
                print(f'k_min in the FASTPT universe is {k[0]} while k_min_input is {self.k_extrap[0]}')

        self.__k_final = k #log spaced k, with padding and extrap
        self.k_size = k.size
        # self.scalar_nu=-2
        self.N = k.size

        # define eta_m and eta_n=eta_m
        omega = 2 * pi / (float(self.N) * delta_L)
        self.m = np.arange(-self.N // 2, self.N // 2 + 1)
        self.eta_m = omega * self.m

        self.verbose = verbose

        # define l and tau_l
        self.n_l = self.m.size + self.m.size - 1
        self.l = np.arange(-self.n_l // 2 + 1, self.n_l // 2 + 1)
        self.tau_l = omega * self.l

        self.todo_dict = {
            'one_loop_dd': False, 'one_loop_cleft_dd': False, 
            'dd_bias': False, 'IA_all': False,
            'IA_tt': False, 'IA_ta': False, 
            'IA_mix': False, 'OV': False, 'kPol': False,
            'RSD': False, 'IRres': False, 
            'tij': False, 'gb2': False, 
            'all': False, 'everything': False
        }

        if to_do: 
            print("Warning: to_do list is no longer needed for FAST-PT initialization. Terms will now be calculated as needed. It may still be used to pre-compute matrices for faster initial runs.")
        
            for entry in to_do:
                if entry in {'all', 'everything'}:
                    for key in self.todo_dict:
                        self.todo_dict[key] = True
                elif entry in {'IA_all', 'IA'}:
                    for key in ['IA_tt', 'IA_ta', 'IA_mix', 'gb2', 'tij']:
                        self.todo_dict[key] = True
                elif entry == 'dd_bias':
                    self.todo_dict['one_loop_dd'] = True
                    self.todo_dict['dd_bias'] = True
                elif entry == 'tij':
                    for key in ['gb2', 'one_loop_dd', 'tij', 'IA_tt', 'IA_ta', 'IA_mix']:
                        self.todo_dict[key] = True
                elif entry in self.todo_dict:
                    self.todo_dict[entry] = True
                else:
                    raise ValueError(f'FAST-PT does not recognize {entry} in the to_do list.\n{self.todo_dict.keys()} are the valid entries.')

        
        ### INITIALIZATION of k-grid quantities ###
        if self.todo_dict['one_loop_dd'] or self.todo_dict['dd_bias'] or self.todo_dict['IRres']:
            self.X_spt
            self.X_lpt
            self.X_sptG

        if self.todo_dict['IA_tt']: 
            self.X_IA_E 
            self.X_IA_B

        if self.todo_dict['IA_mix']:
            self.X_IA_A
            self.X_IA_DEE
            self.X_IA_DBB

        if self.todo_dict['IA_ta']:
            self.X_IA_deltaE1
            self.X_IA_0E0E
            self.X_IA_0B0B

        if self.todo_dict['gb2']:
            self.X_IA_gb2_fe
            self.X_IA_gb2_he

        if self.todo_dict['tij']:
            self.X_IA_tij_feG2
            self.X_IA_tij_heG2
            self.X_IA_tij_F2F2
            self.X_IA_tij_G2G2
            self.X_IA_tij_F2G2
            self.X_IA_tij_F2G2reg
            self.X_IA_gb2_F2
            self.X_IA_gb2_G2
            self.X_IA_gb2_S2F2
            self.X_IA_gb2_S2fe
            self.X_IA_gb2_S2he
            self.X_IA_gb2_S2G2

        if self.todo_dict['OV']: 
            self.X_OV

        if self.todo_dict['kPol']:
            self.X_kP1
            self.X_kP2
            self.X_kP3

        if self.todo_dict['RSD']:
            self.X_RSDA
            self.X_RSDB
        
    @property
    def k_original(self):
        return self.__k_original
    
    @property
    def k_extrap(self):
        return self.__k_extrap
    
    @property
    def k_final(self):
        return self.__k_final

    @cached_property
    def X_spt(self):
        nu = -2
        p_mat = np.array([[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [2, -2, 2, 0],
                    [1, -1, 1, 0], [1, -1, 3, 0], [2, -2, 0, 1]])
        result = scalar_stuff(p_mat, nu, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_spt'
        return result
    @cached_property
    def X_lpt(self):
        nu = -2
        p_mat = np.array([[0, 0, 0, 0], [0, 0, 2, 0], [2, -2, 2, 0],
                    [1, -1, 1, 0], [1, -1, 3, 0], [0, 0, 4, 0], [2, -2, 0, 1]])
        result = scalar_stuff(p_mat, nu, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_lpt'
        return result
    @cached_property
    def X_sptG(self):
        nu = -2
        p_mat = np.array([[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [2, -2, 2, 0],
                        [1, -1, 1, 0], [1, -1, 3, 0], [2, -2, 0, 1]])
        result = scalar_stuff(p_mat, nu, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_sptG'
        return result
    @cached_property
    def X_IA_E(self):
        hE_tab, _ = IA_tt()
        p_mat_E = hE_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_E, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_E'
        return result
    @cached_property
    def X_IA_B(self):
        _, hB_tab = IA_tt()
        p_mat_B = hB_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_B, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_B'
        return result
    @cached_property
    def X_IA_A(self):
        IA_A_tab = IA_A()
        p_mat_A = IA_A_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_A, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_A'
        return result
    @cached_property
    def X_IA_DEE(self):
        IA_DEE_tab = IA_DEE()
        p_mat_DEE = IA_DEE_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_DEE, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_DEE'
        return result
    @cached_property
    def X_IA_DBB(self):
        IA_DBB_tab = IA_DBB()
        p_mat_DBB = IA_DBB_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_DBB, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_DBB'
        return result
    @cached_property
    def X_IA_deltaE1(self):
        IA_deltaE1_tab = IA_deltaE1()
        result = tensor_stuff(IA_deltaE1_tab[:, [0, 1, 5, 6, 7, 8, 9]], self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_deltaE1'
        return result
    @cached_property
    def X_IA_0E0E(self):
        IA_0E0E_tab = IA_0E0E()
        result = tensor_stuff(IA_0E0E_tab[:, [0, 1, 5, 6, 7, 8, 9]], self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_0E0E'
        return result
    @cached_property
    def X_IA_0B0B(self):
        IA_0B0B_tab = IA_0B0B()
        result = tensor_stuff(IA_0B0B_tab[:, [0, 1, 5, 6, 7, 8, 9]], self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_0B0B'  
        return result
    @cached_property
    def X_IA_gb2_fe(self):
        IA_gb2_fe_tab = IA_gb2_fe()
        p_mat_gb2_fe = IA_gb2_fe_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_gb2_fe, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_gb2_fe'
        return result
    @cached_property
    def X_IA_gb2_he(self):
        IA_gb2_he_tab = IA_gb2_he()
        p_mat_gb2_he = IA_gb2_he_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_gb2_he, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_gb2_he'
        return result
    @cached_property
    def X_IA_tij_feG2(self):
        IA_tij_feG2_tab = IA_tij_feG2()
        p_mat_tij_feG2 = IA_tij_feG2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_tij_feG2, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_tij_feG2'
        return result
    @cached_property
    def X_IA_tij_heG2(self):
        IA_tij_heG2_tab = IA_tij_heG2()
        p_mat_tij_heG2 = IA_tij_heG2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_tij_heG2, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_tij_heG2'
        return result
    @cached_property
    def X_IA_tij_F2F2(self):
        IA_tij_F2F2_tab = IA_tij_F2F2()
        p_mat_tij_F2F2 = IA_tij_F2F2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_tij_F2F2, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_tij_F2F2'
        return result
    @cached_property
    def X_IA_tij_G2G2(self):
        IA_tij_G2G2_tab = IA_tij_G2G2()
        p_mat_tij_G2G2 = IA_tij_G2G2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_tij_G2G2, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_tij_G2G2'
        return result
    @cached_property
    def X_IA_tij_F2G2(self):
        IA_tij_F2G2_tab = IA_tij_F2G2()
        p_mat_tij_F2G2 = IA_tij_F2G2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_tij_F2G2, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_tij_F2G2'
        return result
    @cached_property
    def X_IA_tij_F2G2reg(self):
        IA_tij_F2G2reg_tab =IA_tij_F2G2reg()
        p_mat_tij_F2G2reg_tab = IA_tij_F2G2reg_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_tij_F2G2reg_tab, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_tij_F2G2reg'
        return result
    @cached_property
    def X_IA_gb2_F2(self):
        IA_gb2_F2_tab = IA_gb2_F2()
        p_mat_gb2_F2 = IA_gb2_F2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_gb2_F2, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_gb2_F2'
        return result
    @cached_property
    def X_IA_gb2_G2(self):
        IA_gb2_G2_tab = IA_gb2_G2()
        p_mat_gb2_G2 = IA_gb2_G2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_gb2_G2, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_gb2_G2'
        return result
    @cached_property
    def X_IA_gb2_S2F2(self):
        IA_gb2_S2F2_tab = IA_gb2_S2F2()
        p_mat_gb2_S2F2 = IA_gb2_S2F2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_gb2_S2F2, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_gb2_S2F2'
        return result
    @cached_property
    def X_IA_gb2_S2fe(self):
        IA_gb2_S2fe_tab = IA_gb2_S2fe()
        p_mat_gb2_S2fe = IA_gb2_S2fe_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_gb2_S2fe, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_gb2_S2fe'
        return result
    @cached_property
    def X_IA_gb2_S2he(self):
        IA_gb2_S2he_tab = IA_gb2_S2he()
        p_mat_gb2_S2he = IA_gb2_S2he_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_gb2_S2he, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_gb2_S2he'
        return result
    @cached_property
    def X_IA_gb2_S2G2(self):
        IA_gb2_S2G2_tab = IA_gb2_S2G2()
        p_mat_gb2_S2G2 = IA_gb2_S2G2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat_gb2_S2G2, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_IA_gb2_S2G2'
        return result
    @cached_property
    def X_OV(self):
        OV_tab = OV()
        p_mat = OV_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_OV'
        return result
    @cached_property
    def X_kP1(self):
        tab1, _, _ = kPol()
        p_mat = tab1[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_kP1'
        return result
    @cached_property
    def X_kP2(self):
        _, tab2, _ = kPol()
        p_mat = tab2[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_kP2'
        return result
    @cached_property
    def X_kP3(self):
        _, _, tab3 = kPol()
        p_mat = tab3[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_kP3'
        return result    
    @cached_property
    def X_RSDA(self):
        tabA, self.A_coeff = RSDA()
        p_mat = tabA[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_RSDA'  
        return result    
    @cached_property
    def X_RSDB(self):
        tabB, self.B_coeff = RSDB()
        p_mat = tabB[:, [0, 1, 5, 6, 7, 8, 9]]
        result = tensor_stuff(p_mat, self.N, self.m, self.eta_m, self.l, self.tau_l)
        self.X_registry[id(result)] = 'X_RSDB'
        return result


    def _validate_params(self, **params):
        """" Same function as before """
        valid_params = ('P', 'P_window', 'C_window', 'f', 'X', 'nu', 'mu_n', 'L', 'h', 'rsdrag')
        for key in params.keys():
            if key not in valid_params:
                raise ValueError(f'Invalid parameter: {key}. Valid parameters are: {valid_params}')
        P = params.get('P', None)
        if (P is None or len(P) == 0):
            raise ValueError('You must provide an input power spectrum array.')
        if (len(P) != len(self.k_original)):
            raise ValueError(f'Input k and P arrays must have the same size. P:{len(P)}, K:{len(self.k_original)}')
            
        if (np.all(P == 0.0)):
            raise ValueError('Your input power spectrum array is all zeros.')

        P_window = params.get('P_window', np.array([]))
        C_window = params.get('C_window', None)

        if P_window is not None and len(P_window) > 0:
            maxP = (log(self.k_final[-1]) - log(self.k_final[0])) / 2
            if len(P_window) != 2:
                raise ValueError(f'P_window must be a tuple of two values.')
            if P_window[0] > maxP or P_window[1] > maxP:
                raise ValueError(f'P_window value is too large. Decrease to less than {maxP} to avoid over tapering.')

        if C_window is not None:
            if C_window < 0 or C_window > 1:
                raise ValueError('C_window must be between 0 and 1.')

        return params
    
    ############## ABSTRACTED BEHAVIOR METHODS ##############
    def _apply_extrapolation(self, *args):
        """ Applies extrapolation to multiple variables at once """
        if not self.extrap:
            return args if len(args) > 1 else args[0]
        return [self.EK.PK_original(var)[1] for var in args] if len(args) > 1 else self.EK.PK_original(args[0])[1]

    def _hash_arrays(self, arrays):
        """Helper function to create a hash from multiple numpy arrays or scalars"""
        if arrays is None: 
            return hash(None)
        if isinstance(arrays, (tuple, list)):
            hash_key_hash = 0
            for i, item in enumerate(arrays):
                if isinstance(item, np.ndarray):
                    # Use a prime multiplier to avoid collisions
                    item_hash = hash(item.tobytes())
                elif isinstance(item, (tuple, list)):
                    # Recursively compute hash of nested structure
                    item_hash = self._hash_arrays(item)
                else:
                    item_hash = hash(item)
                # Combine hashes using a prime-based approach to reduce collisions
                hash_key_hash = hash_key_hash ^ (item_hash + 0x9e3779b9 + (hash_key_hash << 6) + (hash_key_hash >> 2))
            return hash_key_hash

        # Single item case
        if isinstance(arrays, np.ndarray):
            return hash(arrays.tobytes())
        return hash(arrays)

    def _create_hash_key(self, term, X, P, P_window, C_window):
        """Create a hash key from the term and input parameters"""
        P_hash = self._hash_arrays(P)
        P_win_hash = self._hash_arrays(P_window)
        if X is None:
            X_id = hash(None)
        else:
            X_id = hash(self.X_registry.get(id(X), f"unknown_{id(X)}"))
        term_hash = hash(term) #Included for differentiating between similar param sets
        hash_list = [term_hash, X_id, P_hash, P_win_hash, hash(C_window)]
        hash_key = 0
        for h in hash_list:
            if h is not None:
                hash_key = hash_key ^ (h + 0x9e3779b9 + (hash_key << 6) + (hash_key >> 2))
        return hash_key, P_hash

    def compute_term(self, term, X, operation=None, P=None, P_window=None, C_window=None):
        """
        Computes a Fast-PT term with caching support.

        Parameters
        ----------
        term : str
            Name of the term to compute, used for cache identification
        X : tuple
            Fast-PT coefficient matrices for the calculation
        operation : callable, optional
            Function to apply to the result(s) after computation
        P : array_like
            Input power spectrum
        P_window : tuple, optional
            Window parameters for tapering the power spectrum at the endpoints
        C_window : float, optional
            Window parameter for tapering the Fourier coefficients
        
        Returns
        -------
        array_like
            The computed Fast-PT term
        """
        if P is None: 
            raise ValueError('Compute term requires an input power spectrum array.')        

        hash_key, P_hash = self._create_hash_key(term, X, P, P_window, C_window)
        result = self.cache.get(term, hash_key)
        if result is not None: 
            return result

        result, _ = self.J_k_tensor(P, X, P_window=P_window, C_window=C_window)
        result = self._apply_extrapolation(result)

        if operation:
            final_result = operation(result)
            self.cache.set(final_result, term, hash_key, P_hash)
            return final_result

        self.cache.set(result, term, hash_key, P_hash)
        return result
    



    ### Top-level functions to output final quantities ###
    def one_loop_dd(self, P, P_window=None, C_window=None):
        """
        Computes the standard 1-loop density-density corrections to the power spectrum.
    
        Returns
        -------
        tuple
            (P_1loop, Ps) where:
        P_1loop : 1-loop correction (P_22 + P_13)
        Ps : Smoothed input power spectrum
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        Ps = self._get_Ps(P, P_window=P_window, C_window=C_window)
        P_1loop = self._get_P_1loop(P, P_window=P_window, C_window=C_window)
        return P_1loop, Ps #This return is going to be different than the original bc the original return is 
                        # different depending on the todo list which is going to be deprecated.
    
    def _get_Ps(self, P, P_window=None, C_window=None):
        Ps, _ = self.J_k_scalar(P, self.X_spt, -2, P_window=P_window, C_window=C_window)
        Ps = self._apply_extrapolation(Ps)
        return Ps
    
    def _get_P_1loop(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("P_1loop", self.X_spt, P, P_window, C_window)
        result = self.cache.get("P_1loop", hash_key)
        if result is not None: return result
        # get the roundtrip Fourier power spectrum, i.e. P=IFFT[FFT[P]]
        # get the matrix for each J_k component
        Ps, mat = self.J_k_scalar(P, self.X_spt, -2, P_window=P_window, C_window=C_window)
        P22_coef = np.array([2*1219/1470., 2*671/1029., 2*32/1715., 2*1/3., 2*62/35., 2*8/35., 1/3.])
        P22_mat = np.multiply(P22_coef, np.transpose(mat))
        P22 = np.sum(P22_mat, axis=1)
        P13 = P_13_reg(self.k_extrap, Ps)
        P_1loop = P22 + P13
        P_1loop = self._apply_extrapolation(P_1loop)
        self.cache.set(P_1loop, "P_1loop", hash_key, P_hash)
        return P_1loop


    #TODO add comments back explaining math behind one loop
    def one_loop_dd_bias(self, P, P_window=None, C_window=None):
        """
        Computes 1-loop corrections with standard bias terms.
    
        Returns
        -------
        tuple
            (P_1loop, Ps, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4) where:
        P_1loop : 1-loop correction (P_22 + P_13)
        Ps : Smoothed input power spectrum
        Pd1d2 : First order density-second order density correlation
        Pd2d2 : Second order density auto-correlation
        Pd1s2 : First order density-second order tidal correlation
        Pd2s2 : Second order density-second order tidal correlation
        Ps2s2 : Second order tidal auto-correlation
        sig4 : σ^4 integral for nonlinear bias
        """
        P_1loop, Ps = self.one_loop_dd(P, P_window=P_window, C_window=C_window)
        Pd1d2 = self._get_Pd1d2(P, P_window=P_window, C_window=C_window)
        Pd2d2 = self._get_Pd2d2(P, P_window=P_window, C_window=C_window)
        Pd1s2 = self._get_Pd1s2(P, P_window=P_window, C_window=C_window)
        Pd2s2 = self._get_Pd2s2(P, P_window=P_window, C_window=C_window)
        Ps2s2 = self._get_Ps2s2(P, P_window=P_window, C_window=C_window)
        sig4 = self._get_sig4(P, P_window=P_window, C_window=C_window)
        return P_1loop, Ps, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4
    
    def _get_sig4(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("sig4", self.X_spt, P, P_window, C_window)
        result = self.cache.get("sig4", hash_key)
        if result is not None: return result
        Ps, _ = self.J_k_scalar(P, self.X_spt, -2, P_window=P_window, C_window=C_window)
        # Quadraric bias Legendre components
        # See eg section B of Baldauf+ 2012 (arxiv: 1201.4827)
        # Note pre-factor convention is not standardized
        # Returns relevant correlations (including contraction factors),
        # but WITHOUT bias values and other pre-factors.
        # Uses standard "full initialization" of J terms
        sig4 = np.trapz(self.k_extrap ** 3 * Ps ** 2, x=np.log(self.k_extrap)) / (2. * pi ** 2)
        self.cache.set(sig4, "sig4", hash_key, P_hash)
        return sig4

    def _get_Pd1d2(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("Pd1d2", self.X_spt, P, P_window, C_window)
        result = self.cache.get("Pd1d2", hash_key)
        if result is not None: return result
        _, mat = self.J_k_scalar(P, self.X_spt, -2, P_window=P_window, C_window=C_window)
        Pd1d2 = 2. * (17. / 21 * mat[0, :] + mat[4, :] + 4. / 21 * mat[1, :])
        Pd1d2 = self._apply_extrapolation(Pd1d2)
        self.cache.set(Pd1d2, "Pd1d2", hash_key, P_hash)
        return Pd1d2
    
    def _get_Pd2d2(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("Pd2d2", self.X_spt, P, P_window, C_window)
        result = self.cache.get("Pd2d2", hash_key)
        if result is not None: return result
        _, mat = self.J_k_scalar(P, self.X_spt, -2, P_window=P_window, C_window=C_window)
        Pd2d2 = 2. * (mat[0, :])
        Pd2d2 = self._apply_extrapolation(Pd2d2)
        self.cache.set(Pd2d2, "Pd2d2", hash_key, P_hash)
        return Pd2d2
    
    def _get_Pd1s2(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("Pd1s2", self.X_spt, P, P_window, C_window)
        result = self.cache.get("Pd1s2", hash_key)
        if result is not None: return result
        _, mat = self.J_k_scalar(P, self.X_spt, -2, P_window=P_window, C_window=C_window)
        Pd1s2 = 2. * (8. / 315 * mat[0, :] + 4. / 15 * mat[4, :] + 254. / 441 * mat[1, :] + 2. / 5 * mat[5,
                                                                                                     :] + 16. / 245 * mat[
                                                                                                                      2,
                                                                                                                      :])
        Pd1s2 = self._apply_extrapolation(Pd1s2)
        self.cache.set(Pd1s2, "Pd1s2", hash_key, P_hash)
        return Pd1s2
    
    def _get_Pd2s2(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("Pd2s2", self.X_spt, P, P_window, C_window)
        result = self.cache.get("Pd2s2", hash_key)
        if result is not None: return result
        _, mat = self.J_k_scalar(P, self.X_spt, -2, P_window=P_window, C_window=C_window)
        Pd2s2 = 2. * (2. / 3 * mat[1, :])
        Pd2s2 = self._apply_extrapolation(Pd2s2)
        self.cache.set(Pd2s2, "Pd2s2", hash_key, P_hash)
        return Pd2s2
    
    def _get_Ps2s2(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("Ps2s2", self.X_spt, P, P_window, C_window)
        result = self.cache.get("Ps2s2", hash_key)
        if result is not None: return result
        _, mat = self.J_k_scalar(P, self.X_spt, -2, P_window=P_window, C_window=C_window)
        Pd2s2 = 2. * (4. / 45 * mat[0, :] + 8. / 63 * mat[1, :] + 8. / 35 * mat[2, :])
        Pd2s2 = self._apply_extrapolation(Pd2s2)
        self.cache.set(Pd2s2, "Ps2s2", hash_key, P_hash)
        return Pd2s2

    
    def one_loop_dd_bias_b3nl(self, P, P_window=None, C_window=None):
        """
        Computes 1-loop corrections with bias terms including third-order non-local bias.
    
        Returns
        -------
        tuple
            (P_1loop, Ps, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4, sig3nl) where:
        The first 8 terms are identical to those returned by one_loop_dd_bias
        sig3nl : Third order non-local bias term
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        P_1loop, Ps = self.one_loop_dd(P, P_window=P_window, C_window=C_window)
        Pd1d2 = self._get_Pd1d2(P, P_window=P_window, C_window=C_window)
        Pd2d2 = self._get_Pd2d2(P, P_window=P_window, C_window=C_window)
        Pd1s2 = self._get_Pd1s2(P, P_window=P_window, C_window=C_window)
        Pd2s2 = self._get_Pd2s2(P, P_window=P_window, C_window=C_window)
        Ps2s2 = self._get_Ps2s2(P, P_window=P_window, C_window=C_window)
        sig4 = self._get_sig4(P, P_window=P_window, C_window=C_window)
        sig3nl = self._get_sig3nl(P, P_window=P_window, C_window=C_window)
        return P_1loop, Ps, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4, sig3nl
    
    def _get_sig3nl(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("sig3nl", self.X_spt, P, P_window, C_window)
        result = self.cache.get("sig3nl", hash_key)
        if result is not None: return result
        Ps, _ = self.J_k_scalar(P, self.X_spt, -2, P_window=P_window, C_window=C_window)
        sig3nl = Y1_reg_NL(self.k_extrap, Ps)
        sig3nl = self._apply_extrapolation(sig3nl)
        self.cache.set(sig3nl, "sig3nl", hash_key, P_hash)
        return sig3nl

    
    def one_loop_dd_bias_lpt_NL(self, P, P_window=None, C_window=None):
        """
        Computes bias corrections in Lagrangian Perturbation Theory (LPT).
    
        Returns
        -------
        tuple
            (Ps, Pb1L, Pb1L_2, Pb1L_b2L, Pb2L, Pb2L_2, sig4) where:
        Ps : Smoothed input power spectrum
        Pb1L : First-order Lagrangian bias correlation term
        Pb1L_2 : First-order Lagrangian bias squared correlation
        Pb1L_b2L : First-order and second-order Lagrangian bias cross-correlation
        Pb2L : Second-order Lagrangian bias correlation
        Pb2L_2 : Second-order Lagrangian bias squared correlation 
        sig4 : σ^4 integral for nonlinear bias
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        _, Ps = self.one_loop_dd(P, P_window=P_window, C_window=C_window)
        Pb1L = self._get_Pb1L(P, P_window=P_window, C_window=C_window)
        Pb1L_2 = self._get_Pb1L_2(P, P_window=P_window, C_window=C_window)
        Pb1L_b2L = self._get_Pb1L_b2L(P, P_window=P_window, C_window=C_window)
        Pb2L = self._get_Pb2L(P, P_window=P_window, C_window=C_window)
        Pb2L_2 = self._get_Pb2L_2(P, P_window=P_window, C_window=C_window)
        sig4 = self._get_sig4(P, P_window=P_window, C_window=C_window)
        return Ps, Pb1L, Pb1L_2, Pb1L_b2L, Pb2L, Pb2L_2, sig4
    
    def _get_Pb1L(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("Pb1L", self.X_lpt, P, P_window, C_window)
        result = self.cache.get("Pb1L", hash_key)
        if result is not None: return result
        Ps, mat = self.J_k_scalar(P, self.X_lpt, -2, P_window=P_window, C_window=C_window)
        [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                          mat[5, :], mat[6, :]]
        X1 = ((144. / 245.) * j000 - (176. / 343.) * j002 - (128. / 1715.) * j004 + (16. / 35.) * j1n11 - (
                16. / 35.) * j1n13)
        Y1 = Y1_reg_NL(self.k_extrap, Ps)
        Pb1L = X1 + Y1
        Pb1L = self._apply_extrapolation(Pb1L)
        self.cache.set(Pb1L, "Pb1L", hash_key, P_hash)
        return Pb1L
    
    def _get_Pb1L_2(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("Pb1L_2", self.X_lpt, P, P_window, C_window)
        result = self.cache.get("Pb1L_2", hash_key)
        if result is not None: return result
        Ps, mat = self.J_k_scalar(P, self.X_lpt, -2, P_window=P_window, C_window=C_window)
        [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                          mat[5, :], mat[6, :]]
        X2 = ((16. / 21.) * j000 - (16. / 21.) * j002 + (16. / 35.) * j1n11 - (16. / 35.) * j1n13)
        Y2 = Y2_reg_NL(self.k_extrap, Ps)
        Pb1L_2 = X2 + Y2
        Pb1L_2 = self._apply_extrapolation(Pb1L_2)
        self.cache.set(Pb1L_2, "Pb1L_2", hash_key, P_hash)
        return Pb1L_2

    def _get_Pb1L_b2L(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("Pb1L_b2L", self.X_lpt, P, P_window, C_window)
        result = self.cache.get("Pb1L_b2L", hash_key)
        if result is not None: return result
        Ps, mat = self.J_k_scalar(P, self.X_lpt, -2, P_window=P_window, C_window=C_window)
        [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                          mat[5, :], mat[6, :]]
        X3 = (50. / 21.) * j000 + 2. * j1n11 - (8. / 21.) * j002
        Pb1L_b2L = X3
        Pb1L_b2L = self._apply_extrapolation(Pb1L_b2L)
        self.cache.set(Pb1L_b2L, "Pb1L_b2L", hash_key, P_hash)
        return Pb1L_b2L
    
    def _get_Pb2L(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("Pb2L", self.X_lpt, P, P_window, C_window)
        result = self.cache.get("Pb2L", hash_key)
        if result is not None: return result
        Ps, mat = self.J_k_scalar(P, self.X_lpt, -2, P_window=P_window, C_window=C_window)
        [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                          mat[5, :], mat[6, :]]
        X4 = (34. / 21.) * j000 + 2. * j1n11 + (8. / 21.) * j002
        Pb2L = X4
        Pb2L = self._apply_extrapolation(Pb2L)
        self.cache.set(Pb2L, "Pb2L", hash_key, P_hash)
        return Pb2L
    
    def _get_Pb2L_2(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("Pb2L_2", self.X_lpt, P, P_window, C_window)
        result = self.cache.get("Pb2L_2", hash_key)
        if result is not None: return result
        Ps, mat = self.J_k_scalar(P, self.X_lpt, -2, P_window=P_window, C_window=C_window)
        [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                          mat[5, :], mat[6, :]]
        X5 = j000
        Pb2L_2 = X5
        Pb2L_2 = self._apply_extrapolation(Pb2L_2)
        self.cache.set(Pb2L_2, "Pb2L_2", hash_key, P_hash)
        return Pb2L_2

    def IA_tt(self, P, P_window=None, C_window=None):
        """
        Computes intrinsic alignment tidal torque contributions.
    
        Returns
        -------
        tuple
            (P_E, P_B) where:
        P_E : E-mode (curl-free) tidal torque power spectrum
        P_B : B-mode (divergence-free) tidal torque power spectrum
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        P_E = self.compute_term("P_E", self.X_IA_E, operation=lambda x: 2 * x, 
                                 P=P, P_window=P_window, C_window=C_window)
        P_B = self.compute_term("P_B", self.X_IA_B, operation=lambda x: 2 * x,
                                 P=P, P_window=P_window, C_window=C_window)
        return P_E, P_B

    ## eq 21 EE; eq 21 BB

    def IA_mix(self, P, P_window=None, C_window=None):
        """
        Computes mixed intrinsic alignment contributions combining tidal 
        alignment and tidal torque.
    
        Returns
        -------
        tuple
            (P_A, P_Btype2, P_DEE, P_DBB) where:
        P_A : Mixed tidal alignment/tidal torque term
        P_Btype2 : Second-type B-mode term
        P_DEE : Contribution to the E-mode power spectrum
        P_DBB : Contribution to the B-mode power spectrum
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        P_A = self.compute_term("P_A", self.X_IA_A, operation=lambda x: 2 * x, 
                                 P=P, P_window=P_window, C_window=C_window)
        P_Btype2 = self._get_P_Btype2(P) #Calculated differently then other terms, can't use compute_term
        P_DEE = self.compute_term("P_DEE", self.X_IA_DEE, operation=lambda x: 2 * x, 
                                   P=P, P_window=P_window, C_window=C_window)
        P_DBB = self.compute_term("P_DBB", self.X_IA_DBB, operation=lambda x: 2 * x,
                                   P=P, P_window=P_window, C_window=C_window)
        return P_A, P_Btype2, P_DEE, P_DBB
    
    def _get_P_Btype2(self, P):
        hash_key, P_hash = self._create_hash_key("P_Btype2", None, P, None, None)
        result = self.cache.get("P_Btype2", hash_key)
        if result is not None: return result
        P_Btype2 = P_IA_B(self.k_original, P)
        P_Btype2 = 4 * P_Btype2
        self.cache.set(P_Btype2, "P_Btype2", hash_key, P_hash)
        return P_Btype2

    ## eq 18; eq 19; eq 27 EE; eq 27 BB

    
    def IA_ta(self, P, P_window=None, C_window=None):
        """
        Computes intrinsic alignment tidal alignment contributions.
    
        Returns
        -------
        tuple
            (P_deltaE1, P_deltaE2, P_0E0E, P_0B0B) where:
        P_deltaE1 : First density-E mode correlation
        P_deltaE2 : Second density-E mode correlation
        P_0E0E : E-mode auto-correlation
        P_0B0B : B-mode auto-correlation
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        P_deltaE1 = self.compute_term("P_deltaE1", self.X_IA_deltaE1, operation=lambda x: 2 * x, 
                                       P=P, P_window=P_window, C_window=C_window)
        P_deltaE2 = self._get_P_deltaE2(P) #Calculated differently then other terms, can't use compute_term
        P_0E0E = self.compute_term("P_0E0E", self.X_IA_0E0E, P=P, P_window=P_window, C_window=C_window)
        P_0B0B = self.compute_term("P_0B0B", self.X_IA_0B0B, P=P, P_window=P_window, C_window=C_window)
        return P_deltaE1, P_deltaE2, P_0E0E, P_0B0B
    
    def _get_P_deltaE2(self, P):
        hash_key, P_hash = self._create_hash_key("P_deltaE2", None, P, None, None)
        result = self.cache.get("P_deltaE2", hash_key)
        if result is not None: return result
        P_deltaE2 = P_IA_deltaE2(self.k_original, P)
        #Add extrap?
        P_deltaE2 = 2 * P_deltaE2
        self.cache.set(P_deltaE2, "P_deltaE2", hash_key, P_hash)
        return P_deltaE2

    ## eq 12 (line 2); eq 12 (line 3); eq 15 EE; eq 15 BB

    
    def IA_der(self, P, P_window=None, C_window=None):
        """
        Computes k^2 * P(k) derivative term for intrinsic alignment models.
    
        Returns
        -------
        array_like
            P_der : Derivative term of the power spectrum
        """
        hash_key, P_hash = self._create_hash_key("IA_der", None, P, P_window, C_window)
        result = self.cache.get("IA_der", hash_key)
        if result is not None: return result
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        P_der = (self.k_original**2)*P
        self.cache.set(P_der, "IA_der", hash_key, P_hash)
        return P_der
    
    def IA_ct(self,P,P_window=None, C_window=None):
        """
        Computes intrinsic alignment velocity-shear contributions.
    
        Returns
        -------
        tuple
            (P_0tE, P_0EtE, P_E2tE, P_tEtE) where:
        P_0tE : Density-velocity E-mode correlation
        P_0EtE : E-mode-velocity E-mode correlation
        P_E2tE : Second torquing-velocity E-mode correlation
        P_tEtE : Velocity E-mode auto-correlation
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        P_0tE = self._get_P_0tE(P, P_window=P_window, C_window=C_window)
        P_0EtE = self._get_P_0EtE(P, P_window=P_window, C_window=C_window)
        P_E2tE = self._get_P_E2tE(P, P_window=P_window, C_window=C_window)
        P_tEtE = self._get_P_tEtE(P, P_window=P_window, C_window=C_window)
        return P_0tE,P_0EtE,P_E2tE,P_tEtE
    
    def _get_P_0tE(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("P_0tE", self.X_spt, P, P_window, C_window)
        result = self.cache.get("P_0tE", hash_key)
        if result is not None: return result
        nu=-2
        Ps, mat = self.J_k_scalar(P, self.X_spt, nu, P_window=P_window, C_window=C_window)
        one_loop_coef = np.array(
            [2 * 1219 / 1470., 2 * 671 / 1029., 2 * 32 / 1715., 2 * 1 / 3., 2 * 62 / 35., 2 * 8 / 35., 1 / 3.])
        P22_mat = np.multiply(one_loop_coef, np.transpose(mat))
        P_22F = np.sum(P22_mat, 1)

        one_loop_coefG= np.array(
            [2*1003/1470, 2*803/1029, 2*64/1715, 2*1/3, 2*58/35, 2*12/35, 1/3])
        PsG, matG = self.J_k_scalar(P, self.X_sptG, nu, P_window=P_window, C_window=C_window)
        P22G_mat = np.multiply(one_loop_coefG, np.transpose(matG))
        P_22G = np.sum(P22G_mat, 1)
        P_22F, P_22G = self._apply_extrapolation(P_22F, P_22G)
        P_13G = P_IA_13G(self.k_original,P,)
        P_13F = P_IA_13F(self.k_original, P)
        P_0tE = P_22G-P_22F+P_13G-P_13F
        P_0tE = 2*P_0tE
        self.cache.set(P_0tE, "P_0tE", hash_key, P_hash)
        return P_0tE
    
    def _get_P_0EtE(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("P_0EtE", self.X_spt, P, P_window, C_window)
        result = self.cache.get("P_0EtE", hash_key)
        if result is not None: return result
        P_feG2, A = self.J_k_tensor(P,self.X_IA_tij_feG2, P_window=P_window, C_window=C_window)
        P_feG2 = self._apply_extrapolation(P_feG2)
        P_A00E = self.compute_term("P_deltaE1", self.X_IA_deltaE1, operation=lambda x: 2 * x, 
                                       P=P, P_window=P_window, C_window=C_window) #OG: P_A00E, _, _, _ = self.IA_ta()
        P_0EtE = np.subtract(P_feG2,(1/2)*P_A00E)
        P_0EtE = 2*P_0EtE
        self.cache.set(P_0EtE, "P_0EtE", hash_key, P_hash)
        return P_0EtE
    
    def _get_P_E2tE(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("P_E2tE", self.X_spt, P, P_window, C_window)
        result = self.cache.get("P_E2tE", hash_key)
        if result is not None: return result
        P_heG2, A = self.J_k_tensor(P,self.X_IA_tij_heG2, P_window=P_window, C_window=C_window)
        P_heG2 = self._apply_extrapolation(P_heG2)
        P_A0E2 = self.compute_term("P_A", self.X_IA_A, operation=lambda x: 2 * x, 
                                 P=P, P_window=P_window, C_window=C_window) #OG: P_A0E2, _, _, _ = self.IA_mix()
        P_E2tE = np.subtract(P_heG2,(1/2)*P_A0E2)
        P_E2tE = 2*P_E2tE
        self.cache.set(P_E2tE, "P_E2tE", hash_key, P_hash)
        return P_E2tE
    
    def _get_P_tEtE(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("P_tEtE", self.X_spt, P, P_window, C_window)
        result = self.cache.get("P_tEtE", hash_key)
        if result is not None: return result
        P_F2F2, A = self.J_k_tensor(P,self.X_IA_tij_F2F2, P_window=P_window, C_window=C_window)
        P_G2G2, A = self.J_k_tensor(P,self.X_IA_tij_G2G2, P_window=P_window, C_window=C_window)
        P_F2G2, A = self.J_k_tensor(P,self.X_IA_tij_F2G2, P_window=P_window, C_window=C_window)
        P_F2F2, P_G2G2, P_F2G2 = self._apply_extrapolation(P_F2F2, P_G2G2, P_F2G2)
        P_tEtE = P_F2F2+P_G2G2-2*P_F2G2
        P_tEtE = 2*P_tEtE
        self.cache.set(P_tEtE, "P_tEtE", hash_key, P_hash)
        return P_tEtE

    def gI_ct(self,P,P_window=None, C_window=None):
        """
        Computes galaxy bias cross intrinsic alignment velocity-shear contributions.
    
        Returns
        -------
        tuple
            (P_d2tE, P_s2tE) where:
        P_d2tE : Second-order density-velocity E-mode correlation
        P_s2tE : Second-order tidal-velocity E-mode correlation
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        P_d2tE = self._get_P_d2tE(P, P_window=P_window, C_window=C_window)
        P_s2tE = self._get_P_s2tE(P, P_window=P_window, C_window=C_window)
        return P_d2tE, P_s2tE

    def _get_P_d2tE(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("Pd2tE", self.X_IA_gb2_F2, P, P_window, C_window)
        result = self.cache.get("Pd2tE", hash_key)
        if result is not None: return result
        P_F2, _ = self.J_k_tensor(P, self.X_IA_gb2_F2, P_window=P_window, C_window=C_window)
        P_G2, _ = self.J_k_tensor(P, self.X_IA_gb2_G2, P_window=P_window, C_window=C_window)
        P_F2, P_G2 = self._apply_extrapolation(P_F2, P_G2)
        P_d2tE = 2 * (P_G2 - P_F2)
        self.cache.set(P_d2tE, "Pd2tE", hash_key, P_hash)
        return P_d2tE
    
    def _get_P_s2tE(self, P, P_window=None, C_window=None):
        hash_key, P_hash = self._create_hash_key("P_s2tE", self.X_IA_gb2_S2F2, P, P_window, C_window)
        result = self.cache.get("P_s2tE", hash_key)
        if result is not None: return result
        P_S2F2, _ = self.J_k_tensor(P, self.X_IA_gb2_S2F2, P_window=P_window, C_window=C_window)
        P_S2G2, _ = self.J_k_tensor(P, self.X_IA_gb2_S2G2, P_window=P_window, C_window=C_window)
        P_S2F2, P_S2G2 = self._apply_extrapolation(P_S2F2, P_S2G2)
        P_s2tE = 2 * (P_S2G2 - P_S2F2)
        self.cache.set(P_s2tE, "P_s2tE", hash_key, P_hash)
        return P_s2tE
    
    def gI_ta(self,P,P_window=None, C_window=None):
        """
        Computes intrinsic alignment 2nd-order density correlations.
    
        Returns
        -------
        tuple
            (P_d2E, P_d20E, P_s2E, P_s20E) where:
        P_d2E : 2nd-order density-E-mode correlation
        P_d20E : 2nd-order density-density-E-mode correlation
	P_s2E : 2nd-order tidal-E-mode correlation
        P_s20E : 2nd-order tidal-density-E-mode correlation
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        P_d2E = self.compute_term("P_d2E", self.X_IA_gb2_F2, operation=lambda x: 2 * x,
                                   P=P, P_window=P_window, C_window=C_window)
        P_d20E = self.compute_term("P_d20E", self.X_IA_gb2_fe, operation=lambda x: 2 * x,
                                    P=P, P_window=P_window, C_window=C_window)
        P_s2E = self.compute_term("P_s2E", self.X_IA_gb2_S2F2, operation=lambda x: 2 * x,
                                   P=P, P_window=P_window, C_window=C_window)
        P_s20E = self.compute_term("P_s20E", self.X_IA_gb2_S2fe, operation=lambda x: 2 * x,
                                    P=P, P_window=P_window, C_window=C_window)
        return P_d2E, P_d20E, P_s2E, P_s20E

    def gI_tt(self, P, P_window=None, C_window=None):
        """
        Computes intrinsic alignment 2nd-order tidal correlations.
    
        Returns
        -------
        tuple
            (P_s2E2, P_d2E2) where:
        P_s2E2 : 2nd-order tidal-E-mode squared correlation
	P_d2E2 : 2nd-order density-E-mode squared correlation
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        P_s2E2 = self.compute_term("P_s2E2", self.X_IA_gb2_S2he, operation=lambda x: 2 * x,
                                    P=P, P_window=P_window, C_window=C_window)
        P_d2E2 = self.compute_term("P_d2E2", self.X_IA_gb2_he, operation=lambda x: 2 * x,
                                    P=P, P_window=P_window, C_window=C_window)
        return P_s2E2, P_d2E2

    
    def OV(self, P, P_window=None, C_window=None):
        """
        Computes the Ostriker-Vishniac effect power spectrum.
    
        Returns
        -------
        array_like
            P_OV : Ostriker-Vishniac effect power spectrum
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        hash_key, P_hash = self._create_hash_key("OV", None, P, P_window, C_window)
        result = self.cache.get("P_OV", hash_key)
        if result is not None: return result
        P, A = self.J_k_tensor(P, self.X_OV, P_window=P_window, C_window=C_window)
        P = self._apply_extrapolation(P)
        P_OV = P * (2 * pi) ** 2
        self.cache.set(P_OV, "P_OV", hash_key, P_hash)
        return P_OV

    
    def kPol(self, P, P_window=None, C_window=None):
        """
        Computes k-dependent polarization power spectra.
    
        Returns
        -------
        tuple
            (P1, P2, P3) where:
        P1 : First k-dependent polarization power spectrum
        P2 : Second k-dependent polarization power spectrum
        P3 : Third k-dependent polarization power spectrum
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        P1 = self.compute_term("P_kP1", self.X_kP1, operation=lambda x: x / (80 * pi ** 2),
                                P=P, P_window=P_window, C_window=C_window)
        P2 = self.compute_term("P_kP2", self.X_kP2, operation=lambda x: x / (160 * pi ** 2),
                                P=P, P_window=P_window, C_window=C_window)
        P3 = self.compute_term("P_kP3", self.X_kP3, operation=lambda x: x / (80 * pi ** 2),
                                P=P, P_window=P_window, C_window=C_window)
        return P1, P2, P3


    def RSD_components(self, P, f, P_window=None, C_window=None):
        """
        Computes redshift-space distortion component terms.
    
        Parameters
        ----------
        f : float
            Logarithmic growth rate
    
        Returns
        -------
        tuple
            (A1, A3, A5, B0, B2, B4, B6, P_Ap1, P_Ap3, P_Ap5) where:
        A1, A3, A5 : A-type RSD components with different powers of μ
        B0, B2, B4, B6 : B-type RSD components with different powers of μ
        P_Ap1, P_Ap3, P_Ap5 : Additional RSD A-prime components
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        _, A = self.J_k_tensor(P, self.X_RSDA, P_window=P_window, C_window=C_window)

        A1 = np.dot(self.A_coeff[:, 0], A) + f * np.dot(self.A_coeff[:, 1], A) + f ** 2 * np.dot(self.A_coeff[:, 2], A)
        A3 = np.dot(self.A_coeff[:, 3], A) + f * np.dot(self.A_coeff[:, 4], A) + f ** 2 * np.dot(self.A_coeff[:, 5], A)
        A5 = np.dot(self.A_coeff[:, 6], A) + f * np.dot(self.A_coeff[:, 7], A) + f ** 2 * np.dot(self.A_coeff[:, 8], A)

        _, B = self.J_k_tensor(P, self.X_RSDB, P_window=P_window, C_window=C_window)

        B0 = np.dot(self.B_coeff[:, 0], B) + f * np.dot(self.B_coeff[:, 1], B) + f ** 2 * np.dot(self.B_coeff[:, 2], B)
        B2 = np.dot(self.B_coeff[:, 3], B) + f * np.dot(self.B_coeff[:, 4], B) + f ** 2 * np.dot(self.B_coeff[:, 5], B)
        B4 = np.dot(self.B_coeff[:, 6], B) + f * np.dot(self.B_coeff[:, 7], B) + f ** 2 * np.dot(self.B_coeff[:, 8], B)
        B6 = np.dot(self.B_coeff[:, 9], B) + f * np.dot(self.B_coeff[:, 10], B) + f ** 2 * np.dot(self.B_coeff[:, 11], B)

        A1, A3, A5, B0, B2, B4, B6 = self._apply_extrapolation(A1, A3, A5, B0, B2, B4, B6)

        P_Ap1 = RSD_P_Ap1(self.k_original, P, f)
        P_Ap3 = RSD_P_Ap3(self.k_original, P, f)
        P_Ap5 = RSD_P_Ap5(self.k_original, P, f)

        return A1, A3, A5, B0, B2, B4, B6, P_Ap1, P_Ap3, P_Ap5

    
    def RSD_ABsum_components(self, P, f, P_window=None, C_window=None):
        """
        Computes combined redshift-space distortion terms by powers of μ.
    
        Parameters
        ----------
        f : float
            Logarithmic growth rate
    
        Returns
        -------
        tuple
            (ABsum_mu2, ABsum_mu4, ABsum_mu6, ABsum_mu8) where:
        ABsum_mu2 : Combined term with μ^2 dependence
        ABsum_mu4 : Combined term with μ^4 dependence
        ABsum_mu6 : Combined term with μ^6 dependence
        ABsum_mu8 : Combined term with μ^8 dependence
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        A1, A3, A5, B0, B2, B4, B6, P_Ap1, P_Ap3, P_Ap5 = self.RSD_components(P, f, P_window, C_window)
        ABsum_mu2 = self.k_original * f * (A1 + P_Ap1) + (f * self.k_original) ** 2 * B0
        ABsum_mu4 = self.k_original * f * (A3 + P_Ap3) + (f * self.k_original) ** 2 * B2
        ABsum_mu6 = self.k_original * f * (A5 + P_Ap5) + (f * self.k_original) ** 2 * B4
        ABsum_mu8 = (f * self.k_original) ** 2 * B6

        return ABsum_mu2, ABsum_mu4, ABsum_mu6, ABsum_mu8

    
    def RSD_ABsum_mu(self, P, f, mu_n, P_window=None, C_window=None):
        """
        Computes the total redshift-space distortion correction at a given μ.
    
        Parameters
        ----------
        f : float
            Logarithmic growth rate
        mu_n : float
            Cosine of the angle between the wavevector and the line-of-sight
    
        Returns
        -------
        array_like
            ABsum : The total RSD contribution at the specified μ angle
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        ABsum_mu2, ABsum_mu4, ABsum_mu6, ABsum_mu8 = self.RSD_ABsum_components(P, f, P_window, C_window)
        ABsum = ABsum_mu2 * mu_n ** 2 + ABsum_mu4 * mu_n ** 4 + ABsum_mu6 * mu_n ** 6 + ABsum_mu8 * mu_n ** 8
        return ABsum

    
    def IRres(self, P, L=0.2, h=0.67, rsdrag=135, P_window=None, C_window=None):
        """
        Computes the IR-resummed power spectrum, which includes BAO damping.
    
        Parameters
        ----------
        L : float, optional
            IR resummation scale, default is 0.2
        h : float, optional
            Dimensionless Hubble parameter, default is 0.67
        rsdrag : float, optional
            Sound horizon at drag epoch in Mpc, default is 135
    
        Returns
        -------
        array_like
            P_IRres : IR-resummed power spectrum with damped BAO features
        """
        self._validate_params(P=P, P_window=P_window, C_window=C_window)
        # based on script by M. Ivanov. See arxiv:1605.02149, eq 7.4

        # put this function in the typical fast-pt format, with minimal additional function calls.
        # We can also include a script to do additional things: e.g read in r_BAO from class/camb output
        # or calculate r_BAO from cosmological params.
        from scipy import interpolate
        k = self.k_original
        rbao = h * rsdrag * 1.027  # linear BAO scale
        # set up splining to create P_nowiggle
        kmin = k[0]
        kmax = k[-1]
        knode1 = pi / rbao
        knode2 = 2 * pi / rbao
        klogleft = np.arange(log(kmin), log(3.e-3), 0.2)
        klogright = np.arange(log(0.6), log(kmax), 0.085)
        klogright = np.hstack((log(knode1), log(knode2), klogright))
        kloglist = np.concatenate((klogleft, klogright))
        klist = np.exp(kloglist)

        # how to deal with extended k and P? which values should be used here? probably the extended versions?
        plin = interpolate.InterpolatedUnivariateSpline(k, P)
        logPs = np.log(plin(klist))
        logpsmooth = interpolate.InterpolatedUnivariateSpline(kloglist, logPs)

        def psmooth(x):
            return exp(logpsmooth(log(x)))

        def pw(x):
            return plin(x) - psmooth(x)

        # compute Sigma^2 and the tree-level IR-resummed PS
        import scipy.integrate as integrate

        Sigma = integrate.quad(lambda x: (4 * pi) * psmooth(x) * (
                1 - 3 * (2 * rbao * x * cos(x * rbao) + (-2 + rbao ** 2 * x ** 2) * sin(rbao * x)) / (
                x * rbao) ** 3) / (3 * (2 * pi) ** 3), kmin, L)[0]

        # speed up by using trap rule integration?
        # change to integration over log-k(?):
        # 		Sigma = integrate.quad(lambda x: x*(4*pi)*psmooth(x)*(1-3*(2*rbao*x*cos(x*rbao)+(-2+rbao**2*x**2)*sin(rbao*x))/(x*rbao)**3)/(3*(2*pi)**3), np.log(kmin), np.log(L))[0]
        def presum(x):
            return psmooth(x) + pw(x) * exp(-x ** 2 * Sigma)

        P = presum(k)
        out_1loop = self._get_P_1loop(P, P_window=P_window, C_window=C_window)
        # p1loop = interpolate.InterpolatedUnivariateSpline(k,out_1loop) # is this necessary? out_1loop should already be at the correct k spacing
        return psmooth(k) + out_1loop + pw(k) * exp(-k ** 2 * Sigma) * (1 + Sigma * k ** 2)

    ######################################################################################
    ### functions that use the older version structures. ###
    
    def one_loop(self, P, P_window=None, C_window=None):

        return self.pt_simple.one_loop(P, P_window=P_window, C_window=C_window)

    
    def P_bias(self, P, P_window=None, C_window=None):

        return self.pt_simple.P_bias(P, P_window=P_window, C_window=C_window)

    ######################################################################################
    ### Core functions used by top-level functions ###

    def J_k_scalar(self, P, X, nu, P_window=None, C_window=None):
        
        hash_key, P_hash = self._create_hash_key("J_k_scalar", X, P, P_window, C_window)
        result = self.cache.get("J_k_scalar", hash_key)
        if result is not None: return result

        pf, p, g_m, g_n, two_part_l, h_l = X

        if (self.low_extrap is not None):
            P = self.EK.extrap_P_low(P)

        if (self.high_extrap is not None):
            P = self.EK.extrap_P_high(P)

        P_b = P * self.k_extrap ** (-nu)

        if (self.n_pad > 0):
            P_b = np.pad(P_b, pad_width=(self.n_pad, self.n_pad), mode='constant', constant_values=0)

        c_m = self._cache_fourier_coefficients(P_b, C_window, scalar=True)

        A_out = np.zeros((pf.shape[0], self.k_size))
        for i in range(pf.shape[0]):
            # convolve f_c and g_c
            # C_l=np.convolve(c_m*self.g_m[i,:],c_m*self.g_n[i,:])
            C_l = self._cache_convolution(c_m, c_m, g_m[i,:], g_n[i,:], h_l[i,:], two_part_l[i])

            # set up to feed ifft an array ordered with l=0,1,...,-1,...,N/2-1
            c_plus = C_l[self.l >= 0]
            c_minus = C_l[self.l < 0]

            C_l = np.hstack((c_plus[:-1], c_minus))
            A_k = ifft(C_l) * C_l.size  # multiply by size to get rid of the normalization in ifft

            A_out[i, :] = np.real(A_k[::2]) * pf[i] * self.k_final ** (-p[i] - 2)
        # note that you have to take every other element
        # in A_k, due to the extended array created from the
        # discrete convolution

        P_out = irfft(c_m[self.m >= 0]) * self.k_final ** nu * float(self.N)
        if (self.n_pad > 0):
            # get rid of the elements created from padding
            P_out = P_out[self.id_pad]
            A_out = A_out[:, self.id_pad]

        self.cache.set((P_out, A_out), "J_k_scalar", hash_key, P_hash)
        return P_out, A_out

    def J_k_tensor(self, P, X, P_window=None, C_window=None):

        hash_key, P_hash = self._create_hash_key("J_k_tensor", X, P, P_window, C_window)
        result = self.cache.get("J_k_tensor", hash_key)
        if result is not None: return result

        pf, p, nu1, nu2, g_m, g_n, h_l = X

        if (self.low_extrap is not None):
            P = self.EK.extrap_P_low(P)

        if (self.high_extrap is not None):
            P = self.EK.extrap_P_high(P)

        A_out = np.zeros((pf.size, self.k_size))

        P_fin = np.zeros(self.k_size)

        for i in range(pf.size):

            P_b1 = P * self.k_extrap ** (-nu1[i])
            P_b2 = P * self.k_extrap ** (-nu2[i])

            if (P_window is not None):
                # window the input power spectrum, so that at high and low k
                # the signal smoothly tappers to zero. This make the input
                # more "like" a periodic signal

                if (self.verbose):
                    print('windowing biased power spectrum')
                W = p_window(self.k_extrap, P_window[0], P_window[1])
                P_b1 = P_b1 * W
                P_b2 = P_b2 * W

            if (self.n_pad > 0):
                P_b1 = np.pad(P_b1, pad_width=(self.n_pad, self.n_pad), mode='constant', constant_values=0)
                P_b2 = np.pad(P_b2, pad_width=(self.n_pad, self.n_pad), mode='constant', constant_values=0)
            c_m = self._cache_fourier_coefficients(P_b1, C_window, scalar=False)
            c_n = self._cache_fourier_coefficients(P_b2, C_window, scalar=False)
            # convolve f_c and g_c
            C_l = self._cache_convolution(c_m, c_n, g_m[i,:], g_n[i,:], h_l[i,:])

            # set up to feed ifft an array ordered with l=0,1,...,-1,...,N/2-1
            c_plus = C_l[self.l >= 0]
            c_minus = C_l[self.l < 0]

            C_l = np.hstack((c_plus[:-1], c_minus))
            A_k = ifft(C_l) * C_l.size  # multiply by size to get rid of the normalization in ifft

            A_out[i, :] = np.real(A_k[::2]) * pf[i] * self.k_final ** (p[i])
            # note that you have to take every other element
            # in A_k, due to the extended array created from the
            # discrete convolution
            P_fin += A_out[i, :]
        # P_out=irfft(c_m[self.m>=0])*self.k**self.nu*float(self.N)
        if (self.n_pad > 0):
            # get rid of the elements created from padding
            # P_out=P_out[self.id_pad]
            A_out = A_out[:, self.id_pad]
            P_fin = P_fin[self.id_pad]

        self.cache.set((P_fin, A_out), "J_k_tensor", hash_key, P_hash)
        return P_fin, A_out

    def _cache_fourier_coefficients(self, P_b, C_window=None, scalar=False):
        """Cache and return Fourier coefficients for a given biased power spectrum"""
    
        hash_key, P_hash = self._create_hash_key("fourier_coefficients", None, P_b, None, C_window)
        hash_key = hash_key ^ (hash(scalar) + 0x9e3779b9 + (hash_key << 6) + (hash_key >> 2))
        result = self.cache.get("fourier_coefficients", hash_key)
        if result is not None: 
            return result
    
        c_m_positive = rfft(P_b)
        # We always filter the Fourier coefficients, so the last element is zero.
        # But in case someone does not filter and this is a scalar calculation, divide the end point by two
        if scalar: c_m_positive[-1] = c_m_positive[-1] / 2. 
        c_m_negative = np.conjugate(c_m_positive[1:])
        c_m = np.hstack((c_m_negative[::-1], c_m_positive)) / float(self.N)
    
        if C_window is not None:
            # Window the Fourier coefficients.
            # This will damp the highest frequencies
            if self.verbose:
                print('windowing the Fourier coefficients')
            c_m = c_m * c_window(self.m, int(C_window * self.N // 2.))
        self.cache.set(c_m, "fourier_coefficients", hash_key, None)
        return c_m

    def _cache_convolution(self, c1, c2, g_m, g_n, h_l, two_part_l=None):
        """Cache and return convolution results"""

        # Use MurmurHash for faster hashing with low collision rate, original hashing method is much slower for this case
        def fast_hash(arr):
            if arr is None:
                return 0
            if isinstance(arr, np.ndarray):
                if arr.size > 1000:
                    sample = arr.ravel()[:1000]
                    return hash(sample.tobytes()) ^ hash(arr.shape) ^ hash(arr.size)
                return hash(arr.tobytes()) ^ hash(arr.shape)
            return hash(arr)
        
        # Combine hashes with different prime multipliers to reduce collisions
        hash_key = 0
        primes = [17, 31, 61, 127, 257, 509]
        arrays = [c1, c2, g_m, g_n, h_l, two_part_l]
        
        for i, arr in enumerate(arrays):
            h = fast_hash(arr)
            hash_key = hash_key ^ (h * primes[i % len(primes)])

        result = self.cache.get("convolution", hash_key)
        if result is not None: 
            return result

        # Calculate convolution
        C_l = fftconvolve(c1 * g_m, c2 * g_n)
        # C_l=convolve(c_m*self.g_m[i,:],c_m*self.g_n[i,:])
        # multiply all l terms together
        # C_l=C_l*self.h_l[i,:]*self.two_part_l[i]
    
        # Apply additional terms
        if two_part_l is not None:
            C_l = C_l * h_l * two_part_l
        else:
            C_l = C_l * h_l
        # Cache and return
        self.cache.set(C_l, "convolution", hash_key, None)
        return C_l
