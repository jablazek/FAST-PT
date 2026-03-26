import pytest
import numpy as np
from fastpt import FASTPT
import os
import warnings
import sys
import platform

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
d = np.loadtxt(data_path)
P = d[:, 1]    
k = d[:, 0]
C_window = 0.75

@pytest.fixture
def fpt(): 

    n_pad = int(0.5 * len(k))
    return FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad)

from matplotlib import pyplot as plt

def calc_and_show(bmark, stored, func):
    if not np.allclose(bmark, stored):
        print(f"Max difference: {np.max(np.abs(bmark - stored))}")
        rel_diff = np.where(np.abs(stored) > 1e-10,
                            np.abs(bmark - stored) / np.abs(stored),
                            0)
        print(f"Relative difference: {np.max(rel_diff)}")
        idx = np.searchsorted(k, 10.0)
        print(f"Max difference up until k=10: {np.max(np.abs(bmark - stored)[:idx+1])}")
        print(f"Max Relative difference up until k=10: {np.max(rel_diff[:idx+1])}")
        plt.plot(k, rel_diff, label='Relative difference')
        plt.title(f'Relative difference for {func}')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

@pytest.mark.skipif(
        sys.version_info >= (3, 13) or platform.machine() == "arm64",
        reason="Strict benchmark comparison is not reliable on Python 3.13+ or ARM64 runners"
    )
def test_one_loop_dd(fpt):
    bmark = fpt.one_loop_dd(P, C_window=C_window)[0]
    stored = np.loadtxt('tests/benchmarking/P_dd_benchmark.txt')
    # calc_and_show(bmark, stored, "one_loop_dd")
    if np.__version__ >= '2.0':
        warnings.warn("The benchmarks were generated with NumPy 1.x, the P_1loop term is known to fail np.allclose when using NumPy 2.x." +
                      " We can guarantee a precision of 9e-5 up until a k value of 10.")
    assert np.allclose(bmark, stored)

@pytest.mark.skipif(
        sys.version_info >= (3, 13) or platform.machine() == "arm64",
        reason="Strict benchmark comparison is not reliable on Python 3.13+ or ARM64 runners"
    )
def test_one_loop_dd_bias(fpt):
    bmark = list(fpt.one_loop_dd_bias(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[7]
    bmark[7] = new_array
    
    stored = np.transpose(np.loadtxt('tests/benchmarking/P_bias_benchmark.txt'))
    # calc_and_show(bmark[0], stored[0], "one_loop_dd_bias")
    if np.__version__ >= '2.0':
        warnings.warn("The benchmarks were generated with NumPy 1.x, the P_1loop term is known to fail np.allclose when using NumPy 2.x." +
                      " We can guarantee a precision of 9e-5 up until a k value of 10.")
    assert np.allclose(bmark, stored)

@pytest.mark.skipif(
        sys.version_info >= (3, 13) or platform.machine() == "arm64",
        reason="Strict benchmark comparison is not reliable on Python 3.13+ or ARM64 runners"
    )
def test_one_loop_dd_bias_b3nl(fpt):
    bmark = list(fpt.one_loop_dd_bias_b3nl(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[7]
    bmark[7] = new_array

    stored = np.transpose(np.loadtxt('tests/benchmarking/P_bias_b3nl_benchmark.txt'))
    # calc_and_show(bmark[8], stored[8], "one_loop_dd_bias_b3nl")
    if np.__version__ >= '2.0':
        warnings.warn("The benchmarks were generated with NumPy 1.x, the P_1loop and sig3nl terms are known to fail np.allclose when using NumPy 2.x." +
                      " We can guarantee a precision of 6e-4 up until a k value of 10.")
    assert np.allclose(bmark, stored)

@pytest.mark.skipif(
        sys.version_info >= (3, 13) or platform.machine() == "arm64",
        reason="Strict benchmark comparison is not reliable on Python 3.13+ or ARM64 runners"
    )
def test_one_loop_dd_bias_lpt_NL(fpt):
    bmark = list(fpt.one_loop_dd_bias_lpt_NL(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[6]
    bmark[6] = new_array
    
    stored = np.transpose(np.loadtxt('tests/benchmarking/P_bias_lpt_NL_benchmark.txt'))
    # calc_and_show(bmark[1], stored[1], "one_loop_dd_bias_lpt_NL")
    # calc_and_show(bmark[2], stored[2], "one_loop_dd_bias_lpt_NL")
    if np.__version__ >= '2.0':
        warnings.warn("The benchmarks were generated with NumPy 1.x, the Pb1L and Pb1L_2 terms are known to fail np.allclose when using NumPy 2.x." +
                      " We can guarantee a precision of 2e-4 up until a k value of 10.")
    assert np.allclose(bmark, stored)

def test_IA_TT(fpt):
    bmark = np.transpose(fpt.IA_tt(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/PIA_tt_benchmark.txt'))

def test_IA_mix(fpt):
    bmark = np.transpose(fpt.IA_mix(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_IA_mix_benchmark.txt'))

def test_IA_ta(fpt):
    bmark = np.transpose(fpt.IA_ta(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_IA_ta_benchmark.txt'))

def test_IA_der(fpt):
    bmark = np.transpose(fpt.IA_der(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_IA_der_benchmark.txt'))

def test_IA_ct(fpt):
    bmark = np.transpose(fpt.IA_ct(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_IA_ct_benchmark.txt'))

def test_gI_ct(fpt):
    bmark = np.transpose(fpt.gI_ct(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_gI_ct_benchmark.txt'))

def test_gI_ta(fpt):
    bmark = np.transpose(fpt.gI_ta(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_gI_ta_benchmark.txt'))

def test_gI_tt(fpt):
    bmark = np.transpose(fpt.gI_tt(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_gI_tt_benchmark.txt'))

def test_OV(fpt):
    bmark = np.transpose(fpt.OV(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_OV_benchmark.txt'))

def test_kPol(fpt):
    bmark = np.transpose(fpt.kPol(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_kPol_benchmark.txt'))

def test_RSD_components(fpt):
    bmark = np.transpose(fpt.RSD_components(P, 1.0, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_RSD_benchmark.txt'))

def test_RSD_ABsum_components(fpt):
    bmark = np.transpose(fpt.RSD_ABsum_components(P, 1.0, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_RSD_ABsum_components_benchmark.txt'))

def test_RSD_ABsum_mu(fpt):
    bmark = np.transpose(fpt.RSD_ABsum_mu(P, 1.0, 1.0, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_RSD_ABsum_mu_benchmark.txt'))

@pytest.mark.skipif(
        sys.version_info >= (3, 13) or platform.machine() == "arm64",
        reason="Strict benchmark comparison is not reliable on Python 3.13+ or ARM64 runners"
    )
def test_IRres(fpt):
    bmark = fpt.IRres(P, C_window=C_window)
    stored = np.transpose(np.loadtxt('tests/benchmarking/P_IRres_benchmark.txt'))
    # calc_and_show(bmark, stored, "IRres")
    if np.__version__ >= '2.0':
        warnings.warn("The benchmarks were generated with NumPy 1.x, the IRres term is known to fail np.allclose when using NumPy 2.x." +
                      " We can guarantee a precision of 5e-5 up until a k value of 10.")
    assert np.allclose(bmark, stored)
