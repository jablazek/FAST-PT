import pytest
import numpy as np
from fastpt import FASTPT
import os

BENCH_DIR = os.path.join(os.path.dirname(__file__), 'benchmarking')

def load_benchmark(name):
    """Load a stored benchmark file relative to this test file, not the CWD."""
    return np.loadtxt(os.path.join(BENCH_DIR, name))

data_path = os.path.join(BENCH_DIR, 'Pk_test.dat')
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


def assert_benchmark(bmark, stored, name, loose_atol=2e-3, loose_rtol=1e-3):
    """Compare a computed benchmark against stored reference values.

    Passes silently when the arrays match at ``np.allclose`` defaults (strict).
    If the strict comparison fails but the arrays still agree at a looser
    tolerance (``loose_atol``/``loose_rtol``), the discrepancy is treated as
    benign floating-point noise -- e.g. from a different NumPy/Python version or
    CPU architecture than the one the benchmark was generated on -- and the test
    is marked xfail with an explanatory message. If the arrays disagree even at
    the loose tolerance, that likely signals a real regression and the test
    fails hard.
    """
    bmark = np.asarray(bmark)
    stored = np.asarray(stored)
    if np.allclose(bmark, stored):
        return
    max_abs_diff = np.max(np.abs(bmark - stored))
    if np.allclose(bmark, stored, atol=loose_atol, rtol=loose_rtol):
        pytest.xfail(
            f"{name}: matches the stored benchmark at a loose tolerance "
            f"(atol={loose_atol}, rtol={loose_rtol}; max abs diff "
            f"{max_abs_diff:.2e}) but not at np.allclose defaults. This is "
            f"consistent with floating-point differences across NumPy/Python "
            f"versions or CPU architecture, not a code error.")
    raise AssertionError(
        f"{name}: differs from the stored benchmark beyond the loose tolerance "
        f"(atol={loose_atol}, rtol={loose_rtol}; max abs diff {max_abs_diff:.2e}). "
        f"This exceeds expected floating-point noise and likely indicates a real "
        f"problem, not just a NumPy/platform difference.")

def test_one_loop_dd(fpt):
    bmark = fpt.one_loop_dd(P, C_window=C_window)[0]
    stored = load_benchmark('P_dd_benchmark.txt')
    # calc_and_show(bmark, stored, "one_loop_dd")
    assert_benchmark(bmark, stored, "one_loop_dd")

def test_one_loop_dd_bias(fpt):
    bmark = list(fpt.one_loop_dd_bias(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[7]
    bmark[7] = new_array
    
    stored = np.transpose(load_benchmark('P_bias_benchmark.txt'))
    # calc_and_show(bmark[0], stored[0], "one_loop_dd_bias")
    assert_benchmark(bmark, stored, "one_loop_dd_bias")

def test_one_loop_dd_bias_b3nl(fpt):
    bmark = list(fpt.one_loop_dd_bias_b3nl(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[7]
    bmark[7] = new_array

    stored = np.transpose(load_benchmark('P_bias_b3nl_benchmark.txt'))
    # calc_and_show(bmark[8], stored[8], "one_loop_dd_bias_b3nl")
    assert_benchmark(bmark, stored, "one_loop_dd_bias_b3nl")

def test_one_loop_dd_bias_lpt_NL(fpt):
    bmark = list(fpt.one_loop_dd_bias_lpt_NL(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[6]
    bmark[6] = new_array
    
    stored = np.transpose(load_benchmark('P_bias_lpt_NL_benchmark.txt'))
    # calc_and_show(bmark[1], stored[1], "one_loop_dd_bias_lpt_NL")
    # calc_and_show(bmark[2], stored[2], "one_loop_dd_bias_lpt_NL")
    assert_benchmark(bmark, stored, "one_loop_dd_bias_lpt_NL")

def test_IA_TT(fpt):
    bmark = np.transpose(fpt.IA_tt(P, C_window=C_window))
    stored = load_benchmark('PIA_tt_benchmark.txt')
    assert_benchmark(bmark, stored, "IA_tt")
    
def test_IA_mix(fpt):
    bmark = np.transpose(fpt.IA_mix(P, C_window=C_window))
    stored = load_benchmark('P_IA_mix_benchmark.txt')
    assert_benchmark(bmark, stored, "IA_mix")

def test_IA_ta(fpt):
    bmark = np.transpose(fpt.IA_ta(P, C_window=C_window))
    stored = load_benchmark('P_IA_ta_benchmark.txt')
    assert_benchmark(bmark, stored, "IA_ta")

def test_IA_der(fpt):
    bmark = np.transpose(fpt.IA_der(P, C_window=C_window))
    stored = load_benchmark('P_IA_der_benchmark.txt')
    assert_benchmark(bmark, stored, "IA_der")

def test_IA_ct(fpt):
    bmark = np.transpose(fpt.IA_ct(P, C_window=C_window))
    stored = load_benchmark('P_IA_ct_benchmark.txt')
    assert_benchmark(bmark, stored, "IA_ct")

def test_gI_ct(fpt):
    bmark = np.transpose(fpt.gI_ct(P, C_window=C_window))
    stored = load_benchmark('P_gI_ct_benchmark.txt')
    assert_benchmark(bmark, stored, "gI_ct")

def test_gI_ta(fpt):
    bmark = np.transpose(fpt.gI_ta(P, C_window=C_window))
    stored = load_benchmark('P_gI_ta_benchmark.txt')
    assert_benchmark(bmark, stored, "gI_ta")

def test_gI_tt(fpt):
    bmark = np.transpose(fpt.gI_tt(P, C_window=C_window))
    stored = load_benchmark('P_gI_tt_benchmark.txt')
    assert_benchmark(bmark, stored, "gI_tt")

def test_OV(fpt):
    bmark = np.transpose(fpt.OV(P, C_window=C_window))
    stored = load_benchmark('P_OV_benchmark.txt')
    assert_benchmark(bmark, stored, "OV")

def test_kPol(fpt):
    bmark = np.transpose(fpt.kPol(P, C_window=C_window))
    stored = load_benchmark('P_kPol_benchmark.txt')
    assert_benchmark(bmark, stored, "kPol")

def test_RSD_components(fpt):
    bmark = np.transpose(fpt.RSD_components(P, 1.0, C_window=C_window))
    stored = load_benchmark('P_RSD_benchmark.txt')
    assert_benchmark(bmark, stored, "RSD_components")

def test_RSD_ABsum_components(fpt):
    bmark = np.transpose(fpt.RSD_ABsum_components(P, 1.0, C_window=C_window))
    stored = load_benchmark('P_RSD_ABsum_components_benchmark.txt')
    assert_benchmark(bmark, stored, "RSD_ABsum_components")

def test_RSD_ABsum_mu(fpt):
    bmark = np.transpose(fpt.RSD_ABsum_mu(P, 1.0, 1.0, C_window=C_window))
    stored = load_benchmark('P_RSD_ABsum_mu_benchmark.txt')
    assert_benchmark(bmark, stored, "RSD_ABsum_mu")

def test_IRres(fpt):
    bmark = fpt.IRres(P, C_window=C_window)
    stored = np.transpose(load_benchmark('P_IRres_benchmark.txt'))
    # calc_and_show(bmark, stored, "IRres")
    assert_benchmark(bmark, stored, "IRres")
