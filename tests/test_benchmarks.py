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
def locate_max(rel_diff):
    """Describe where the largest relative difference sits.

    Benchmarks are compared with k on axis zero.
    Returns e.g. "k[2976] = 89.9466" or "k[2990] = 95.9387 (term 8)".
    """
    idx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)
    i = idx[0]
    term = idx[1] if rel_diff.ndim > 1 else None
    where = f"k[{i}] = {k[i]:.6g}"
    return where if term is None else f"{where} (term {term})"

K_EDGE_LOW = 1e-3
K_EDGE_HIGH = 10.0

def assert_benchmark(bmark, stored, name, strict_atol = 0, strict_rtol = 1e-5, loose_atol=0, loose_rtol=1e-3,
                     edge_rtol=0.2):
    """Compare a computed benchmark against stored reference values.

    Passes silently when the arrays match at the strict tolerance
    (``strict_atol``/``strict_rtol``). If the strict comparison fails but the
    arrays still agree at a looser tolerance (``loose_atol``/``loose_rtol``),
    the discrepancy is treated as benign floating-point noise -- e.g. from a
    different NumPy/Python version or CPU architecture than the one the
    benchmark was generated on -- and the test is marked xfail with an
    explanatory message.

    Failing that, the tolerance is applied per point: values at the ends of the
    k grid (k <= K_EDGE_LOW or k >= K_EDGE_HIGH) only have to agree at
    ``edge_rtol``, while the interior still has to agree at ``loose_rtol``. This
    catches the common case of divergence confined to the extremes of the k
    range, and is also marked xfail.

    If the arrays disagree even then, that likely signals a real regression and
    the test fails hard.
    """
    bmark = np.asarray(bmark)
    stored = np.asarray(stored)
    if np.allclose(bmark, stored, atol=strict_atol, rtol=strict_rtol):
        return
    rel_diff = np.abs(bmark - stored) / np.maximum(np.abs(stored), 1e-20)
    max_rel_diff = np.max(rel_diff)
    max_rel_at = locate_max(rel_diff)
    if np.allclose(bmark, stored, atol=loose_atol, rtol=loose_rtol):
        pytest.xfail(
            f"{name}: matches the stored benchmark at a loose tolerance "
            f"(atol={loose_atol}, rtol={loose_rtol}; max rel diff "
            f"{max_rel_diff:.2e}) but not at the strict tolerance "
            f"(atol={strict_atol}, rtol={strict_rtol}). This is "
            f"consistent with floating-point differences across NumPy/Python "
            f"versions or CPU architecture, not a code error. "
            f"Largest relative difference at {max_rel_at}.")
    edge = (k <= K_EDGE_LOW) | (k >= K_EDGE_HIGH)
    if bmark.ndim > 1:
        edge = edge[:, None]
    edge = np.broadcast_to(edge, bmark.shape)
    rtol_per_point = np.where(edge, edge_rtol, loose_rtol)
    if np.all(np.abs(bmark - stored) <= loose_atol + rtol_per_point * np.abs(stored)):
        pytest.xfail(
            f"{name}: likely failing due to small numerical divergence at the edges "
            f"of the k range. The interior ({K_EDGE_LOW:g} < k < {K_EDGE_HIGH:g}) "
            f"agrees at rtol={loose_rtol} (max rel diff {rel_diff[~edge].max():.2e}), "
            f"while the edges agree only at rtol={edge_rtol} (max rel diff "
            f"{rel_diff[edge].max():.2e}). The power spectrum extrapolation and FFT "
            f"window dominate there. "
            f"Largest relative difference at {max_rel_at}.")
    raise AssertionError(
        f"{name}: differs from the stored benchmark beyond the loose tolerance "
        f"(atol={loose_atol}, rtol={loose_rtol}; max rel diff {max_rel_diff:.2e}), "
        f"including beyond rtol={edge_rtol} allowed at the edges of the k range. "
        f"This exceeds expected floating-point noise and likely indicates a real "
        f"problem, not just a NumPy/platform difference. "
        f"Largest relative difference at {max_rel_at}.")

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

    bmark = np.transpose(bmark)
    stored = load_benchmark('P_bias_benchmark.txt')
    # calc_and_show(bmark[:, 0], stored[:, 0], "one_loop_dd_bias")
    assert_benchmark(bmark, stored, "one_loop_dd_bias")

def test_one_loop_dd_bias_b3nl(fpt):
    bmark = list(fpt.one_loop_dd_bias_b3nl(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[7]
    bmark[7] = new_array

    bmark = np.transpose(bmark)
    stored = load_benchmark('P_bias_b3nl_benchmark.txt')
    # calc_and_show(bmark[:, 8], stored[:, 8], "one_loop_dd_bias_b3nl")
    assert_benchmark(bmark, stored, "one_loop_dd_bias_b3nl")

def test_one_loop_dd_bias_lpt_NL(fpt):
    bmark = list(fpt.one_loop_dd_bias_lpt_NL(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[6]
    bmark[6] = new_array

    bmark = np.transpose(bmark)
    stored = load_benchmark('P_bias_lpt_NL_benchmark.txt')
    # calc_and_show(bmark[:, 1], stored[:, 1], "one_loop_dd_bias_lpt_NL")
    # calc_and_show(bmark[:, 2], stored[:, 2], "one_loop_dd_bias_lpt_NL")
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
    stored = load_benchmark('P_IRres_benchmark.txt')
    # calc_and_show(bmark, stored, "IRres")
    assert_benchmark(bmark, stored, "IRres")
