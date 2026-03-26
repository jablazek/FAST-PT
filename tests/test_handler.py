import pytest
import numpy as np
from fastpt import FASTPT
from fastpt import FPTHandler
import os
import tempfile

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
P = np.loadtxt(data_path)[:, 1]
k = np.loadtxt(data_path)[:, 0]
C_window = 0.75
P_window = np.array([0.2, 0.2])
    

if __name__ == "__main__":
    fast = FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5*len(k)))
    handler = FPTHandler(fast, P=P, P_window=P_window, C_window=C_window)
    # handler.save_instance("this_instance")




@pytest.fixture
def fpt():
    k = np.loadtxt(data_path)[:, 0]
    n_pad=int(0.5*len(k))
    return FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad)

@pytest.fixture
def handler(fpt):
    return FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)

################# FUNCTIONALITY TESTS #################
def test_init_with_valid_params(handler):
    assert isinstance(handler.fastpt, FASTPT)
    assert handler.default_params['P'] is not None
    assert handler.default_params['P_window'].all() == P_window.all()
    assert handler.default_params['C_window'] == 0.75

def test_run_without_power_spectrum(fpt):
    handler = FPTHandler(fpt) #P is not required at handler init but must be passed at every function call
    with pytest.raises(ValueError, match="Missing required parameters for 'one_loop_dd': \\['P'\\]. Please recall with the missing parameters."):
        handler.run('one_loop_dd')

def test_init_with_zero_power_spectrum(fpt):
    P = np.zeros_like(fpt.k_original)
    with pytest.raises(ValueError, match='Your input power spectrum array is all zeros'):
        FPTHandler(fpt, P=P)

def test_init_with_mismatched_arrays(fpt):
    P = np.ones(10)  # Wrong size
    with pytest.raises(ValueError, match='Input k and P arrays must have the same size'):
        FPTHandler(fpt, P=P)

def test_invalid_c_window(fpt):
    with pytest.raises(ValueError, match='C_window must be between 0 and 1'):
        FPTHandler(fpt, P=P, C_window=1.5)
    with pytest.raises(ValueError, match='C_window must be between 0 and 1'):
        FPTHandler(fpt, P=P, C_window=-0.5)

def test_invalid_p_window(fpt):
    with pytest.raises(ValueError, match='P_window must be a tuple of two values'):
        FPTHandler(fpt, P=P, P_window=np.array([1.0]))

def test_invalid_function_call(handler):
    with pytest.raises(ValueError, match="Function 'nonexistent_function' not found"):
        handler.run('nonexistent_function')

def test_missing_required_params(fpt):
    handler = FPTHandler(fpt, P=P)
    with pytest.raises(ValueError, match="Missing required parameters"):
        handler.run('RSD_components')

def test_all_fastpt_functions_with_handler_params(fpt):
    """Test FASTPT functions with parameters set during handler initialization"""
    # Initialize handler with all possible parameters
    default_params = {
        'P': P,
        'P_window': P_window,
        'C_window': C_window,
        'f': 0.5,
        'mu_n': 0.5,
        'nu': -2,
        'L': 0.2,
        'h': 0.67,
        'rsdrag': 135
    }
    handler = FPTHandler(fpt, **default_params)
    
    # Dictionary mapping functions to any additional required parameters
    func_names = (
        'one_loop_dd', 'one_loop_dd_bias', 'one_loop_dd_bias_b3nl',
        'one_loop_dd_bias_lpt_NL', 'IA_tt', 'IA_mix', 'IA_ta',
        'IA_der', 'IA_ct', 'gI_ct', 'gI_ta',
        'gI_tt', 'OV', 'kPol', 'RSD_components', 'IRres',
        'RSD_ABsum_components', 'RSD_ABsum_mu',
    )
    
    for name in func_names:
        try:
            result = handler.run(name)
            assert result is not None, f"Function {name} returned None"
            if isinstance(result, tuple):
                assert all(r is not None for r in result), f"Function {name} returned None in tuple"
        except Exception as e:
            pytest.fail(f"Function {name} failed to run with error: {str(e)}")

def test_all_fastpt_functions_with_run_params(fpt):
    """Test FASTPT functions with parameters passed during run call"""
    handler = FPTHandler(fpt)
    
    # Dictionary mapping functions to their required run-time parameters
    function_params = {
        'one_loop_dd': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'one_loop_dd_bias': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'one_loop_dd_bias_b3nl': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'one_loop_dd_bias_lpt_NL': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_tt': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_mix': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_ta': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_der': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_ct': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'gI_ct': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'gI_ta': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'gI_tt': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'OV': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'kPol': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'RSD_components': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5},
        'RSD_ABsum_components': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5},
        'RSD_ABsum_mu': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5, 'mu_n': 0.5}
    }
    
    for func_name, params in function_params.items():
        try:
            result = handler.run(func_name, **params)
            assert result is not None, f"Function {func_name} returned None"
            if isinstance(result, tuple):
                assert all(r is not None for r in result), f"Function {func_name} returned None in tuple"
        except Exception as e:
            pytest.fail(f"Function {func_name} failed to run with error: {str(e)}")

def test_clear_params(handler):
    handler.clear_default_params()
    assert handler.default_params == {}

def test_override_params(handler):
    original_params = handler.default_params.copy()
    r1 = handler.run('one_loop_dd')
    new_params = {'P': P * 2, 'P_window': np.array([0.1, 0.1]), 'C_window': 0.5}
    # Run with overridden parameters (but this doesn't update default_params)
    r2 = handler.run('one_loop_dd', **new_params)
    assert not np.array_equal(r1, r2)
    # Assert that default_params weren't changed
    for key in original_params:
        if isinstance(original_params[key], np.ndarray):
            assert np.array_equal(handler.default_params[key], original_params[key])
        else:
            assert handler.default_params[key] == original_params[key]

def test_update_params(handler):
    new_params = {'P': P * 2, 'P_window': np.array([0.1, 0.1]), 'C_window': 0.5}
    handler.update_default_params(**new_params)
    assert handler.default_params == new_params

def test_update_fpt_instance(fpt, handler):
    handler.run('one_loop_dd')  # Run to initialize cache
    new_fpt = FASTPT(fpt.k_original)
    handler.update_fastpt_instance(new_fpt)
    assert handler.fastpt == new_fpt

################# result_direct TESTS #################
def test_handler_function_equality(fpt):
    """Test that handler produces identical results to direct FASTPT function calls"""
    handler = FPTHandler(fpt)
    
    function_params = {
        'one_loop_dd': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'one_loop_dd_bias': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'one_loop_dd_bias_b3nl': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'one_loop_dd_bias_lpt_NL': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_tt': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_mix': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_ta': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_der': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_ct': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'gI_ct': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'gI_ta': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'gI_tt': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'OV': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'kPol': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'RSD_components': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5},
        'RSD_ABsum_components': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5},
        'RSD_ABsum_mu': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5, 'mu_n': 0.5}
    }
    
    for func_name, params in function_params.items():
        try:
            # Get direct FASTPT function
            fastpt_func = getattr(fpt, func_name)
            
            # Run both ways
            direct_result = fastpt_func(**params)
            handler_result = handler.run(func_name, **params)
            
            # Compare results
            if isinstance(direct_result, (tuple, list)):
                assert isinstance(handler_result, (tuple, list))
                assert len(direct_result) == len(handler_result)
                for dr, hr in zip(direct_result, handler_result):
                    if isinstance(dr, np.ndarray):
                        assert np.array_equal(dr, hr)
                    else:
                        assert dr == hr
            elif isinstance(direct_result, np.ndarray):
                assert np.array_equal(direct_result, handler_result)
            else:
                assert direct_result == handler_result
                
        except Exception as e:
            pytest.fail(f"Function {func_name} comparison failed with error: {str(e)}")


########### GET METHOD TESTING ############
def test_get_method_basics(fpt, handler):
        """Test the basic functionality of the get method"""
        
        # Test single term retrieval
        p_deltaE1 = handler.get("P_deltaE1")
        p_deltaE1_direct = fpt.IA_ta(P=P, P_window=P_window, C_window=C_window)[0]
        assert np.array_equal(p_deltaE1, p_deltaE1_direct)
        
        # Test multiple terms retrieval
        terms = handler.get("P_deltaE1", "P_0E0E")
        assert "P_deltaE1" in terms
        assert "P_0E0E" in terms
        assert np.array_equal(terms["P_deltaE1"], p_deltaE1_direct)
        assert np.array_equal(terms["P_0E0E"], fpt.IA_ta(P=P, P_window=P_window, C_window=C_window)[2])

@pytest.mark.parametrize("term_name", ["P_1loop", "Ps", 
                                       "Pd1d2", "Pd2d2", "Pd1s2", "Pd2s2", "Ps2s2", "sig4",
                                       "sig3nl",
                                       "Pb1L", "Pb1L_2", "Pb1L_b2L", "Pb2L", "Pb2L_2",
                                       "P_E", "P_B",
                                       "P_A", "P_Btype2", "P_DEE", "P_DBB",
                                       "P_deltaE1", "P_deltaE2", "P_0E0E", "P_0B0B",
                                       "P_der",
                                       "P_0tE", "P_0EtE", "P_E2tE", "P_tEtE",
                                       "P_d2tE", "P_s2tE",
                                       "P_s2E2", "P_d2E2",
                                       "P_d2E", "P_d20E", "P_s2E", "P_s20E",
                                       "P_OV",
                                       "P_kP1", "P_kP2", "P_kP3"])                   
def test_get_all_terms(fpt, handler, term_name):
    term_sources = {
            "P_1loop": ("one_loop_dd", 0),
            "Ps": ("one_loop_dd", 1),
            "Pd1d2": ("one_loop_dd_bias", 2),  
            "Pd2d2": ("one_loop_dd_bias", 3),
            "Pd1s2": ("one_loop_dd_bias", 4),
            "Pd2s2": ("one_loop_dd_bias", 5),
            "Ps2s2": ("one_loop_dd_bias", 6),
            "sig4": ("one_loop_dd_bias", 7),
        
            "sig3nl": ("one_loop_dd_bias_b3nl", 8),
        
            "Pb1L": ("one_loop_dd_bias_lpt_NL", 1),
            "Pb1L_2": ("one_loop_dd_bias_lpt_NL", 2),
            "Pb1L_b2L": ("one_loop_dd_bias_lpt_NL", 3),
            "Pb2L": ("one_loop_dd_bias_lpt_NL", 4),
            "Pb2L_2": ("one_loop_dd_bias_lpt_NL", 5),
        
            "P_E": ("IA_tt", 0),
            "P_B": ("IA_tt", 1),
        
            "P_A": ("IA_mix", 0),
            "P_Btype2": ("IA_mix", 1),
            "P_DEE": ("IA_mix", 2),
            "P_DBB": ("IA_mix", 3),
        
            "P_deltaE1": ("IA_ta", 0),
            "P_deltaE2": ("IA_ta", 1),
            "P_0E0E": ("IA_ta", 2),
            "P_0B0B": ("IA_ta", 3),

            "P_der": ("IA_der", 0),

            "P_0tE": ("IA_ct", 0),
            "P_0EtE": ("IA_ct", 1),
            "P_E2tE": ("IA_ct", 2),
            "P_tEtE": ("IA_ct", 3),
        
            "P_d2tE": ("gI_ct", 0),
            "P_s2tE": ("gI_ct", 1),
        
            "P_s2E2": ("gI_tt", 0),
            "P_d2E2": ("gI_tt", 1),
        
            "P_d2E": ("gI_ta", 0),
            "P_d20E": ("gI_ta", 1),
            "P_s2E": ("gI_ta", 2),
            "P_s20E": ("gI_ta", 3),
        
            "P_OV": ("OV", 0),
        
            "P_kP1": ("kPol", 0),
            "P_kP2": ("kPol", 1),
            "P_kP3": ("kPol", 2),
        
            # "A1": ("RSD_components", 0),
            # "A3": ("RSD_components", 1),
            # "A5": ("RSD_components", 2),
            # "B0": ("RSD_components", 3),
            # "B2": ("RSD_components", 4),
            # "B4": ("RSD_components", 5),
            # "B6": ("RSD_components", 6),
            # "P_Ap1": ("RSD_components", 7),
            # "P_Ap3": ("RSD_components", 8),
            # "P_Ap5": ("RSD_components", 9),
        
            # "ABsum_mu2": ("RSD_ABsum_components", 0),
            # "ABsum_mu4": ("RSD_ABsum_components", 1),
            # "ABsum_mu6": ("RSD_ABsum_components", 2),
            # "ABsum_mu8": ("RSD_ABsum_components", 3),
        
            # "ABsum": ("RSD_ABsum_components", 0),

            # "P_IRres": ("IRres", 0),
        }
    term_source = term_sources[term_name]
    result = handler.get(term_name)
    result_direct = getattr(fpt, term_source[0])(P=P, P_window=P_window, C_window=C_window)
    if isinstance(result_direct, tuple):
        result_direct = result_direct[term_source[1]]
    assert np.allclose(result, result_direct)
    fpt.cache.clear()
    result_direct2 = getattr(fpt, term_source[0])(P=P, P_window=P_window, C_window=C_window)
    if isinstance(result_direct2, tuple):
        result_direct2 = result_direct2[term_source[1]]
    assert np.allclose(result, result_direct2)


def test_get_with_caching(fpt):
    """Test get method with caching enabled"""
    handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
        
    # First call - should compute
    result1 = handler.get("P_deltaE1")
    
    # Check cache statistics before second call
    initial_hits = fpt.cache.hits
    initial_misses = fpt.cache.misses
    
    # Second call - should use cache
    result2 = handler.get("P_deltaE1")
    
    # Verify cache was used (hits increased)
    assert fpt.cache.hits > initial_hits
    assert fpt.cache.misses == initial_misses
    assert np.array_equal(result1, result2)
    
    # Clear the cache
    fpt.cache.clear()
    
    # After clearing, the next call should recompute (miss)
    pre_miss_count = fpt.cache.misses
    handler.get("P_deltaE1")
    assert fpt.cache.misses > pre_miss_count
    
    # Check that multiple different calculations create different cache entries
    pre_cache_count = len(fpt.cache.cache)
    handler.get("P_0E0E")  # Different calculation
    assert len(fpt.cache.cache) > pre_cache_count

def test_get_with_different_params(fpt):
    """Test get method with different parameter combinations"""
    handler = FPTHandler(fpt)
        
    # Test with parameters provided at runtime
    result1 = handler.get("P_deltaE1", P=P, P_window=P_window, C_window=C_window)
    fpt.cache.clear()
    # Test with default parameters
    handler2 = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    result2 = handler2.get("P_deltaE1")
        
    assert np.array_equal(result1, result2)
        
    # Test with override parameters
    handler3 = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    new_P = P * 5
    result3 = handler3.get("P_deltaE1", P=new_P)
        
    assert not np.array_equal(result1, result3)

def test_get_invalid_term(handler):
    """Test get method with invalid term name"""
    with pytest.raises(ValueError, match="Term 'nonexistent_term' not found in FASTPT"):
        handler.get("nonexistent_term")

def test_get_missing_params(fpt):
        """Test get method with missing required parameters"""
        handler = FPTHandler(fpt)
        with pytest.raises(ValueError, match="Missing required parameters"):
            handler.get("P_deltaE1")  # P is required

def test_get_special_terms(fpt, handler):
    """Test get method with special terms that have their own functions"""
        
    p_btype2 = handler.get("P_Btype2")
    fpt.cache.clear()
    p_btype2_direct = fpt._get_P_Btype2(P)
        
    assert np.array_equal(p_btype2, p_btype2_direct)
        
    p_deltaE2 = handler.get("P_deltaE2")
    fpt.cache.clear()
    p_deltaE2_direct = fpt._get_P_deltaE2(P)
        
    assert np.array_equal(p_deltaE2, p_deltaE2_direct)
    
    p_ov = handler.get("P_OV")
    fpt.cache.clear()
    p_ov_direct = fpt.OV(P, P_window=P_window, C_window=C_window)
    assert np.array_equal(p_ov, p_ov_direct), f"p_ov: {p_ov}, p_ov_direct: {p_ov_direct}"

    p_der = handler.get("P_der")
    fpt.cache.clear()
    p_der_direct = fpt.IA_der(P)
    assert np.array_equal(p_der, p_der_direct)


def test_get_edge_cases(fpt):
    """Test edge cases for the get method"""
    # Test with empty parameters
    handler = FPTHandler(fpt)
    
    # Test with empty term list
    with pytest.raises(ValueError, match="At least one term must be provided."):
        handler.get()
    
    # Test with required parameters passed directly at call
    result = handler.get("P_E", P=P, P_window=P_window, C_window=C_window)
    assert result is not None
    
    # Test with mixed valid and invalid terms
    with pytest.raises(ValueError, match="not found in FASTPT"):
        handler.get("P_E", "nonexistent_term", P=P, P_window=P_window, C_window=C_window)
    
    # Test parameter validation - P length must match k length
    with pytest.raises(ValueError):
        handler.get("P_E", P=np.ones(10))
    
    # Test parameter validation - C_window must be between 0 and 1
    with pytest.raises(ValueError):
        handler.get("P_E", P=P, C_window=1.5)
    
    # Test with overridden default parameters
    handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    new_P = P * 1.1
    result1 = handler.get("P_E")
    result2 = handler.get("P_E", P=new_P)
    assert not np.array_equal(result1, result2)

################# BULK RUN TESTS #################
def test_bulk_run_basic(fpt):
    """Test basic functionality of bulk_run method"""
    handler = FPTHandler(fpt)
    funcs = ['one_loop_dd', 'IA_tt']
    power_spectra = [P, P * 1.1, P * 1.2]
    
    results = handler.bulk_run(funcs, power_spectra, P_window=P_window, C_window=C_window)
    
    # Check that all expected results are present
    assert len(results) == len(funcs) * len(power_spectra)
    
    for func in funcs:
        for i in range(len(power_spectra)):
            assert (func, i) in results
            assert results[(func, i)] is not None

def test_bulk_run_results_correctness(handler):
    """Test that bulk_run results match individual run calls"""
    funcs = ['one_loop_dd', 'IA_tt']
    power_spectra = [P, P * 1.5]
    
    bulk_results = handler.bulk_run(funcs, power_spectra)
    
    # Compare with individual runs
    for func in funcs:
        for i, spec in enumerate(power_spectra):
            individual_result = handler.run(func, P=spec)
            bulk_result = bulk_results[(func, i)]
            
            if isinstance(individual_result, tuple):
                assert isinstance(bulk_result, tuple)
                assert len(individual_result) == len(bulk_result)
                for ir, br in zip(individual_result, bulk_result):
                    assert np.array_equal(ir, br)
            else:
                assert np.array_equal(individual_result, bulk_result)

def test_bulk_run_with_overrides(handler):
    """Test bulk_run with additional override parameters"""
    funcs = ['RSD_components']
    power_spectra = [P]
    
    # RSD_components requires 'f' parameter
    results = handler.bulk_run(funcs, power_spectra, 
                               P_window=P_window, C_window=C_window, f=0.5)
    
    assert (funcs[0], 0) in results
    assert results[(funcs[0], 0)] is not None

def test_bulk_run_empty_inputs(handler):
    """Test bulk_run with empty function list or power spectra list"""    
    # Empty function list
    empty_results = handler.bulk_run([], [P])
    assert len(empty_results) == 0
    
    # Empty power spectra list
    empty_results = handler.bulk_run(['one_loop_dd'], [])
    assert len(empty_results) == 0

def test_bulk_run_with_invalid_function(handler):
    """Test bulk_run with invalid function name"""    
    with pytest.raises(ValueError, match="Function 'invalid_function' not found in FASTPT"):
        handler.bulk_run(['invalid_function'], [P])

def test_bulk_run_missing_params(handler):
    """Test bulk_run with missing required parameters"""    
    with pytest.raises(ValueError, match="Missing required parameters"):
        handler.bulk_run(['RSD_components'], [P], P_window=P_window, C_window=C_window)
        # Missing 'f' parameter for RSD_components

def test_bulk_run_large_input(handler):
    """Test bulk_run with larger number of functions and spectra"""
    funcs = ['one_loop_dd', 'IA_tt', 'IA_mix', 'OV', 'kPol']
    power_spectra = [P, P * 1.1, P * 1.2, P * 1.3, P * 1.4]
    
    results = handler.bulk_run(funcs, power_spectra, P_window=P_window, C_window=C_window)
    
    # Check we got all expected combinations
    assert len(results) == len(funcs) * len(power_spectra)
    for func in funcs:
        for i in range(len(power_spectra)):
            assert (func, i) in results

def test_bulk_run_with_save_all(fpt):
    """Test bulk_run with save_all flag"""
    # Create a temporary outputs directory to avoid cluttering
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock save_output that just records calls rather than saving files
        saved_outputs = []
        
        original_save_output = FPTHandler.save_output
        try:
            def mock_save_output(self, result, func_name, type="txt", output_dir=None):
                saved_outputs.append((result, func_name, type, output_dir))
                
            FPTHandler.save_output = mock_save_output
            
            handler = FPTHandler(fpt, save_all="txt", P_window=P_window, C_window=C_window, save_dir=temp_dir)
            funcs = ['one_loop_dd', 'IA_tt']
            power_spectra = [P, P * 1.1]
            
            handler.bulk_run(funcs, power_spectra)
            
            # Check that save_output was called for each function and power spectrum
            assert len(saved_outputs) == len(funcs) * len(power_spectra)
            
            # Verify the correct arguments were passed to save_output
            for output in saved_outputs:
                assert isinstance(output[0], tuple)  # Result
                assert output[1] in funcs  # Function name
                assert output[2] == "txt"  # File type
                assert output[3] == temp_dir  # Output directory
        finally:
            # Restore original method
            FPTHandler.save_output = original_save_output

################# OUTPUT FILE TESTS #################
def test_save_output_file_types(handler):
    """Test saving output in different file formats"""
    import json
    import csv
    
    with tempfile.TemporaryDirectory() as temp_dir:        
        # Test txt format
        txt_result = handler.run('one_loop_dd', save_type="txt", save_dir=temp_dir)
        txt_file = os.path.join(temp_dir, "one_loop_dd_output.txt")
        assert os.path.exists(txt_file)
        loaded_txt = np.loadtxt(txt_file)
        # Check that the saved data matches the result (accounting for transpose in saving)
        assert np.allclose(loaded_txt, np.transpose(txt_result))
        
        # Test csv format
        csv_result = handler.run('IA_tt', save_type="csv", save_dir=temp_dir)
        csv_file = os.path.join(temp_dir, "IA_tt_output.csv")
        assert os.path.exists(csv_file)
        # Read CSV file and check header and data
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            assert header == ['IA_tt_0', 'IA_tt_1']  # Header should contain function name with indices
            data = list(csv_reader)
            assert len(data) > 0
        
        # Test json format
        json_result = handler.run('one_loop_dd_bias', save_type="json", save_dir=temp_dir)
        json_file = os.path.join(temp_dir, "one_loop_dd_bias_output.json")
        assert os.path.exists(json_file)
        # Read JSON file and check structure
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            assert 'one_loop_dd_bias' in json_data
            # Check that the saved data has the same structure (number of arrays)
            if isinstance(json_result, tuple):
                assert len(json_data['one_loop_dd_bias']) == len(json_result)

def test_save_output_auto_naming(handler):
    """Test automatic filename adjustment when files already exist"""
    
    with tempfile.TemporaryDirectory() as temp_dir:        
        # First run - should create base filename
        handler.run('one_loop_dd', save_type="txt", save_dir=temp_dir)
        base_file = os.path.join(temp_dir, "one_loop_dd_output.txt")
        assert os.path.exists(base_file)
        
        # Second run - should create filename with counter
        handler.run('one_loop_dd', save_type="txt", save_dir=temp_dir)
        counter_file = os.path.join(temp_dir, "one_loop_dd_1_output.txt")
        assert os.path.exists(counter_file)
        
        # Third run - should increment counter
        handler.run('one_loop_dd', save_type="txt", save_dir=temp_dir)
        counter_file2 = os.path.join(temp_dir, "one_loop_dd_2_output.txt")
        assert os.path.exists(counter_file2)
        
        # Different function should still use base filename
        handler.run('IA_tt', save_type="txt", save_dir=temp_dir)
        different_file = os.path.join(temp_dir, "IA_tt_output.txt")
        assert os.path.exists(different_file)

def test_output_directory_precedence(fpt):
    """Test precedence of different output directory parameters"""
    
    with tempfile.TemporaryDirectory() as tempdir1:
        with tempfile.TemporaryDirectory() as tempdir2:
            # Initialize with default output directory
            handler = FPTHandler(fpt, save_dir=tempdir1, P=P, P_window=P_window, C_window=C_window)
                
            # 1. Test initializer-specified directory
            handler.run('one_loop_dd', save_type="txt")
            assert os.path.exists(os.path.join(tempdir1, "one_loop_dd_output.txt"))
                
            # 2. Test save_dir parameter overrides initializer
            handler.run('IA_tt', save_type="txt", save_dir=tempdir2)
            assert os.path.exists(os.path.join(tempdir2, "IA_tt_output.txt"))
            assert not os.path.exists(os.path.join(tempdir1, "IA_tt_output.txt"))

def test_invalid_file_type(handler):
    """Test that invalid file types raise appropriate errors"""
    
    with tempfile.TemporaryDirectory() as temp_dir:        
        with pytest.raises(ValueError, match="Invalid file type"):
            handler.run('one_loop_dd', save_type="invalid", save_dir=temp_dir)

def test_save_output_direct_call(handler):
    """Test direct calls to save_output method"""
    
    with tempfile.TemporaryDirectory() as temp_dir:        
        # Run function to get a result
        result = handler.run('one_loop_dd')
        
        # Call save_output directly with different file types
        handler.save_output(result, 'one_loop_dd', type="txt", output_dir=temp_dir)
        handler.save_output(result, 'one_loop_dd', type="csv", output_dir=temp_dir)
        handler.save_output(result, 'one_loop_dd', type="json", output_dir=temp_dir)
        
        # Check that all three files exist
        assert os.path.exists(os.path.join(temp_dir, "one_loop_dd_output.txt"))
        assert os.path.exists(os.path.join(temp_dir, "one_loop_dd_output.csv"))
        assert os.path.exists(os.path.join(temp_dir, "one_loop_dd_output.json"))

def test_bulk_run_with_save_options(handler):
    """Test bulk_run with file saving options"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        funcs = ['one_loop_dd', 'IA_tt']
        power_spectra = [P, P * 1.1]
        
        # Test bulk_run with save_type
        handler.bulk_run(funcs, power_spectra, save_type="txt", save_dir=temp_dir)
        
        # Check that files were created for each function and power spectrum
        for func in funcs:
            for i in range(len(power_spectra)):
                if i == 0:
                    base_path = os.path.join(temp_dir, f"{func}_output.txt")
                else:
                    base_path = os.path.join(temp_dir, f"{func}_{i}_output.txt")
                assert os.path.exists(base_path)

def test_load_method(handler):
    """Test the load method for loading saved output files of different types"""
    
    with tempfile.TemporaryDirectory() as temp_dir:        
        # Test txt format
        txt_result = handler.run('one_loop_dd', save_type="txt", save_dir=temp_dir)
        txt_loaded = handler.load('one_loop_dd_output.txt', load_dir=temp_dir)
        assert isinstance(txt_loaded, tuple)
        assert len(txt_loaded) == len(txt_result)
        for r1, r2 in zip(txt_result, txt_loaded):
            assert np.allclose(r1, r2)
        
        # Test csv format
        csv_result = handler.run('IA_tt', save_type="csv", save_dir=temp_dir)
        csv_loaded = handler.load('IA_tt_output.csv', load_dir=temp_dir)
        assert isinstance(csv_loaded, tuple)
        assert len(csv_loaded) == len(csv_result)
        for r1, r2 in zip(csv_result, csv_loaded):
            assert np.allclose(r1, r2)
        
        # Test json format
        json_result = handler.run('one_loop_dd_bias', save_type="json", save_dir=temp_dir)
        json_loaded = handler.load('one_loop_dd_bias_output.json', load_dir=temp_dir)
        assert isinstance(json_loaded, tuple)
        assert len(json_loaded) == len(json_result)
        for r1, r2 in zip(json_result, json_loaded):
            assert np.allclose(r1, r2), f"r1: {r1}, r2: {r2}"

def test_load_with_absolute_path(handler):
    """Test loading files using absolute paths"""
    
    with tempfile.TemporaryDirectory() as temp_dir:        
        # Save a result
        result = handler.run('one_loop_dd', save_type="txt", save_dir=temp_dir)
        file_path = os.path.join(temp_dir, "one_loop_dd_output.txt")
        
        # Load using absolute path - should ignore load_dir
        loaded_result = handler.load(file_path)
        assert isinstance(loaded_result, tuple)
        for r1, r2 in zip(result, loaded_result):
            assert np.allclose(r1, r2)
        
        # Load using absolute path but with a different load_dir (should ignore load_dir)
        with tempfile.TemporaryDirectory() as another_dir:
            loaded_result2 = handler.load(file_path, load_dir=another_dir)
            assert isinstance(loaded_result2, tuple)
            for r1, r2 in zip(result, loaded_result2):
                assert np.allclose(r1, r2)

def test_load_with_default_dir(fpt):
    """Test loading from the default output directory""" 
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize handler with custom output directory
        handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window, save_dir=temp_dir)
        
        # Save a result to the default directory
        result = handler.run('one_loop_dd', save_type="txt")
        
        # Load without specifying a directory (should use default)
        loaded_result = handler.load('one_loop_dd_output.txt')
        assert isinstance(loaded_result, tuple)
        for r1, r2 in zip(result, loaded_result):
            assert np.allclose(r1, r2)

def test_load_error_handling(handler):
    """Test error handling in the load method"""
    
    with tempfile.TemporaryDirectory() as temp_dir:        
        # Test file not found
        with pytest.raises(FileNotFoundError):
            handler.load('nonexistent_file.txt', load_dir=temp_dir)
        
        # Test unsupported file type
        handler.run('one_loop_dd', save_type="txt", save_dir=temp_dir)
        # Create a file with unsupported extension by renaming
        
        import shutil
        src = os.path.join(temp_dir, "one_loop_dd_output.txt")
        dst = os.path.join(temp_dir, "one_loop_dd_output.xyz")
        shutil.copy(src, dst)
        
        with pytest.raises(FileNotFoundError):
            handler.load('one_loop_dd_output.xyz', load_dir=temp_dir)

def test_load_consistency_across_types(handler):
    """Test that loading from different file types produces consistent results"""  
    
    with tempfile.TemporaryDirectory() as temp_dir:        
        # Save the same result in different formats
        result = handler.run('one_loop_dd')
        handler.save_output(result, 'one_loop_dd', type="txt", output_dir=temp_dir)
        handler.save_output(result, 'one_loop_dd', type="csv", output_dir=temp_dir)
        handler.save_output(result, 'one_loop_dd', type="json", output_dir=temp_dir)
        
        # Load from each format
        txt_loaded = handler.load('one_loop_dd_output.txt', load_dir=temp_dir)
        csv_loaded = handler.load('one_loop_dd_output.csv', load_dir=temp_dir)
        json_loaded = handler.load('one_loop_dd_output.json', load_dir=temp_dir)
        
        # Check that all loaded results match the original
        assert isinstance(txt_loaded, tuple) and isinstance(csv_loaded, tuple) and isinstance(json_loaded, tuple)
        assert len(txt_loaded) == len(csv_loaded) == len(json_loaded) == len(result)
        
        # Compare all formats against the original result
        for i in range(len(result)):
            assert np.allclose(result[i], txt_loaded[i])
            assert np.allclose(result[i], csv_loaded[i])
            assert np.allclose(result[i], json_loaded[i])

def test_load_with_relative_paths(handler):
    """Test loading using relative paths"""  
    # Get current working directory
    orig_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Change to the temporary directory
            os.chdir(temp_dir)
            # Create a subdirectory for outputs
            os.makedirs('outputs', exist_ok=True)                        
            # Save a result
            result = handler.run('one_loop_dd', save_type="txt", save_dir='outputs')            
            # Load using relative path
            loaded_result = handler.load(os.path.join('outputs', 'one_loop_dd_output.txt'))
            
            assert isinstance(loaded_result, tuple)
            for r1, r2 in zip(result, loaded_result):
                assert np.allclose(r1, r2)
        finally:
            # Make sure to return to original directory
            os.chdir(orig_cwd)

import shutil
@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)
################# PARAMS SAVER BASIC FUNCTIONALITY TESTS #################
def test_save_and_load_basic(handler, temp_output_dir):
    """Test basic save and load functionality"""
    # Save parameters
    param_file = os.path.join(temp_output_dir, "test_params")
    handler.save_instance(param_file)
    
    # Load parameters
    newHandler = FPTHandler.load_instance(param_file)
    
    # Check that loaded parameters match original parameters
    old_params = handler.default_params
    for key, value in newHandler.default_params.items():
        assert key in old_params
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, old_params[key])
        else:
            assert value == old_params[key]

################# FILE PATH TESTS #################
def test_file_path_handling(handler, temp_output_dir):
    """Test different ways of specifying file paths"""
    # 1. Relative filename with output_dir
    handler.save_instance("params1", output_dir=temp_output_dir)
    assert os.path.exists(os.path.join(temp_output_dir, "params1.npz"))
    
    # 2. Absolute path (output_dir should be ignored)
    abs_path = os.path.join(temp_output_dir, "params2")
    handler.save_instance(abs_path, output_dir="/should/be/ignored")
    assert os.path.exists(abs_path + ".npz")
    
    # 3. Relative path with directory component
    subdir = "subdir"
    os.makedirs(os.path.join(temp_output_dir, subdir), exist_ok=True)
    rel_path = os.path.join(subdir, "params3")
    handler.save_instance(rel_path, output_dir=temp_output_dir)
    assert os.path.exists(os.path.join(temp_output_dir, rel_path + ".npz"))

def test_file_extension_handling(handler, temp_output_dir):
    """Test handling of file extensions"""
    # Without extension (should add .npz)
    handler.save_instance("no_extension", output_dir=temp_output_dir)
    assert os.path.exists(os.path.join(temp_output_dir, "no_extension.npz"))
    
    # With .npz extension (should not duplicate)
    handler.save_instance("with_extension.npz", output_dir=temp_output_dir)
    assert os.path.exists(os.path.join(temp_output_dir, "with_extension.npz"))
    assert not os.path.exists(os.path.join(temp_output_dir, "with_extension.npz.npz"))
    
    # With other extension (should still add .npz)
    handler.save_instance("other_extension.txt", output_dir=temp_output_dir)
    assert os.path.exists(os.path.join(temp_output_dir, "other_extension.txt.npz"))

def test_directory_creation(handler):
    """Test automatic directory creation"""
    # Create a nested directory path that doesn't exist
    temp_dir = tempfile.mkdtemp()
    try:
        nested_path = os.path.join(temp_dir, "a", "b", "c")
        param_file = os.path.join(nested_path, "params")
        
        # This should create the directory structure
        handler.save_instance(param_file)
        
        # Check that the directories were created
        assert os.path.exists(nested_path)
        assert os.path.exists(param_file + ".npz")
    finally:
        shutil.rmtree(temp_dir)

def test_default_directory_fallback(fpt):
    """Test fallback to default output directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create handler with custom output directory
        handler = FPTHandler(fpt, P=P, save_dir=temp_dir)
        
        # Save without specifying output_dir (should use default)
        handler.save_instance("default_dir_params")
        assert os.path.exists(os.path.join(temp_dir, "default_dir_params.npz"))
        
        # Load without specifying load_dir - should work since we have a handler with save_dir
        loaded_handler = FPTHandler.load_instance("default_dir_params", load_dir=temp_dir)
        assert loaded_handler is not None
        assert isinstance(loaded_handler, FPTHandler)

def test_load_instance_requires_directory():
    """Test that load_instance requires a directory when not using absolute paths"""
    # This should raise an error because no load_dir is provided and filename is relative
    with pytest.raises(FileNotFoundError, match="Parameter file.*not found"):
        FPTHandler.load_instance("nonexistent_params")
    
    # Should also fail with relative path when file doesn't exist in current directory
    with pytest.raises(FileNotFoundError):
        FPTHandler.load_instance("relative_params.npz")

def test_load_instance_with_absolute_path(fpt):
    """Test that load_instance works with absolute paths without requiring load_dir"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and save a handler instance
        handler = FPTHandler(fpt, P=P, save_dir=temp_dir)
        abs_path = os.path.join(temp_dir, "abs_path_params")
        handler.save_instance(abs_path)
        
        # Load using absolute path (no load_dir needed)
        loaded_handler = FPTHandler.load_instance(abs_path + ".npz")
        assert loaded_handler is not None
        assert isinstance(loaded_handler, FPTHandler)
        
        # Should also work without the .npz extension
        loaded_handler2 = FPTHandler.load_instance(abs_path)
        assert loaded_handler2 is not None

def test_current_working_directory_behavior(fpt):
    """Test that the new current working directory behavior works correctly"""
    # Get original working directory
    orig_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Change to temp directory
            os.chdir(temp_dir)
            
            # Create handler - should use current working directory + "outputs" (temp_dir/outputs)
            handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
            
            # Save instance - should save to current directory + "outputs"
            handler.save_instance("cwd_test_params")
            
            # File should exist in temp_dir/outputs (current working directory + "outputs")
            expected_path = os.path.join(temp_dir, "outputs", "cwd_test_params.npz")
            assert os.path.exists(expected_path)
            
            # Load instance - should load from the outputs subdirectory
            outputs_dir = os.path.join(temp_dir, "outputs")
            loaded_handler = FPTHandler.load_instance("cwd_test_params", load_dir=outputs_dir)
            assert loaded_handler is not None
            assert isinstance(loaded_handler, FPTHandler)
            
            # Verify the loaded parameters match
            assert np.array_equal(loaded_handler.default_params['P'], P)
            assert loaded_handler.default_params['C_window'] == C_window
            
        finally:
            # Always restore original working directory
            os.chdir(orig_cwd)

def test_save_dir_parameter_override(fpt):
    """Test that explicit save_dir parameter overrides current working directory default"""
    orig_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            try:
                # Change to temp_dir1
                os.chdir(temp_dir1)
                
                # Create handler with explicit save_dir (should override cwd)
                handler = FPTHandler(fpt, P=P, save_dir=temp_dir2)
                
                # Save instance - should save to temp_dir2, not current directory (temp_dir1)
                handler.save_instance("override_test_params")
                
                # File should be in temp_dir2, not temp_dir1
                assert os.path.exists(os.path.join(temp_dir2, "override_test_params.npz"))
                assert not os.path.exists(os.path.join(temp_dir1, "override_test_params.npz"))
                
                # Load should work from temp_dir2
                loaded_handler = FPTHandler.load_instance("override_test_params", load_dir=temp_dir2)
                assert loaded_handler is not None
                
            finally:
                os.chdir(orig_cwd)

################# LOAD TESTS #################
def test_load_nonexistent_file(handler):
    """Test loading a file that doesn't exist"""
    with pytest.raises(FileNotFoundError):
        handler.load_instance("nonexistent_file")

def test_load_with_absolute_path(handler, temp_output_dir):
    """Test loading using an absolute file path"""
    # Save parameters
    param_file = os.path.join(temp_output_dir, "abs_path_params")
    handler.save_instance(param_file)
    
    # Load using absolute path
    loaded_params = handler.load_instance(param_file)
    assert loaded_params is not None
    
    # Load using absolute path with .npz extension
    loaded_params = handler.load_instance(param_file + ".npz")
    assert loaded_params is not None
    
    # Load with absolute path but specifying load_dir (should ignore load_dir)
    loaded_params = handler.load_instance(param_file, load_dir="/should/be/ignored")
    assert loaded_params is not None

################# INTEGRATION TESTS #################
def test_save_and_use_in_run(fpt, temp_output_dir):
    """Test saving parameters and then using them in a run"""
    # Create and save parameters
    custom_params = {
        'P': P,
        'P_window': P_window,
        'C_window': C_window
    }
    handler = FPTHandler(fpt, **custom_params)
    param_file = os.path.join(temp_output_dir, "run_params")
    handler.save_instance(param_file)
    
    # Load parameters
    newHandler = FPTHandler.load_instance(param_file)
    
    # Use loaded parameters in a run
    result = newHandler.run('one_loop_dd')
    
    # Compare with direct run
    direct_result = handler.run('one_loop_dd')
    
    # Results should be identical
    assert np.array_equal(result, direct_result)
            

################# POWER SPECTRA GENERATOR TESTS #################

def test_generate_power_spectra_basic(handler):
    pytest.importorskip("classy", reason="classy is not installed")
    pytest.importorskip("camb", reason="camb is not installed")
    """Test basic single-mode power spectra generation with default parameters"""
    # Test class with default parameters
    class_result = handler.generate_power_spectra(method='class')
    assert isinstance(class_result, np.ndarray)
    assert len(class_result) == len(handler.fastpt.k_original)
    assert np.all(class_result > 0)  # Power spectrum should be positive
    
    # Test CAMB with default parameters
    camb_result = handler.generate_power_spectra(method='camb')
    assert isinstance(camb_result, np.ndarray)
    assert len(camb_result) == len(handler.fastpt.k_original)
    assert np.all(camb_result > 0)

def test_generate_power_spectra_methods(handler):
    pytest.importorskip("classy", reason="classy is not installed")
    pytest.importorskip("camb", reason="camb is not installed")
    """Test power spectra generation with different methods"""
    # Generate power spectra with different methods but same params
    class_result = handler.generate_power_spectra(method='class', omega_cdm=0.12, h=0.7, omega_b=0.022, z=0.5)
    camb_result = handler.generate_power_spectra(method='camb', omega_cdm=0.12, h=0.7, omega_b=0.022, z=0.5)
    
    # Results should be similar but not identical (within ~10%)
    # This is a rough check that both methods are working and producing reasonable results
    ratio = np.mean(class_result / camb_result)
    assert 0.8 < ratio < 1.2, f"Results differ too much, ratio: {ratio}"

def test_generate_power_spectra_invalid_method(handler):
    """Test with invalid method names"""
    with pytest.raises(ValueError, match="Invalid method"):
        handler.generate_power_spectra(method='invalid')

def test_generate_power_spectra_invalid_mode(handler):
    """Test with invalid mode names"""
    with pytest.raises(ValueError, match="Invalid mode"):
        handler.generate_power_spectra(mode='invalid')

def test_generate_power_spectra_single_mode_array_error(handler):
    """Test that arrays cannot be passed in single mode"""
    with pytest.raises(ValueError, match="must be a single value"):
        handler.generate_power_spectra(omega_cdm=[0.1, 0.2])

def test_generate_power_spectra_params(handler):
    pytest.importorskip("classy", reason="classy is not installed")
    pytest.importorskip("camb", reason="camb is not installed")
    """Test power spectra generation with different parameter values"""
    # Generate with different parameter values
    base_result = handler.generate_power_spectra(method='class')
    high_h_result = handler.generate_power_spectra(method='class', h=0.75)
    high_cdm_result = handler.generate_power_spectra(method='class', omega_cdm=0.15)
    
    # Results should be different with different parameters
    assert not np.allclose(base_result, high_h_result)
    assert not np.allclose(base_result, high_cdm_result)

def test_bulk_power_spectra(handler):
    pytest.importorskip("classy", reason="classy is not installed")
    pytest.importorskip("camb", reason="camb is not installed")
    """Test bulk mode power spectra generation"""
    # Test with arrays of different lengths
    bulk_results = handler.generate_power_spectra(
        mode='bulk',
        omega_cdm=[0.11, 0.12, 0.13],
        h=[0.67, 0.68],
        omega_b=0.022,
        z=[0.0]
    )
    
    # Should return a list of results with length = max(param_length)
    assert isinstance(bulk_results, list)
    assert len(bulk_results) == 3  # Max length of input parameters
    
    # Each result should be a proper power spectrum
    for result in bulk_results:
        assert isinstance(result, np.ndarray)
        assert len(result) == len(handler.fastpt.k_original)
        assert np.all(result > 0)

def test_bulk_power_spectra_single_entry(handler):
    pytest.importorskip("classy", reason="classy is not installed")
    pytest.importorskip("camb", reason="camb is not installed")
    """Test bulk mode with single entry arrays"""
    # When all parameters are length 1, should return a single result
    single_bulk_result = handler.generate_power_spectra(
        mode='bulk',
        omega_cdm=[0.12],
        h=[0.67],
        omega_b=[0.022],
        z=[0.0]
    )
    
    # Should be a single array, not a list
    assert isinstance(single_bulk_result, np.ndarray)
    assert len(single_bulk_result) == len(handler.fastpt.k_original)

def test_diff_power_spectra_basic(handler):
    pytest.importorskip("classy", reason="classy is not installed")
    pytest.importorskip("camb", reason="camb is not installed")
    """Test diff mode power spectra generation"""
    # Test with basic parameters
    diff_results = handler.generate_power_spectra(
        mode='diff',
        omega_cdm=[0.11, 0.12, 0.13],
        h=0.67,
        omega_b=0.022,
        z=0.0
    )
    
    # Should return a dictionary keyed by parameter tuples
    assert isinstance(diff_results, dict)
    
    # Should contain results for central value + 2 variations
    assert len(diff_results) == 3  # 1 central + 2 variations
    
    # Check format of keys and values
    for key, value in diff_results.items():
        assert isinstance(key, tuple)
        assert len(key) == 2  # (z and int)
        assert isinstance(value, np.ndarray)
        assert len(value) == len(handler.fastpt.k_original)

def test_diff_power_spectra_multi_param(handler):
    pytest.importorskip("classy", reason="classy is not installed")
    pytest.importorskip("camb", reason="camb is not installed")
    """Test diff mode with multiple variable parameters"""
    diff_results = handler.generate_power_spectra(
        mode='diff',
        omega_cdm=[0.11, 0.12, 0.13],
        h=[0.66, 0.67, 0.68],
        omega_b=0.022,
        z=0.0
    )
    
    # Should contain results for central value + 2+2 variations
    assert isinstance(diff_results, dict)
    assert len(diff_results) == 5  # 1 central + 2*2 variations
    
    assert (0.0, 0) in diff_results.keys(), "Central parameter combination not found"
    assert (0.0, -1) in diff_results.keys(), "omega_cdm low variation not found"
    assert (0.0, 2) in diff_results.keys(), "h high variation not found"

def test_diff_power_spectra_requires_length_3(handler):
    pytest.importorskip("classy", reason="classy is not installed")
    pytest.importorskip("camb", reason="camb is not installed")
    """Test that diff mode requires at least one parameter with length 3"""
    with pytest.raises(ValueError, match="must have length 3"):
        handler.generate_power_spectra(
            mode='diff',
            omega_cdm=[0.12],
            h=[0.67],
            omega_b=[0.022],
            z=[0.0]
        )

def test_diff_power_spectra_with_multiple_z(handler):
    pytest.importorskip("classy", reason="classy is not installed")
    pytest.importorskip("camb", reason="camb is not installed")
    """Test diff mode with multiple redshifts"""
    diff_results = handler.generate_power_spectra(
        mode='diff',
        omega_cdm=[0.11, 0.12, 0.13],
        h=0.67,
        omega_b=0.022,
        z=[0.0, 0.5]
    )
    
    # Should return results for each z
    assert isinstance(diff_results, dict)
    assert len(diff_results) == 6  # 3 param combinations * 2 redshifts
    
    # Check that we have results for both redshifts
    z0_keys = [k for k in diff_results.keys() if k[0] == 0.0]
    z05_keys = [k for k in diff_results.keys() if k[0] == 0.5]
    assert len(z0_keys) == 3
    assert len(z05_keys) == 3

def test_camb_specific_params(handler):
    pytest.importorskip("classy", reason="classy is not installed")
    pytest.importorskip("camb", reason="camb is not installed")
    """Test CAMB-specific parameters"""
    # With nonlinear=True
    result_nl = handler.generate_power_spectra(
        method='camb',
        nonlinear=True
    )
    
    # With nonlinear=False
    result_linear = handler.generate_power_spectra(
        method='camb',
        nonlinear=False
    )
    
    # Results should be different
    assert not np.allclose(result_nl, result_linear)
    
    # With different halofit version
    result_halofit = handler.generate_power_spectra(
        method='camb',
        halofit_version='takahashi'
    )
    
    # Should be different from default (mead)
    assert not np.allclose(result_nl, result_halofit)

def test_class_camb_parameter_consistency(handler):
    pytest.importorskip("classy", reason="classy is not installed")
    pytest.importorskip("camb", reason="camb is not installed")
    """Test consistency in parameter handling between CLASS and CAMB"""
    # Generate spectra with same parameters
    params = {
        'omega_cdm': 0.12,
        'h': 0.67,
        'omega_b': 0.022,
        'z': 0.5
    }
    
    class_result = handler.generate_power_spectra(method='class', **params)
    camb_result = handler.generate_power_spectra(method='camb', **params)
    
    # Results should be similar (within reasonable bounds)
    # Note: Some differences are expected due to different implementations
    assert len(class_result) == len(camb_result)
    
    # Check ratio over most of the range (excluding extremes)
    k_idx_range = slice(len(handler.fastpt.k_original) // 10, -len(handler.fastpt.k_original) // 10)
    ratio = class_result[k_idx_range] / camb_result[k_idx_range]
    assert 0.998 < np.median(ratio) < 1.002, "CLASS and CAMB results differ significantly"
