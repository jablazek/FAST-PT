import pytest
import numpy as np
from fastpt import FASTPT
import os

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
P = np.loadtxt(data_path)[:, 1]
C_window = 0.75

@pytest.fixture
def fpt(): 
    d = np.loadtxt(data_path)
    k = d[:, 0]
    n_pad = int(0.5 * len(k))
    return FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad)


####################INITIALIZATION TESTS####################
def test_init_empty_arrays():
    with pytest.raises(ValueError):
        FASTPT([], [])

def test_init_odd_length_k():
    """Test initialization with odd-length k array"""
    k = np.logspace(-3, 1, 201)  # Odd length
    with pytest.raises(ValueError):
        FASTPT(k)

def test_init_non_log_spaced():
    """Test initialization with non-log-spaced k array"""
    k = np.linspace(0.1, 10, 200)  # Linear spacing
    with pytest.raises(AssertionError):
        FASTPT(k)

def test_init_invalid_to_do():
    """Test initialization with invalid to_do parameter"""
    k = np.logspace(-3, 1, 200)
    with pytest.raises(ValueError):
        FASTPT(k, to_do=['invalid_option'])

def test_init_extrapolation_ranges():
    """Test initialization with various extrapolation ranges"""
    k = np.logspace(-3, 1, 200)
            
    # Test valid extrapolation
    fpt = FASTPT(k, low_extrap=-5, high_extrap=3)
    assert fpt.low_extrap == -5
    assert fpt.high_extrap == 3
            
    # Test invalid extrapolation
    with pytest.raises(ValueError):
        FASTPT(k, low_extrap=3, high_extrap=-5)  # Invalid range

def test_init_padding(fpt):
    """Test initialization with different padding values"""
    k = np.logspace(-3, 1, 200)

    # Test with no padding
    fpt1 = FASTPT(k, n_pad=None)
    assert fpt1.n_pad == int(0.5 * len(k))  # Default padding
            
    # Test with padding
    assert hasattr(fpt, 'n_pad')

@pytest.mark.parametrize("to_do", [['all'], ['one_loop_dd'], ['dd_bias'], 
                                   ['one_loop_cleft_dd'], ['IA_tt'], 
                                   ['IA_mix'], ['IA_ta'], ['OV'], 
                                   ['kPol'], ['RSD'], ['tij'], ['gb2'], 
                                   ['IRres'], ['all'], ['everything']])
def test_all_todos(to_do):
    """Test initialization with all possible to_do options"""
    k = np.logspace(-3, 1, 200)
    fpt = FASTPT(k, to_do=to_do)
    assert fpt is not None


def test_skip_todo(fpt):
    """Test to make sure skip is faster (doesn't initialize anything)"""
    from time import time
    t0 = time()
    slowPT = FASTPT(fpt.k_original, to_do=['all'])
    t1 = time()
    diff1 = t1 - t0
    t2 = time()
    fastPT = FASTPT(fpt.k_original)
    t3 = time()
    diff2 = t3 - t2
    assert diff2 < diff1


####################PARAMETER VALIDATION TESTS####################
def test_validate_params(fpt):
    """Test the validate_params function with various inputs
        (Using one_loop_dd as a sample method though all decorating functions
        will follow the same validation behavior)"""
    P_window = np.array([0.2, 0.2])
    # Test 1: Valid cases
    assert fpt.one_loop_dd(P) is not None
    assert fpt.one_loop_dd(P, P_window=P_window, C_window=0.5) is not None
    
    # Test 2: Empty or None power spectrum
    with pytest.raises(ValueError, match=r'You must provide an input power spectrum array'):
        fpt.one_loop_dd(None)
    with pytest.raises(ValueError, match=r'You must provide an input power spectrum array'):
        fpt.one_loop_dd([])

    #Test mismatched P and k
    half_P = P[:len(P)//2]
    with pytest.raises(ValueError, match=r'Input k and P arrays must have the same size'):
        fpt.one_loop_dd(half_P)
        
    # Test 3: Zero power spectrum
    P_zero = np.zeros_like(fpt.k_original)
    with pytest.raises(ValueError, match=r'Your input power spectrum array is all zeros'):
        fpt.one_loop_dd(P_zero)
    
    # Test 4: P_window validation
    max_window = (np.log(fpt.k_final[-1]) - np.log(fpt.k_final[0])) / 2
    Max_P_window = np.array([max_window, max_window])
    print(f"Max_P_window: {Max_P_window}")
    assert fpt.one_loop_dd(P, P_window=Max_P_window / 2) is not None
    
    with pytest.raises(ValueError, match=r'P_window must be a tuple of two values.'):
        fpt.one_loop_dd(P, P_window=Max_P_window[:-1])

    with pytest.raises(ValueError, match=r'P_window value is too large'):
        fpt.one_loop_dd(P, P_window=Max_P_window * 2)
        
    # Test 5: C_window validation
    # Test valid C_window values
    assert fpt.one_loop_dd(P, C_window=0.0) is not None
    assert fpt.one_loop_dd(P, C_window=0.5) is not None
    assert fpt.one_loop_dd(P, C_window=1.0) is not None
    
    # Test invalid C_window values
    with pytest.raises(ValueError, match=r'C_window must be between 0 and 1'):
        fpt.one_loop_dd(P, C_window=-0.1)
    with pytest.raises(ValueError, match=r'C_window must be between 0 and 1'):
        fpt.one_loop_dd(P, C_window=1.1)
        
    # Test 6: Combined parameter validation
    with pytest.raises(ValueError):
        fpt.one_loop_dd(None, P_window=P_window, C_window=0.5)
    with pytest.raises(ValueError):
        fpt.one_loop_dd(P, P_window=Max_P_window * 2, C_window=1.1)





####################UNIT TESTS####################
def test_one_loop_dd(fpt):
    """Test the one_loop_dd function with various inputs"""
    # Test with standard input
    result = fpt.one_loop_dd(P)
    assert isinstance(result, tuple)

    # Test with window functions
    P_window = np.array([0.2, 0.2])
    result_window = fpt.one_loop_dd(P, P_window=P_window, C_window=C_window)
    assert isinstance(result_window, tuple)
    
    # Test shape consistency
    assert result[0].shape == P.shape

def test_one_lood_dd_bias(fpt):
    """Test the one_loop_dd_bias function including bias terms"""
        
    # Test standard calculation
    result = fpt.one_loop_dd_bias(P)
    assert isinstance(result, tuple)
        
    # Verify sigma4 calculation is positive
    assert result[-1] > 0  # sig4 should be positive
        
    # Test bias terms have correct shapes
    for term in result[1:-2]:  # Skip P_1loop and sig4
        assert term.shape == P.shape
            
    # Test with window functions
    result_window = fpt.one_loop_dd_bias(P, P_window=None, C_window=C_window)
    assert isinstance(result_window, tuple)

def test_one_loop_dd_bias_b3nl(fpt):
    """Test the one_loop_dd_bias_b3nl function including b3nl terms"""
        
    # Test standard calculation
    result = fpt.one_loop_dd_bias_b3nl(P)
    assert isinstance(result, tuple)
        
    # Test sig3nl term
    assert result[-1].shape == P.shape
        
    # Test with window functions
    result_window = fpt.one_loop_dd_bias_b3nl(P, P_window=None, C_window=C_window)
    assert isinstance(result_window, tuple)
        
    # Verify consistency between b3nl and standard bias results
    result_bias = fpt.one_loop_dd_bias(P)
    # for i in range(min(len(result), len(result_bias))):
    #     assert np.allclose(result[i], result_bias[i])
    # Enhanced debugging for the failing comparison
    for i in range(min(len(result), len(result_bias))):
        try:
            assert np.allclose(result[i], result_bias[i])
        except AssertionError:
            # Provide detailed debugging information
            print(f"\n==== Detailed Comparison for Index {i} ====")
            print(f"Shape of result[{i}]: {len(result[i])}, Shape of result_bias[{i}]: {len(result_bias[i])}")
            
            if isinstance(result[i], np.ndarray) and isinstance(result_bias[i], np.ndarray):
                # Calculate statistics about the differences
                abs_diff = np.abs(result[i] - result_bias[i])
                rel_diff = abs_diff / np.abs(result_bias[i])
                rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)
                
                print(f"Maximum absolute difference: {np.max(abs_diff)}")
                print(f"Mean absolute difference: {np.mean(abs_diff)}")
                print(f"Maximum relative difference: {np.max(rel_diff)}")
                print(f"Mean relative difference: {np.mean(rel_diff)}")
                
                # Print some sample values where differences are largest
                max_diff_idx = np.argmax(abs_diff)
                print(f"\nAt index of max difference ({max_diff_idx}):")
                print(f"  result[{i}][{max_diff_idx}] = {result[i][max_diff_idx]}")
                print(f"  result_bias[{i}][{max_diff_idx}] = {result_bias[i][max_diff_idx]}")
                
                # If the arrays are small enough, print all non-matching elements
                if len(result[i]) < 20:
                    print("\nAll elements comparison:")
                    for j in range(len(result[i])):
                        if not np.isclose(result[i][j], result_bias[i][j]):
                            print(f"  Index {j}: {result[i][j]} vs {result_bias[i][j]}")
            
            # Print the full arrays if scalar values
            else:
                print(f"Full values: {result[i]} vs {result_bias[i]}")
            
            # Re-raise the assertion with more context
            raise AssertionError(f"Arrays at index {i} are not equal within tolerance")

def test_one_loop_dd_bias_lpt_NL(fpt):
    """Test the one_loop_dd_bias_lpt_NL function"""
        
    # Test standard calculation
    result = fpt.one_loop_dd_bias_lpt_NL(P)
    assert isinstance(result, tuple)
        
    # Test shapes of LPT bias terms
    expected_terms = ['Pb1L', 'Pb1L_2', 'Pb1L_b2L', 'Pb2L', 'Pb2L_2']
    for term, name in zip(result, expected_terms):
        assert term.shape == P.shape, f"{name} has incorrect shape"
        
    # Test with window functions
    result_window = fpt.one_loop_dd_bias_lpt_NL(P, P_window=None, C_window=0.75)
    assert isinstance(result_window, tuple)

#def test_cleft_Q_R(fpt):
#   assert True

def test_IA_tt(fpt):
    """Test the IA_tt function"""
    result = fpt.IA_tt(P)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].shape == P.shape
    assert result[1].shape == P.shape

def test_IA_mix(fpt):
    """Test the IA_mix function"""
    result = fpt.IA_mix(P)
    assert isinstance(result, tuple)
    assert len(result) == 4
    for term in result:
        assert term.shape == P.shape

def test_IA_ta(fpt):
    """Test the IA_ta function"""
    result = fpt.IA_ta(P)
    assert isinstance(result, tuple)
    assert len(result) == 4
    for term in result:
        assert term.shape == P.shape

def test_IA_der(fpt):
    """Test the IA_der function"""
    result = fpt.IA_der(P)
    assert isinstance(result, np.ndarray)
    assert result.shape == P.shape

def test_IA_ct(fpt):
    """Test the IA_ct function"""
    result = fpt.IA_ct(P)
    assert isinstance(result, tuple)
    assert len(result) == 4
    for term in result:
        assert term.shape == P.shape

def test_gI_ct(fpt):
    """Test the gI_ct function"""
    result = fpt.gI_ct(P)
    assert isinstance(result, tuple)
    assert len(result) == 2
    for term in result:
        assert term.shape == P.shape

def test_gI_ta(fpt):
    """Test the IA_d2 function"""
    result = fpt.gI_ta(P)
    assert isinstance(result, tuple)
    assert len(result) == 4
    for term in result:
        assert term.shape == P.shape

def test_gI_tt(fpt):
    """Test the IA_s2 function"""
    result = fpt.gI_tt(P)
    assert isinstance(result, tuple)
    assert len(result) == 2
    for term in result:
        assert term.shape == P.shape

def test_OV(fpt):
    """Test the OV function"""
    result = fpt.OV(P)
    assert isinstance(result, np.ndarray)
    assert result.shape == P.shape

def test_kPol(fpt):
    """Test the kPol function"""
    result = fpt.kPol(P)
    assert isinstance(result, tuple)
    assert len(result) == 3
    for term in result:
        assert term.shape == P.shape

def test_RSD_components(fpt):
    """Test the RSD_components function"""
    f = 0.5  # Example growth rate
    result = fpt.RSD_components(P, f)
    assert isinstance(result, tuple)
    assert len(result) == 10
    for term in result:
        assert term.shape == P.shape

def test_RSD_ABsum_components(fpt):
    """Test the RSD_ABsum_components function"""
    f = 0.5  # Example growth rate
    result = fpt.RSD_ABsum_components(P, f)
    assert isinstance(result, tuple)
    assert len(result) == 4
    for term in result:
        assert term.shape == P.shape

def test_RSD_ABsum_mu(fpt):
    """Test the RSD_ABsum_mu function"""
    f = 0.5  # Example growth rate
    mu_n = 0.5  # Example mu_n value
    result = fpt.RSD_ABsum_mu(P, f, mu_n)
    assert isinstance(result, np.ndarray)
    assert result.shape == P.shape

def test_IRres(fpt):
    """Test the IRres function"""
    result = fpt.IRres(P)
    assert isinstance(result, np.ndarray)
    assert result.shape == P.shape

def test_hash_none(fpt):
    """Test hashing None values"""
    result = fpt._hash_arrays(None)
    assert result == hash(None)

def test_hash_numpy_array(fpt):
    """Test hashing numpy arrays"""
    # 1D array
    arr1d = np.array([1, 2, 3, 4, 5])
    result1d = fpt._hash_arrays(arr1d)
    assert isinstance(result1d, int)
    
    # 2D array
    arr2d = np.array([[1, 2], [3, 4]])
    result2d = fpt._hash_arrays(arr2d)
    assert isinstance(result2d, int)
    
    # Array with different dtype
    arr_float = np.array([1.1, 2.2, 3.3])
    result_float = fpt._hash_arrays(arr_float)
    assert isinstance(result_float, int)
    
    # Test that identical arrays produce the same hash
    arr_copy = np.copy(arr1d)
    result_copy = fpt._hash_arrays(arr_copy)
    assert result1d == result_copy
    
    # Test that different arrays produce different hashes
    arr_diff = np.array([1, 2, 3, 4, 6])  # Changed last element
    result_diff = fpt._hash_arrays(arr_diff)
    assert result1d != result_diff

def test_hash_tuple_list(fpt):
    """Test hashing tuples and lists of items"""
    # Simple tuple of scalars
    tuple_result = fpt._hash_arrays((1, 2, 3))
    assert isinstance(tuple_result, int)
    
    # Simple list of scalars
    list_result = fpt._hash_arrays([1, 2, 3])
    assert isinstance(list_result, int)
    
    # Mixed list
    mixed_list = [1, "string", True]
    mixed_result = fpt._hash_arrays(mixed_list)
    assert isinstance(mixed_result, int) #Now returns single hash

def test_hash_X_parameters(fpt):
    """Test hashing X parameter tuples (like X_spt, X_lpt, etc.)"""
    # Create a mock X parameter tuple similar to what's returned by scalar_stuff
    mock_pf = np.array([1.0, 2.0, 3.0])
    mock_p = np.array([0.5, 1.5, 2.5])
    mock_g_m = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
    mock_g_n = np.array([[9+10j, 11+12j], [13+14j, 15+16j]])
    mock_two_part_l = np.array([[17+18j, 19+20j], [21+22j, 23+24j]])
    mock_h_l = np.array([[25+26j, 27+28j], [29+30j, 31+32j]])
    
    mock_X = (mock_pf, mock_p, mock_g_m, mock_g_n, mock_two_part_l, mock_h_l)
    X_hash = fpt._hash_arrays(mock_X)
    
    assert isinstance(X_hash, int)

def test_hash_consistency(fpt):
    """Test that the same input always produces the same hash"""
    # Array
    arr = np.array([1, 2, 3, 4, 5])
    hash1 = fpt._hash_arrays(arr)
    hash2 = fpt._hash_arrays(arr)
    assert hash1 == hash2
    
    # Complex structure
    complex_struct = ([1, 2, 3], np.array([4, 5, 6]), (7, 8, 9))
    hash1 = fpt._hash_arrays(complex_struct)
    hash2 = fpt._hash_arrays(complex_struct)
    assert hash1 == hash2

def test_hash_uniqueness(fpt):
    """Test that different inputs produce different hashes"""
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([1, 2, 4])  # Just one element different
    hash1 = fpt._hash_arrays(arr1)
    hash2 = fpt._hash_arrays(arr2)
    assert hash1 != hash2
    
    # Similar structure but different values
    struct1 = ([1, 2, 3], np.array([4, 5, 6]))
    struct2 = ([1, 2, 3], np.array([4, 5, 7]))  # Just one element different
    hash1 = fpt._hash_arrays(struct1)
    hash2 = fpt._hash_arrays(struct2)
    assert hash1 != hash2

def test_hash_power_spectrum(fpt, monkeypatch):
    """Test hashing and cache key generation for power spectrum"""
    
    # Test direct hashing
    P_hash = fpt._hash_arrays(P)
    assert isinstance(P_hash, int)
    
    # Test that create_hash_key returns both the hash_key and P_hash
    # Use some dummy values for testing
    term = "test_term"
    X = fpt.X_spt
    P_window = None
    C_window = 0.75
    
    hash_key, returned_P_hash = fpt._create_hash_key(term, X, P, P_window, C_window)
    assert isinstance(hash_key, int)
    assert isinstance(returned_P_hash, int)
    assert returned_P_hash == P_hash, "P_hash should match the direct hash of P"
    
    # Test access to cached properties creates entries in X_registry
    assert id(X) in fpt.X_registry
    
    mock_gets = []
    mock_sets = []
    original_get = fpt.cache.get
    original_set = fpt.cache.set
    
    def mock_get(*args):
        mock_gets.append(args)
        return original_get(*args)
        
    def mock_set(*args):
        mock_sets.append(args)
        return original_set(*args)
    
    monkeypatch.setattr(fpt.cache, 'get', mock_get)
    monkeypatch.setattr(fpt.cache, 'set', mock_set)
    
    fpt.one_loop_dd(P)
    assert len(mock_gets) > 0, "Cache get was not called"
    assert len(mock_sets) > 0, "Cache set was not called"

    for call in mock_gets:
        assert len(call) >= 2, "Cache get call should have category and hash"
        assert isinstance(call[0], str), "First arg should be category (string)"
        assert isinstance(call[1], int), "Second arg should be hash key (int)"
    
    mock_gets.clear()
    mock_sets.clear()
    
    # Call the same function again - should use cache
    result1 = fpt.one_loop_dd(P)
    
    assert len(mock_gets) > 0, "Cache get was not called on repeated call"
    
    # Different P should create different hash
    P_modified = np.copy(P)
    P_modified[0] *= 1.01
    
    # Verify P_hash correctly identifies different arrays
    P_hash_original = fpt._hash_arrays(P)
    P_hash_modified = fpt._hash_arrays(P_modified)
    assert P_hash_original != P_hash_modified, "Different P arrays should have different hash values"
    
    # Test that create_hash_key returns different P_hash values for different P
    hash_key1, P_hash1 = fpt._create_hash_key(term, X, P, P_window, C_window)
    hash_key2, P_hash2 = fpt._create_hash_key(term, X, P_modified, P_window, C_window)
    assert P_hash1 != P_hash2, "P_hash values should differ for different P arrays"
    assert hash_key1 != hash_key2, "Hash keys should differ for different P arrays"
    
    mock_gets.clear()
    mock_sets.clear()
    
    result2 = fpt.one_loop_dd(P_modified)
    
    # Should have both gets and sets (different input = new cache entry)
    assert len(mock_sets) > 0, "Cache set wasn't called with new input"
    assert not np.array_equal(result1[0], result2[0]), "Results should differ with different inputs"

def test_hash_nested_numpy_arrays(fpt):
    """Test hashing complex nested structures with numpy arrays"""
    # Create a complex nested structure with numpy arrays
    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([[4.0, 5.0], [6.0, 7.0]])
    arr3 = np.array([8.0, 9.0, 10.0])
    
    # Create a deep nested structure: tuple -> list -> tuple -> array
    nested_structure = (
        [arr1, (arr2, arr3)], 
        (np.array([11.0, 12.0]), [np.array([13.0]), np.array([[14.0, 15.0]])])
    )
    
    # Hash the nested structure
    hash_result = fpt._hash_arrays(nested_structure)
    
    # Verify structure and types
    assert isinstance(hash_result, int)
    
    # Test consistency - same structure should hash to same value
    hash_result2 = fpt._hash_arrays(nested_structure)
    assert hash_result == hash_result2
    
    # Test that modifying a deep nested array changes the hash
    modified_structure = (
        [arr1, (arr2, np.array([8.0, 9.0, 10.1]))],  # Changed last value from 10.0 to 10.1
        (np.array([11.0, 12.0]), [np.array([13.0]), np.array([[14.0, 15.0]])])
    )
    modified_hash = fpt._hash_arrays(modified_structure)
    assert hash_result != modified_hash
    
    # Test typical deep nested X parameter scenarios from FAST-PT
    # X parameters might contain tuples with numpy arrays of complex values
    complex_arr = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
    complex_nested = (arr1, arr2, complex_arr, (arr3, complex_arr))
    
    complex_hash = fpt._hash_arrays(complex_nested)
    assert isinstance(complex_hash, int)
