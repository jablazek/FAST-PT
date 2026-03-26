from fastpt import FASTPT
import numpy as np
import os
from time import time
from statistics import mean, stdev
import json
from datetime import datetime

def run_benchmark(fpt, function_name, params, n_runs=5):
    """Run a specific FASTPT function multiple times and collect timing stats."""
    times = []
    method = getattr(fpt, function_name)
    
    for _ in range(n_runs):
        t0 = time()
        method(**params)
        t1 = time()
        times.append(t1 - t0)
    
    return {
        'mean': mean(times),
        'std': stdev(times),
        'min': min(times),
        'max': max(times),
        'all_runs': times
    }


def numpy_to_python(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, 
        np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {'real': obj.real, 'imag': obj.imag}
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    return obj


def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'Pk_test.dat')
    d = np.loadtxt(data_path)
    k = d[:, 0]
    P = d[:, 1]
    P_window = np.array([0.2, 0.2])
    C_window = 0.75
    n_pad = int(0.5 * len(k))
    fpt = FASTPT(k, to_do=['all'], low_extrap=-5, high_extrap=3, n_pad=n_pad)
    
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
        'IA_ctbias': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_gb2': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_d2': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_s2': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'OV': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'kPol': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'RSD_components': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5},
        'RSD_ABsum_components': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5},
        'RSD_ABsum_mu': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5, 'mu_n': 0.5},
        'J_k_scalar': {'P': P, 'X': fpt.X_spt, 'nu': -2, 'P_window': P_window, 'C_window': C_window},
        'J_k_tensor': {'P': P, 'X': fpt.X_IA_E, 'P_window': P_window, 'C_window': C_window},
    }
    
    results = {}

        
    for func in function_params:
        print(f"Testing {func}...")
        try:
            results[func] = run_benchmark(fpt, func, function_params[func])
        except Exception as e:
            print(f"Error running {func}: {str(e)}")
            results[func] = {'error': str(e)}
    
    total_time = 0
    for func, result in results.items():
        if isinstance(result, dict) and 'mean' in result:
            total_time += result['mean']
    
    results['total_time'] = total_time

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(os.path.dirname(__file__), f'timed_results_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(numpy_to_python(results), f, indent=2)

if __name__ == '__main__':
    main()
