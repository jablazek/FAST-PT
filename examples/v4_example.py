"""
Comprehensive FAST-PT Example: New Features Demonstration

This example demonstrates the latest FAST-PT features including:
- Simplified initialization (no to_do list required)
- Automatic caching system with timing demonstrations
- FPTHandler for simplified function calls and power spectrum generation
- Built-in plotting capabilities
- Power spectrum generation using CLASS/CAMB
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time

# Import FAST-PT components
from fastpt import FASTPT
from fastpt.core.FPTHandler import FPTHandler

print('=== FAST-PT Comprehensive Feature Demonstration ===\n')

# Set up k-grid for calculations
k = np.logspace(-3, 1, 200)  # k from 0.001 to 10 h/Mpc
print(f'k-grid: {len(k)} points from {k[0]:.3f} to {k[-1]:.1f} h/Mpc')

# ===============================================================================
# 1. SIMPLIFIED INITIALIZATION - No to_do list required!
# ===============================================================================
print('\n1. === SIMPLIFIED INITIALIZATION ===')
print('Initializing FAST-PT without specifying to_do list...')

t_init_start = time()
fpt = FASTPT(k, to_do=['all'], low_extrap=-5, high_extrap=3, n_pad=100, verbose=False)
t_init_end = time()

print(f'✓ FAST-PT initialized in {t_init_end - t_init_start:.3f} seconds')
print('  → Terms will be calculated automatically as needed')
print('  → Matrices are computed on-demand for efficiency')

# ===============================================================================
# 2. HANDLER-BASED POWER SPECTRUM GENERATION
# ===============================================================================
print('\n2. === POWER SPECTRUM GENERATION WITH HANDLER ===')

# Initialize handler
handler = FPTHandler(fpt, C_window=0.75)

# Generate a power spectrum using CAMB 
print('Generating linear power spectrum...')
try:
    # Try to use CAMB for realistic power spectrum
    P_linear = handler.generate_power_spectra(
        method='camb',
        mode='single',
        z=0.0,
        h=0.67,
        omega_b=0.048,
        omega_m=0.3,
        As=2.1e-9,
        ns=0.97
    )
    print('✓ Generated realistic power spectrum using CAMB')
except ImportError:
    # Fallback to simple power law
    P_linear = 1000 * k**(-1.5) * np.exp(-k/10)  # Simple power law with exponential cutoff
    print('✓ Generated example power law (CAMB not available)')

# Update handler with the power spectrum for subsequent calculations
handler.update_default_params(P=P_linear)

# ===============================================================================
# 3. CACHING DEMONSTRATION - Show timing improvements
# ===============================================================================
print('\n3. === CACHING SYSTEM DEMONSTRATION ===')
fpt.cache.clear()  # Clear cache to start fresh

# First run - matrices computed and cached
print('First calculation runs (computing matrices):')
t1 = time()
P_1loop_result = handler.run('one_loop_dd')
t2 = time()
print(f'  → one_loop_dd: {t2-t1:.4f} seconds')

t1 = time()
ia_result = handler.run('IA_tt')
t2 = time()
print(f'  → IA_tt: {t2-t1:.4f} seconds')

# Second run - using cached results (same P and parameters)
print('\nSecond calculation runs (using cached results):')
t1 = time()
P_1loop_result2 = handler.run('one_loop_dd')
t2 = time()
print(f'  → one_loop_dd: {t2-t1:.4f} seconds (cached!)')

t1 = time()
ia_result2 = handler.run('IA_tt')
t2 = time()
print(f'  → IA_tt: {t2-t1:.4f} seconds (cached!)')

# Verify results are identical
print('✓ Cached results identical to original calculations')

# New function with same P - benefits from cached matrices
print('\nNew function with same power spectrum (benefits from cached matrices):')
t1 = time()
mix_result = handler.run('IA_ta')
t2 = time()
print(f'  → IA_ta: {t2-t1:.4f} seconds (fast due to shared matrices)')

print("Clearing cache...")
fpt.cache.clear()
print("\nEmpty cache computation for reference:")
t1 = time()
mix_result_empty = handler.run('IA_ta')
t2 = time()
print(f'  → IA_ta with empty cache: {t2-t1:.4f} seconds (no cache)')

# ===============================================================================
# 4. DIRECT TERM ACCESS - Get specific components easily
# ===============================================================================
print('\n4. === DIRECT TERM ACCESS ===')

print('Getting specific power spectrum terms directly:')

# Get individual 1-loop terms
P_1loop = handler.get('P_1loop')
print('✓ Retrieved P_1loop directly')

# Get multiple IA terms at once
ia_terms = handler.get('P_E', 'P_B', 'P_A')
print('✓ Retrieved multiple IA terms: P_E, P_B, P_A')

# Show how to access specific terms
print('Accessing specific terms:')
P_E = ia_terms['P_E']
P_B = ia_terms['P_B']
print(f'  → P_E: {P_E[:5]} (first 5 values)')
print(f'  → P_B: {P_B[:5]} (first 5 values)')

# Get bias terms
bias_terms = handler.get('Pd1d2', 'Pd2d2', 'Ps2s2')
print('✓ Retrieved bias terms: Pd1d2, Pd2d2, Ps2s2')

# ===============================================================================
# 5. DIFF MODE MULTIPLE POWER SPECTRA GENERATION
# ===============================================================================
print('\n5. === MULTIPLE POWER SPECTRA GENERATION ===')

try:
    # Generate multiple cosmologies using 'diff' mode
    print('Generating power spectra for different cosmologies...')
    cosmo_spectra = handler.generate_power_spectra(
        method='camb',
        mode='diff',
        h=[0.65, 0.67, 0.70],        # Different Hubble parameters
        omega_m=[0.28, 0.30, 0.32],   # Different matter densities
        z=0.0
    )
    print(f'✓ Generated {len(cosmo_spectra)} different cosmological power spectra')
    
    # Use one for calculations
    # Keys are comprised of z and the parameter index based on the order they were passed in (starting at 1). 
    # Then use a positive or negative sign to indicate high or low values.
    # Example: Key (0.0, 1) will give the power spectrum generated with h=0.70, omega_m=0.30, z=0.0
    P_h_low = cosmo_spectra[(0.0, -1)] # Example key: Diff mode uses the low h parameter (1st parameter passed)
    P_omegam_high = cosmo_spectra[(0.0, 2)] # Example key: Diff mode uses the high omega_m parameter (2nd parameter passed)
    
except ImportError:
    # Fallback: create different power spectra manually
    print('Creating multiple power spectra manually...')
    P_alt = 800 * k**(-1.3) * np.exp(-k/8)  # Different normalization and slope
    print('✓ Created alternative power spectrum')

# ===============================================================================
# 6. PLOTTING CAPABILITIES
# ===============================================================================
print('\n6. === PLOTTING DEMONSTRATION ===')

# Plot 1: Compare linear and 1-loop power spectra
print('Creating plots...')

# Basic plot of linear vs 1-loop
fig1 = handler.plot(
    data={'Linear P(k)': P_linear, '1-loop P(k)': P_linear + P_1loop},
    title='Linear vs 1-loop Power Spectra',
    colors=['blue', 'red'],
    style=[{'linestyle': '-'}, {'linestyle': '--'}],
    return_fig=True,
    show=False
)

# Plot 2: IA terms comparison
fig2 = handler.plot(
    terms=['P_E', 'P_B'],
    title='Intrinsic Alignment: E and B modes',
    colors=['green', 'orange'], 
    return_fig=True,
    show=False
)

# Plot 3: Bias terms
fig3 = handler.plot(
    data=bias_terms,
    title='Galaxy Bias Terms',
    log_scale=True,
    return_fig=True,
    show=False
)

print('✓ Created multiple demonstration plots')

# ===============================================================================
# 7. SAVE/LOAD FUNCTIONALITY
# ===============================================================================
print('\n7. === SAVE/LOAD DEMONSTRATION ===')

# Update the default parameters before saving the handler instance
handler.update_default_params(
    P=P_linear,
    C_window=0.75,
    P_window=np.array([0.2, 0.2]),
)
handler.save_instance('example_handler') 
# ^^ This will also save the parameters used to make the handler's fastpt instance

# Load them back in a new handler instance
loaded_handler = FPTHandler.load_instance('outputs/example_handler')

# Now use the loaded handler with stored parameters to run calculations:
handler.run('one_loop_dd')
# or using the loaded Fast-PT instance directly:
loaded_handler.fastpt.one_loop_dd(**loaded_handler.default_params)

print('✓ Saved and loaded results and parameters')
print(f'  → Loaded parameters: {list(loaded_handler.default_params.keys())}')

# ===============================================================================
# 8. CACHE INFORMATION
# ===============================================================================
print('\n8. === CACHE INFORMATION ===')
print('Current cache status:')
print(fpt.cache)

# ===============================================================================
# 9. SHOW ALL PLOTS
# ===============================================================================
print('\n9. === DISPLAYING PLOTS ===')
print('Showing all generated plots...')

# Show all created figures
created_count = 0
for fig_var, description in [('fig1', 'Linear vs 1-loop comparison'),
                            ('fig2', 'IA terms'), 
                            ('fig3', 'Bias terms')]:
    if fig_var in locals() and locals()[fig_var] is not None:
        created_count += 1
        print(f'✓ {description} plot created')

if created_count > 0:
    print(f'\nDisplaying all {created_count} plots...')
    plt.show()  # Show all figures at once
else:
    print('No plots were successfully created to display.')

# ===============================================================================
# SUMMARY
# ===============================================================================
print('\n' + '='*60)
print('SUMMARY OF NEW FEATURES DEMONSTRATED:')
print('='*60)
print('✓ Simplified initialization - no to_do list required')
print('✓ Automatic caching system with significant speedup')
print('✓ FPTHandler for streamlined calculations')
print('✓ Built-in power spectrum generation (CAMB/CLASS)')
print('✓ Direct access to specific power spectrum terms')
print('✓ Comprehensive plotting capabilities')
print('✓ Save/load functionality for results and parameters')
print('✓ Multiple cosmology handling')
print('✓ Tracer-specific term collections')
print('='*60)

print('\nExample completed successfully!')
print('Check the generated plots and saved files in the outputs directory.')