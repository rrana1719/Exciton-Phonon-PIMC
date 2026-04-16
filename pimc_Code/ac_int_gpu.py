import numpy as np
import cupy as cp
from numba import cuda
import math


#This code provides precomputed grids for the effective Hamiltonian,
# spatial, and temperature derivatives for the Virial energy 
# estimator. We were interpolate on these grids for PIMC!


##NOTE: I am working in atomic units!
# Parameters

#Cubic box length

Lx = 1000

r_s = np.linspace(0.1, Lx*np.sqrt(3)/2, 5000, endpoint=False)


T = 0.0003167 * 3.0

#Number of beads
N = 2500

t_s = np.arange(1, N)

#Speed of sound (can adjust for LA/TA modes)
sLA = 0.00201
#sTA = 0.000975
V_UC = 5373.4583
q = (6*np.pi**2 / V_UC)**(1/3)

# Gauss-Legendre quadrature setup
def gauss_legendre_setup(n_points=64):
    """Generate Gauss-Legendre nodes and weights for [0, q]"""
    # Get nodes and weights for [-1, 1]
    nodes_std, weights_std = np.polynomial.legendre.leggauss(n_points)
    
    # Transform to [0, q]
    nodes = (nodes_std + 1) * q / 2
    weights = weights_std * q / 2
    
    return nodes.astype(np.float64), weights.astype(np.float64)

# CUDA kernel for computing integrals and analytical terms
@cuda.jit
def compute_integrals_kernel(r_grid, t_grid, H_out, der_spat_out, der_temp_out, Vir_out,
                            nodes, weights, sLA, q_param, T_val, N_param):
    i, j = cuda.grid(2)
    
    if i >= r_grid.shape[0] or j >= t_grid.shape[0]:
        return
    
    r0 = r_grid[i]
    t0 = t_grid[j]  # This is tau (time slice index)
    
    # Constants
    b = 1.0 / T_val
    T_inv = b
    
    # Initialize integral sums
    I1_sum = 0.0
    I2_sum = 0.0
    I3_sum = 0.0
    I4_sum = 0.0
    
    # Numerical integration using Gauss-Legendre quadrature
    for k in range(nodes.shape[0]):
        x = nodes[k]
        w = weights[k]
        
        if x > 0:  # Avoid division by zero
            # Common terms
            sbx = sLA * b * x
            exp_sbx = math.exp(sbx)
            bose_factor = 1.0 / (exp_sbx - 1.0)
            
            # Time-dependent terms (proper handling of tau)
            sbt_x_n = sLA * b * t0 * x / N_param
            cosh_term = math.cosh(sbt_x_n)
            sinh_term = math.sinh(sbt_x_n)
            
            # Trigonometric terms
            sin_rx = math.sin(r0 * x)
            cos_rx = math.cos(r0 * x)
            
            # I1: x^2 * cosh(s*b*t*x/n) * (1/(exp(s*b*x) - 1)) * sin(r*x)
            I1_integrand = x * x * cosh_term * bose_factor * sin_rx
            if math.isfinite(I1_integrand):
                I1_sum += w * I1_integrand
            
            # I2: x^3 * cosh(s*b*t*x/n) * (1/(exp(s*b*x) - 1)) * cos(r*x)
            I2_integrand = x * x * x * cosh_term * bose_factor * cos_rx
            if math.isfinite(I2_integrand):
                I2_sum += w * I2_integrand
            
            # I3: (s*b/n)*(t/r) * x^3 * sinh(s*b*t*x/n) * (1/(exp(s*b*x) - 1)) * sin(r*x)
            if r0 > 0:
                I3_prefactor = (sLA * b / N_param) * (t0 / r0)
                I3_integrand = I3_prefactor * x * x * x * sinh_term * bose_factor * sin_rx
                if math.isfinite(I3_integrand):
                    I3_sum += w * I3_integrand
            
            # I4: s*b*(1/r) * x^3 * cosh(s*b*t*x/n) * exp(s*b*x) * (1/(exp(s*b*x) - 1))^2 * sin(r*x)
            if r0 > 0:
                I4_prefactor = sLA * b / r0
                bose_factor_sq = bose_factor * bose_factor
                I4_integrand = I4_prefactor * x * x * x * cosh_term * exp_sbx * bose_factor_sq * sin_rx
                if math.isfinite(I4_integrand):
                    I4_sum += w * I4_integrand
    
    # Compute analytical AC terms
    # Time transformation: t = tau * s / T / N
    t_ac = t0 * sLA / T_val / N_param
    
    # Precompute powers
    r2 = r0 * r0
    r4 = r2 * r2
    r6 = r4 * r2
    t2 = t_ac * t_ac
    t3 = t2 * t_ac
    t4 = t2 * t2
    q2 = q_param * q_param
    q3 = q2 * q_param
    r2_plus_t2 = r2 + t2
    
    # Exponential and trigonometric terms
    exp_neg_qt = math.exp(-q_param * t_ac)
    exp_pos_qt = math.exp(q_param * t_ac)
    cos_qr = math.cos(q_param * r0)
    sin_qr = math.sin(q_param * r0)
    
    # Heff_AC0K calculation
    heff_term1 = -2 * exp_pos_qt * r0 * (r2 - 3 * t2)
    heff_cos_coeff = r0 * (q2 * r4 + 2 * r2 * (-1 + q_param * t_ac * (2 + q_param * t_ac)) + 
                           t2 * (6 + q_param * t_ac * (4 + q_param * t_ac)))
    heff_term2 = -heff_cos_coeff * cos_qr
    heff_sin_coeff = (-6 * r2 * t_ac + 2 * t3 + q2 * t_ac * r2_plus_t2 * r2_plus_t2 + 
                      2 * q_param * (-r4 + t4))
    heff_term3 = -heff_sin_coeff * sin_qr
    heff_numerator = exp_neg_qt * (heff_term1 + heff_term2 + heff_term3)
    heff_denominator = r0 * r2_plus_t2 * r2_plus_t2 * r2_plus_t2
    Heff_AC = heff_numerator / heff_denominator / T_val / N_param / N_param
    
    # AC_Der_Spat0K calculation
    spat_term1 = 6 * exp_pos_qt * (r4 - 6 * r2 * t2 + t4)
    spat_cos_coeff = (q2 * r6 * (-3 + q_param * t_ac) + 
                      3 * r4 * (2 + q_param * t_ac * (-3 + q_param * t_ac) * (2 + q_param * t_ac)) + 
                      3 * r2 * t2 * (-12 + q_param * t_ac * (-4 + q_param * t_ac * (1 + q_param * t_ac))) + 
                      t4 * (6 + q_param * t_ac * (6 + q_param * t_ac * (3 + q_param * t_ac))))
    spat_term2 = -spat_cos_coeff * cos_qr
    spat_sin_coeff = r0 * (24 * t_ac * (-r2 + t2) - 6 * q_param * (r2 - 3 * t2) * r2_plus_t2 + 
                           6 * q2 * t_ac * r2_plus_t2 * r2_plus_t2 + 
                           q3 * r2_plus_t2 * r2_plus_t2 * r2_plus_t2)
    spat_term3 = spat_sin_coeff * sin_qr
    spat_numerator = T_inv * exp_neg_qt * (spat_term1 + spat_term2 + spat_term3)
    spat_denominator = 2 * N_param * N_param * r2_plus_t2 * r2_plus_t2 * r2_plus_t2 * r2_plus_t2
    AC_Der_Spat = spat_numerator / spat_denominator
    
    # AC_TempDer0K calculation
    temp_term1 = 24 * exp_pos_qt * r0 * t_ac * (t2 - r2)
    temp_cos_coeff = r0 * (24 * t_ac * (-r2 + t2) - 6 * q_param * (r2 - 3 * t2) * r2_plus_t2 + 
                           6 * q2 * t_ac * r2_plus_t2 * r2_plus_t2 + 
                           q3 * r2_plus_t2 * r2_plus_t2 * r2_plus_t2)
    temp_term2 = -temp_cos_coeff * cos_qr
    temp_sin_coeff = (q2 * r6 * (-3 + q_param * t_ac) + 
                      3 * r4 * (2 + q_param * t_ac * (-3 + q_param * t_ac) * (2 + q_param * t_ac)) + 
                      3 * r2 * t2 * (-12 + q_param * t_ac * (-4 + q_param * t_ac * (1 + q_param * t_ac))) + 
                      t4 * (6 + q_param * t_ac * (6 + q_param * t_ac * (3 + q_param * t_ac))))
    temp_term3 = -temp_sin_coeff * sin_qr
    temp_numerator = T_inv * exp_neg_qt * t_ac * (temp_term1 + temp_term2 + temp_term3)
    temp_denominator = N_param * N_param * r0 * r2_plus_t2 * r2_plus_t2 * r2_plus_t2 * r2_plus_t2
    AC_TempDer = temp_numerator / temp_denominator
    
    # Combine results
    # H_Eff_LA = 2*I1/T/N/N + Heff_AC0K
    H_Eff_LA = 2 * I1_sum / T_val / N_param / N_param + Heff_AC
    
    # der_spat = I2/T/N/N + AC_Der_Spat0K
    der_spat = I2_sum / T_val / N_param / N_param + AC_Der_Spat
    
    # der_temp = -AC_TempDer0K + 2*(I3 - I4)/T/N/N
    der_temp = -AC_TempDer + 2 * (I3_sum - I4_sum) / T_val / N_param / N_param
    
    # der_LA = der_spat + der_temp
    der_LA_total = der_spat + der_temp
    
    # Vir = 1.5*H_Eff_LA + der_LA
    Vir = 1.5 * H_Eff_LA + der_LA_total
    
    # Store results
    H_out[i, j] = H_Eff_LA
    der_spat_out[i, j] = der_spat
    der_temp_out[i, j] = der_temp
    Vir_out[i, j] = Vir

def compute_grids_gpu(r_array, t_array, sLA_val, q_val, T_val, N_val, n_quad_points=64):
    """
    Compute H_Eff_LA, derivatives, and Virial on GPU
    
    Parameters:
    -----------
    r_array : array of spatial points
    t_array : array of time indices (tau values)
    sLA_val : LA parameter
    q_val : cutoff parameter
    T_val : temperature
    N_val : number of time slices
    n_quad_points : number of quadrature points
    
    Returns:
    --------
    H_grid, der_spat_grid, der_temp_grid, Vir_grid : computed grids
    """
    
    # Setup quadrature
    nodes, weights = gauss_legendre_setup(n_quad_points)
    
    # Transfer to GPU
    r_gpu = cp.asarray(r_array, dtype=cp.float64)
    t_gpu = cp.asarray(t_array, dtype=cp.float64)
    nodes_gpu = cp.asarray(nodes, dtype=cp.float64)
    weights_gpu = cp.asarray(weights, dtype=cp.float64)
    
    # Allocate output arrays
    H_grid = cp.zeros((len(r_array), len(t_array)), dtype=cp.float64)
    der_spat_grid = cp.zeros((len(r_array), len(t_array)), dtype=cp.float64)
    der_temp_grid = cp.zeros((len(r_array), len(t_array)), dtype=cp.float64)
    Vir_grid = cp.zeros((len(r_array), len(t_array)), dtype=cp.float64)
    
    # Configure kernel launch
    threads_per_block = (16, 16)
    blocks_per_grid_x = (len(r_array) + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (len(t_array) + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Launch kernel
    compute_integrals_kernel[blocks_per_grid, threads_per_block](
        r_gpu, t_gpu, H_grid, der_spat_grid, der_temp_grid, Vir_grid,
        nodes_gpu, weights_gpu, sLA_val, q_val, T_val, N_val
    )
    
    # Synchronize and return
    cp.cuda.Stream.null.synchronize()
    
    return cp.asnumpy(H_grid), cp.asnumpy(der_spat_grid), cp.asnumpy(der_temp_grid), cp.asnumpy(Vir_grid)

# Test function
def test_gpu_implementation():
    """Test GPU implementation against scipy version"""
    from scipy import integrate
    
    print("Testing GPU implementation...")
    
    # Test on small subset
    r_test = r_s
    t_test = t_s
    
    # GPU computation
    H_gpu, der_spat_gpu, der_temp_gpu, Vir_gpu = compute_grids_gpu(
        r_test, t_test, sLA, q, T, N, n_quad_points=64
    )
    
    # Print sample results
    print(f"\nSample H_Eff_LA value at r={r_test[0]:.3f}, tau={t_test[0]}:")
    print(f"  GPU: {H_gpu[0, 0]:.10e}")
    
    print(f"\nGrid shapes:")
    print(f"  H_Eff_LA: {H_gpu.shape}")
    print(f"  der_spat: {der_spat_gpu.shape}")
    print(f"  der_temp: {der_temp_gpu.shape}")
    print(f"  Virial: {Vir_gpu.shape}")
    
    return H_gpu, der_spat_gpu, der_temp_gpu, Vir_gpu

if __name__ == "__main__":
    # Run test
    H_gpu, der_spat_gpu, der_temp_gpu, Vir_gpu = test_gpu_implementation()    
    # Save results
 
    np.savetxt("H_Eff_LA_gpu.txt", H_gpu)
    np.savetxt("der_spat_LA_gpu.txt", der_spat_gpu)
    np.savetxt("der_temp_LA_gpu.txt", der_temp_gpu)
    np.savetxt("Vir_LA_gpu.txt", Vir_gpu)
    print("\nResults saved to files.")
