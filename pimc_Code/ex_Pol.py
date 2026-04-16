import numba, sys
import random
import os
import numpy as np
import h5py

r_s = np.linspace(0.1, 1000*np.sqrt(3)/2, 5000, endpoint=False)
t_s = np.arange(1, 2500)
r_min = r_s[0]
dr_inv = 1.0 / (r_s[1] - r_s[0])


#We use the Virial energy estimator and interpolate on grids for the acoustic modes.
#This is because the effective action for these modes cannot be analytically transformed
#into real space.

#See the interpolator code in the same directory.

# Load the grids and ensure they're in the right format
LAg = np.loadtxt(sys.argv[2])
TAg = np.loadtxt(sys.argv[3])
TA2g = np.loadtxt(sys.argv[4])

vLAg = np.loadtxt(sys.argv[5])
vTAg = np.loadtxt(sys.argv[6])
vTA2g = np.loadtxt(sys.argv[7])

# Reshape if needed
if LAg.ndim == 1:
    LAg = LAg.reshape(len(r_s), len(t_s))
if TAg.ndim == 1:
    TAg = TAg.reshape(len(r_s), len(t_s))
if TA2g.ndim == 1:
    TA2g = TA2g.reshape(len(r_s), len(t_s))
if vLAg.ndim == 1:
    vLAg = vLAg.reshape(len(r_s), len(t_s))
if vTAg.ndim == 1:
    vTAg = vTAg.reshape(len(r_s), len(t_s))
if vTA2g.ndim == 1:
    vTA2g = vTA2g.reshape(len(r_s), len(t_s))

# Make them contiguous for better performance
LAg = np.ascontiguousarray(LAg, dtype=np.float64)
TAg = np.ascontiguousarray(TAg, dtype=np.float64)
TA2g = np.ascontiguousarray(TA2g, dtype=np.float64)
vLAg = np.ascontiguousarray(vLAg, dtype=np.float64)
vTAg = np.ascontiguousarray(vTAg, dtype=np.float64)
vTA2g = np.ascontiguousarray(vTA2g, dtype=np.float64)


@numba.jit(nopython=True, fastmath=True, cache=True)
def find_r_index_fast(r_val, nr):
    """Fast index finding for uniform grid"""
    # Direct calculation for uniform grid
    idx = int((r_val - r_min) * dr_inv)
    return max(0, min(idx, nr - 2))


@numba.jit(nopython=True, fastmath=True, cache=True)
def bilin(r_s, t_s, grid_values, r_val, t_val):
    """
    Bilinear interpolation with:
    - Direct indexing for integer t values
    - Fast index calculation for uniform r grid
    - Minimized memory accesses
    - Fastmath enabled
    """
    nr = len(r_s)
    nt = len(t_s)
    
    # Direct index for t (since t_s = [1, 2, 3, ...])
    # Much faster than binary search
    t_idx = int(t_val) - 1
    if t_idx < 0:
        t_idx = 0
    elif t_idx >= nt - 1:
        t_idx = nt - 2
    
    # Fast index finding for uniform r grid
    r_idx = find_r_index_fast(r_val, nr)
    
    # Calculate interpolation fractions
    # For uniform grids, we can optimize this
    r_frac = (r_val - r_s[r_idx]) * dr_inv
    r_frac = max(0.0, min(1.0, r_frac))
    
    # For integer t_s, the fraction calculation is simple
    t_frac = t_val - t_s[t_idx]  # Since dt = 1 for integer spacing
    t_frac = max(0.0, min(1.0, t_frac))
    
    # Single memory access to get all 4 corners at once
    # This is more cache-friendly
    v00 = grid_values[r_idx, t_idx]
    v10 = grid_values[r_idx + 1, t_idx]
    v01 = grid_values[r_idx, t_idx + 1]
    v11 = grid_values[r_idx + 1, t_idx + 1]
    
    # Optimized bilinear interpolation
    return v00 + r_frac * (v10 - v00) + t_frac * ((v01 - v00) + r_frac * (v00 - v10 - v01 + v11))


#Coulomb Potential
@numba.jit(nopython=True)
def C(xn0,yn0,zn0,hn0,hn1,hn2):
    xu = (xn0-hn0) - Lx*np.round((xn0-hn0)/Lx)
    yu = (yn0-hn1) - Ly*np.round((yn0-hn1)/Ly)
    zu = (zn0-hn2) - Lz*np.round((zn0-hn2)/Lz)
    r = np.sqrt(xu**2 + yu**2 + zu**2)
    #potS = interpolate.interp1d(r_s, potSums)
    #derivS = interpolate.interp1d(r_s, derivSums)
    return( -1/ep/(np.sqrt(r**2 + ae**2))    )
    
#Virial Estimator for Coulomb Potential
@numba.jit(nopython=True)
def Est(xn0,yn0,zn0,hn0,hn1,hn2):
    xu = (xn0-hn0) - Lx*np.round((xn0-hn0)/Lx)
    yu = (yn0-hn1) - Ly*np.round((yn0-hn1)/Ly)
    zu = (zn0-hn2) - Lz*np.round((zn0-hn2)/Lz)
    r = np.sqrt(xu**2 + yu**2 + zu**2)
    return( -1/ep/(np.sqrt(r**2 + ae**2)) + 0.5*(r**2)/ep/((r**2 + ae**2)**1.5) )

#E-Ph Effective Hamiltonian for LO Mode
@numba.jit(nopython=True)
def Heff_LO(r,t,gam_e,gam_h,w):
    chi = np.exp(-w*t/T/N) + 2*np.cosh(w*t/T/N)*(1/(np.exp(w/T)-1))
    y = gam_e*gam_h
    return( y*chi*(1/(T*(N**2)))*(1/r)   )
    

#E-Ph Self-Energy Virial Estimator for LO Mode
@numba.jit(nopython=True)
def Est_self_LO(x, yFr, w):
    # Pre-compute all constants outside loops
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz

    yFr_sq = yFr * yFr
    const_factor = 1.0 / (T * N * N)  # Common factor: 1/(T*N^2)
    w_over_TN = w / (T * N)
    w_over_T = w / T
    factor_minus_1p5 = -1.5  # Pre-compute (-2 + 0.5)
    
    # Pre-compute exponential terms for derFT
    exp_w_T = np.exp(w_over_T)
    exp_w_T_minus_1 = exp_w_T - 1.0
    inv_exp_w_T_minus_1 = 1.0 / exp_w_T_minus_1
    inv_exp_w_T_minus_1_sq = inv_exp_w_T_minus_1 * inv_exp_w_T_minus_1
    exp_w_T_factor = exp_w_T * inv_exp_w_T_minus_1_sq

    total_energ = 0.0

    # Parallel loop over first particle index
    for i in range(N):
        # Cache particle i coordinates
        xi = x[i, 0]
        yi = x[i, 1]
        zi = x[i, 2]

        # Local energy accumulator for this thread
        local_energ = 0.0

        for k in range(i+1, N):
            # Direct array access for particle k
            dx = xi - x[k, 0]
            dy = yi - x[k, 1]
            dz = zi - x[k, 2]

            # Apply minimum image convention (faster with multiplication)
            dx -= Lx * np.round(dx * inv_Lx)
            dy -= Ly * np.round(dy * inv_Ly)
            dz -= Lz * np.round(dz * inv_Lz)

            # Distance calculation
            rmag = np.sqrt(dx*dx + dy*dy + dz*dz)

            # Time slice difference (no abs needed since k > i)
            tau = k - i

            # Pre-compute frequently used terms
            tau_w_factor = tau * w_over_TN
            inv_rmag = 1.0 / rmag
            yFr_sq_const_inv_rmag = yFr_sq * const_factor * inv_rmag

            # Compute exponential and hyperbolic terms
            exp_neg_tau_w = np.exp(-tau_w_factor)
            sinh_tau_w = np.sinh(tau_w_factor)
            cosh_tau_w = np.cosh(tau_w_factor)

            # Compute der0K term
            der0K = -tau_w_factor * yFr_sq_const_inv_rmag * exp_neg_tau_w
            
            # Compute derFT term
            derFT_term1 = 2.0 * tau_w_factor * yFr_sq_const_inv_rmag * sinh_tau_w * inv_exp_w_T_minus_1
            derFT_term2 = -2.0 * w_over_T * yFr_sq_const_inv_rmag * cosh_tau_w * exp_w_T_factor
            derFT = derFT_term1 + derFT_term2

            # Energy accumulation
            local_energ += factor_minus_1p5 * Heff_LO(rmag, tau, yFr, yFr, w) - der0K - derFT

        total_energ += local_energ

    return 2.0 * total_energ

#E-Ph Cross-Energy Virial Estimator for LO Mode
@numba.jit(nopython=True)
def Est_cross_LO(x, h, yFr_e, yFr_h, w):
    # Pre-compute all constants outside loops
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz

    yFr_prod = yFr_e * yFr_h
    const_factor = 1.0 / (T * N * N)
    w_over_TN = w / (T * N)
    w_over_T = w / T
    factor_1p5 = 1.5  # Pre-compute (2 - 0.5)

    # Pre-compute exponential terms for derFT
    exp_w_T = np.exp(w_over_T)
    exp_w_T_minus_1 = exp_w_T - 1.0
    inv_exp_w_T_minus_1 = 1.0 / exp_w_T_minus_1
    inv_exp_w_T_minus_1_sq = inv_exp_w_T_minus_1 * inv_exp_w_T_minus_1
    exp_w_T_factor = exp_w_T * inv_exp_w_T_minus_1_sq

    total_energ = 0.0

    # Parallel loop over first particle index
    for i in range(N):
        # Cache particle i coordinates from x array
        xi = x[i, 0]
        yi = x[i, 1]
        zi = x[i, 2]

        # Local energy accumulator for this thread
        local_energ = 0.0

        for k in range(N):
            if(k!=-1):
                # Direct array access for particle k from h array
                dx = xi - h[k, 0]
                dy = yi - h[k, 1]
                dz = zi - h[k, 2]

                # Apply minimum image convention
                dx -= Lx * np.round(dx * inv_Lx)
                dy -= Ly * np.round(dy * inv_Ly)
                dz -= Lz * np.round(dz * inv_Lz)

                # Distance calculation
                rmag = np.sqrt(dx*dx + dy*dy + dz*dz)

                # Time slice difference
                tau = abs(k - i)

                # Pre-compute frequently used terms
                tau_w_factor = tau * w_over_TN
                inv_rmag = 1.0 / rmag
                yFr_prod_const_inv_rmag = yFr_prod * const_factor * inv_rmag

                # Compute exponential and hyperbolic terms
                exp_neg_tau_w = np.exp(-tau_w_factor)
                sinh_tau_w = np.sinh(tau_w_factor)
                cosh_tau_w = np.cosh(tau_w_factor)

                # Compute der0K term
                der0K = -tau_w_factor * yFr_prod_const_inv_rmag * exp_neg_tau_w

                # Compute derFT term
                derFT_term1 = 2.0 * tau_w_factor * yFr_prod_const_inv_rmag * sinh_tau_w * inv_exp_w_T_minus_1
                derFT_term2 = -2.0 * w_over_T * yFr_prod_const_inv_rmag * cosh_tau_w * exp_w_T_factor
                derFT = derFT_term1 + derFT_term2

                # Energy accumulation (note: +der0K and +derFT for cross term)
                local_energ += factor_1p5 * Heff_LO(rmag, tau, yFr_e, yFr_h, w) + der0K + derFT

        total_energ += local_energ

    return 1.0 * total_energ

#E-Ph Effective Hamiltonian for TO Mode
@numba.jit(nopython=True)
def Heff_TO_0(r,t,gam_e,gam_h,w):
    chi = np.exp(-w*t/T/N) + 2*np.cosh(w*t/T/N)*(1/(np.exp(w/T)-1))
    y = gam_e*gam_h
    return (chi * y * (-q * r * np.cos(q * r) + np.sin(q * r)) / r**3)*(1/T/N/N)
    
#Helper Function for Energy Estimators for TO mode
@numba.jit(nopython=True)
def derT_TO_0(r, t, gam_e, gam_h, w):
    # Pre-compute constants
    w_over_T = w / T
    w_over_TN = w / (T * N)
    tau_w_factor = t * w_over_TN
    
    # Exponential and hyperbolic functions
    exp_neg_tau_w = np.exp(-tau_w_factor)
    exp_w_T = np.exp(w_over_T)
    exp_w_T_minus_1 = exp_w_T - 1.0
    inv_exp_w_T_minus_1 = 1.0 / exp_w_T_minus_1
    inv_exp_w_T_minus_1_sq = inv_exp_w_T_minus_1 * inv_exp_w_T_minus_1
    
    sinh_tau_w = np.sinh(tau_w_factor)
    cosh_tau_w = np.cosh(tau_w_factor)
    
    # Spatial term
    q_r = q * r
    spatial = (-q * r * np.cos(q_r) + np.sin(q_r)) / r**3
    
    # Product of coupling constants
    y = gam_e * gam_h
    
    # der0K: derivative of exp(-w*t/T/N) term
    # β²/N² * d/dβ[exp(-β*w*t/N)] = -(w*t/T²/N³) * exp(-w*t/T/N)
    der0K = -tau_w_factor * (1.0 / (T * N * N)) * y * spatial * exp_neg_tau_w
    
    # derFT: derivative of 2*cosh/(exp-1) term
    # Term 1: 2 * (w*t/T/N) * sinh(w*t/T/N) / (exp(w/T)-1)
    term1 = 2.0 * tau_w_factor * sinh_tau_w * inv_exp_w_T_minus_1
    
    # Term 2: -2 * (w/T) * exp(w/T) * cosh(w*t/T/N) / (exp(w/T)-1)²
    term2 = -2.0 * w_over_T * exp_w_T * cosh_tau_w * inv_exp_w_T_minus_1_sq
    
    derFT = (1.0 / (T * N * N)) * y * spatial * (term1 + term2)
    
    # Total derivative
    return der0K + derFT

#E-Ph Self-Energy Virial Estimator for TO Mode
@numba.jit(nopython=True)
def Est_self_TO_0(x, yFr, w):
    # Pre-compute all constants outside loops
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz
    
    w_over_TN = w / (T * N)
    inv_TN = 1.0 / (T * N)
    factor_minus_0p5 = -0.5
    yFr_sq = yFr * yFr
    yFr_sq_factor = yFr_sq * 0.5 * inv_TN / N
    
    q_sq = q * q
    
    exp_w_T = np.exp(w / T)
    inv_exp_w_T_minus_1 = 1.0 / (exp_w_T - 1.0)
    
    total_energ = 0.0
    
    for i in range(N):
        xi = x[i, 0]
        yi = x[i, 1]
        zi = x[i, 2]
        
        local_energ = 0.0
        
        for k in range(i+1, N):
            dx = xi - x[k, 0]
            dy = yi - x[k, 1]
            dz = zi - x[k, 2]
            
            dx -= Lx * np.round(dx * inv_Lx)
            dy -= Ly * np.round(dy * inv_Ly)
            dz -= Lz * np.round(dz * inv_Lz)
            
            r_sq = dx*dx + dy*dy + dz*dz
            rmag = np.sqrt(r_sq)
            
            tau = k - i
            
            tau_w_factor = tau * w_over_TN
            exp_factor = np.exp(-tau_w_factor)
            
            q_rmag = q * rmag
            sin_qr = np.sin(q_rmag)
            
            chi = exp_factor + 2.0 * np.cosh(tau_w_factor) * inv_exp_w_T_minus_1
            spat = chi * q_sq * sin_qr / rmag
            
            local_energ += (factor_minus_0p5 * Heff_TO_0(rmag, tau, yFr, yFr, w)
                          - yFr_sq_factor * spat
                          - derT_TO_0(rmag, tau, yFr, yFr, w))
        
        total_energ += local_energ
    
    return 2.0 * total_energ


#E-Ph Cross-Energy Virial Estimator for TO Mode
@numba.jit(nopython=True)
def Est_cross_TO_0(x, h, yFr_e, yFr_h, w):
    # Pre-compute all constants outside loops
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz
    
    w_over_TN = w / (T * N)
    inv_TN = 1.0 / (T * N)
    factor_0p5 = 0.5
    yFr_prod = yFr_e * yFr_h
    yFr_prod_factor = yFr_prod * 0.5 * inv_TN / N
    
    q_sq = q * q
    
    exp_w_T = np.exp(w / T)
    inv_exp_w_T_minus_1 = 1.0 / (exp_w_T - 1.0)
    
    total_energ = 0.0
    
    for i in range(N):
        xi = x[i, 0]
        yi = x[i, 1]
        zi = x[i, 2]
        
        local_energ = 0.0
        
        for k in range(N):
            dx = xi - h[k, 0]
            dy = yi - h[k, 1]
            dz = zi - h[k, 2]
            
            dx -= Lx * np.round(dx * inv_Lx)
            dy -= Ly * np.round(dy * inv_Ly)
            dz -= Lz * np.round(dz * inv_Lz)
            
            r_sq = dx*dx + dy*dy + dz*dz
            rmag = np.sqrt(r_sq)
            
            tau = abs(k - i)
            
            tau_w_factor = tau * w_over_TN
            exp_factor = np.exp(-tau_w_factor)
            
            q_rmag = q * rmag
            sin_qr = np.sin(q_rmag)
            
            chi = exp_factor + 2.0 * np.cosh(tau_w_factor) * inv_exp_w_T_minus_1
            spat = chi * q_sq * sin_qr / rmag
            
            local_energ += (factor_0p5 * Heff_TO_0(rmag, tau, yFr_e, yFr_h, w)
                          + yFr_prod_factor * spat
                          + derT_TO_0(rmag, tau, yFr_e, yFr_h, w))
        
        total_energ += local_energ
    
    return 1.0 * total_energ


#E-Ph Effective Hamiltonian for Acoustic Mode
@numba.jit(nopython=True)
def Heff_AC(r,tau,gam_e,gam_h,s):
    y = gam_e * gam_h
    if (s == sLA):
        return y*bilin(r_s,t_s,LAg,r,abs(tau))
    elif(s ==sTA1):
        return y*bilin(r_s,t_s,TAg,r,abs(tau))
    elif(s ==sTA2):
        return y*bilin(r_s,t_s,TA2g,r,abs(tau))

#E-Ph Self-Energy Virial Estimator for Acoustic Mode
@numba.jit(nopython=True)
def Est_self_AC(x, yAc, s):
    # Pre-compute all constants outside loops
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz
    
    
    total_energ = 0.0
    
    if(s==sLA):
    # Parallel loop over first particle index
        for i in range(N):
        # Cache particle i coordinates
            xi = x[i, 0]
            yi = x[i, 1]
            zi = x[i, 2]
        
        # Local energy accumulator for this thread
            local_energ = 0.0
        
            for k in range(i+1, N):
            # Direct array access for particle k
                dx = xi - x[k, 0]
                dy = yi - x[k, 1]
                dz = zi - x[k, 2]
            
            # Apply minimum image convention (faster with multiplication)
                dx -= Lx * np.round(dx * inv_Lx)
                dy -= Ly * np.round(dy * inv_Ly)
                dz -= Lz * np.round(dz * inv_Lz)
            
            # Distance calculation
                rmag = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Time slice difference (no abs needed since k > i)
                tau = abs(k - i)
            
            # Energy accumulation with all AC terms
                local_energ -= bilin(r_s,t_s,vLAg,rmag,tau)
        
            total_energ += local_energ
    
        return 2.0 *yAc*yAc*total_energ

    elif (s==sTA1):
    # Parallel loop over first particle index
        for i in range(N):
        # Cache particle i coordinates
            xi = x[i, 0]
            yi = x[i, 1]
            zi = x[i, 2]

        # Local energy accumulator for this thread
            local_energ = 0.0

            for k in range(i+1, N):
            # Direct array access for particle k
                dx = xi - x[k, 0]
                dy = yi - x[k, 1]
                dz = zi - x[k, 2]

            # Apply minimum image convention (faster with multiplication)
                dx -= Lx * np.round(dx * inv_Lx)
                dy -= Ly * np.round(dy * inv_Ly)
                dz -= Lz * np.round(dz * inv_Lz)

            # Distance calculation
                rmag = np.sqrt(dx*dx + dy*dy + dz*dz)

            # Time slice difference (no abs needed since k > i)
                tau = abs(k - i)

            # Energy accumulation with all AC terms
                local_energ -= bilin(r_s,t_s,vTAg,rmag,tau)

            total_energ += local_energ

        return 2.0 *yAc*yAc*total_energ
    
    elif (s==sTA2):
    # Parallel loop over first particle index
        for i in range(N):
        # Cache particle i coordinates
            xi = x[i, 0]
            yi = x[i, 1]
            zi = x[i, 2]

        # Local energy accumulator for this thread
            local_energ = 0.0

            for k in range(i+1, N):
            # Direct array access for particle k
                dx = xi - x[k, 0]
                dy = yi - x[k, 1]
                dz = zi - x[k, 2]

            # Apply minimum image convention (faster with multiplication)
                dx -= Lx * np.round(dx * inv_Lx)
                dy -= Ly * np.round(dy * inv_Ly)
                dz -= Lz * np.round(dz * inv_Lz)

            # Distance calculation
                rmag = np.sqrt(dx*dx + dy*dy + dz*dz)

            # Time slice difference (no abs needed since k > i)
                tau = abs(k - i)

            # Energy accumulation with all AC terms
                local_energ -= bilin(r_s,t_s,vTA2g,rmag,tau)

            total_energ += local_energ

        return 2.0 *yAc*yAc*total_energ

#E-Ph Cross-Energy Virial Estimator for LO Mode
@numba.jit(nopython=True)
def Est_cross_AC(x, h, yAc_e, yAc_h, s):
    # Pre-compute all constants outside loops
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz
    
    
    total_energ = 0.0
    

    if(s==sLA):
    # Parallel loop over first particle index
        for i in range(N):
        # Cache particle i coordinates from x array
            xi = x[i, 0]
            yi = x[i, 1]
            zi = x[i, 2]
        
        # Local energy accumulator for this thread
            local_energ = 0.0
        
            for k in range(N):
                if(i!=-1):
            # Direct array access for particle k from h array
                    dx = xi - h[k, 0]
                    dy = yi - h[k, 1]
                    dz = zi - h[k, 2]
            
            # Apply minimum image convention
                    dx -= Lx * np.round(dx * inv_Lx)
                    dy -= Ly * np.round(dy * inv_Ly)
                    dz -= Lz * np.round(dz * inv_Lz)
            
            # Distance calculation
                    rmag = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Time slice difference (no abs needed since k > i)
                    tau = abs(k - i)
            
            # Energy accumulation with all AC terms
                    local_energ += bilin(r_s,t_s,vLAg,rmag,tau)
        
            total_energ += local_energ
    
        return 1.0 *yAc_e*yAc_h* total_energ

    elif(s==sTA1):
    # Parallel loop over first particle index
        for i in range(N):
        # Cache particle i coordinates from x array
            xi = x[i, 0]
            yi = x[i, 1]
            zi = x[i, 2]

        # Local energy accumulator for this thread
            local_energ = 0.0

            for k in range(N):
                if(i!=-1):
            # Direct array access for particle k from h array
                    dx = xi - h[k, 0]
                    dy = yi - h[k, 1]
                    dz = zi - h[k, 2]

            # Apply minimum image convention
                    dx -= Lx * np.round(dx * inv_Lx)
                    dy -= Ly * np.round(dy * inv_Ly)
                    dz -= Lz * np.round(dz * inv_Lz)

            # Distance calculation
                    rmag = np.sqrt(dx*dx + dy*dy + dz*dz)

            # Time slice difference (no abs needed since k > i)
                    tau = abs(k - i)

            # Energy accumulation with all AC terms
                    local_energ += bilin(r_s,t_s,vTAg,rmag,tau)

            total_energ += local_energ

        return 1.0 *yAc_e*yAc_h* total_energ

    elif(s==sTA2):
    # Parallel loop over first particle index
        for i in range(N):
        # Cache particle i coordinates from x array
            xi = x[i, 0]
            yi = x[i, 1]
            zi = x[i, 2]

        # Local energy accumulator for this thread
            local_energ = 0.0

            for k in range(N):
                if(i!=-1):
            # Direct array access for particle k from h array
                    dx = xi - h[k, 0]
                    dy = yi - h[k, 1]
                    dz = zi - h[k, 2]

            # Apply minimum image convention
                    dx -= Lx * np.round(dx * inv_Lx)
                    dy -= Ly * np.round(dy * inv_Ly)
                    dz -= Lz * np.round(dz * inv_Lz)

            # Distance calculation
                    rmag = np.sqrt(dx*dx + dy*dy + dz*dz)

            # Time slice difference (no abs needed since k > i)
                    tau = abs(k - i)

            # Energy accumulation with all AC terms
                    local_energ += bilin(r_s,t_s,vTA2g,rmag,tau)

            total_energ += local_energ

        return 1.0 *yAc_e*yAc_h* total_energ



#Monte Carlo Function
@numba.jit(nopython=True)
def Monte_Carlo(x, h, T, count_e, count_h, count_eC, count_hC, se, sh, seC, shC):
    #You can use these for COM moves, but staging is pretty efficient
    #delta_e = 3.6
    #delta_h = 3.6
    
    track_e = count_e
    track_h = count_h
    track_COMe = count_eC
    track_COMh = count_hC
    
    tau = 1.0 / (N * T)
    
    # Electron moves
    n_steps_e = int(np.ceil(N / (je - 1)))
    for st in range(n_steps_e):
        nn = np.random.randint(0, N)
        j = je
        
        # Must copy exactly as original
        x_N = x.copy()
        se = se + 1
        
        for k in range(j - 1):
            mk = mass_e * (j - k) / (j - (k + 1))
            sqrt_tau_over_mk = np.sqrt(tau / mk)
            
            # Exactly as in original
            xlow = x_N[(nn+k+1)%N, 0] - x_N[(nn+k)%N, 0]
            xlow = xlow - Lx * np.round(xlow / Lx)
            xlow = x_N[(nn+k+1)%N, 0] - xlow
            
            xhigh = x_N[(nn+j)%N, 0] - x_N[(nn+k+1)%N, 0]
            xhigh = xhigh - Lx * np.round(xhigh / Lx)
            xhigh = x_N[(nn+k+1)%N, 0] + xhigh
            
            ylow = x_N[(nn+k+1)%N, 1] - x_N[(nn+k)%N, 1]
            ylow = ylow - Ly * np.round(ylow / Ly)
            ylow = x_N[(nn+k+1)%N, 1] - ylow
            
            yhigh = x_N[(nn+j)%N, 1] - x_N[(nn+k+1)%N, 1]
            yhigh = yhigh - Ly * np.round(yhigh / Ly)
            yhigh = x_N[(nn+k+1)%N, 1] + yhigh
            
            zlow = x_N[(nn+k+1)%N, 2] - x_N[(nn+k)%N, 2]
            zlow = zlow - Lz * np.round(zlow / Lz)
            zlow = x_N[(nn+k+1)%N, 2] - zlow
            
            zhigh = x_N[(nn+j)%N, 2] - x_N[(nn+k+1)%N, 2]
            zhigh = zhigh - Lz * np.round(zhigh / Lz)
            zhigh = x_N[(nn+k+1)%N, 2] + zhigh
            
            ux = (xhigh + xlow * (j - (k + 1))) / (j - k)
            uy = (yhigh + ylow * (j - (k + 1))) / (j - k)
            uz = (zhigh + zlow * (j - (k + 1))) / (j - k)
            
            ux = ux + random.gauss(0, 1) * sqrt_tau_over_mk
            uy = uy + random.gauss(0, 1) * sqrt_tau_over_mk
            uz = uz + random.gauss(0, 1) * sqrt_tau_over_mk
            
            x_N[(nn+k+1)%N, 0] = ux - Lx * np.floor(ux / Lx)
            x_N[(nn+k+1)%N, 1] = uy - Ly * np.floor(uy / Ly)
            x_N[(nn+k+1)%N, 2] = uz - Lz * np.floor(uz / Lz)
        
        energy = MetropolisB_optimized(x, x_N, N, T, mass_e, nn, j, h,
                                       yLa_e_1, yLO_e_1,yLO_e_2,yLO_e_3, yTa_e_1,yTa_e_2, yTo1_e_0, yTo1_e_1, yTo1_10_e,yTo2_e_0, yTo2_e_1, yTo2_10_e)
        
        if np.random.random() <= np.exp(-energy / T):
            x = x_N.copy()
            track_e = track_e + 1
    
    # Hole moves - exactly parallel to electron moves
    n_steps_h = int(np.ceil(N / (jh - 1)))
    for st in range(n_steps_h):
        nn = np.random.randint(0, N)
        j = jh
        
        h_N = h.copy()
        sh = sh + 1
        
        for k in range(j - 1):
            mk = mass_h * (j - k) / (j - (k + 1))
            sqrt_tau_over_mk = np.sqrt(tau / mk)
            
            xlow = h_N[(nn+k+1)%N, 0] - h_N[(nn+k)%N, 0]
            xlow = xlow - Lx * np.round(xlow / Lx)
            xlow = h_N[(nn+k+1)%N, 0] - xlow
            
            xhigh = h_N[(nn+j)%N, 0] - h_N[(nn+k+1)%N, 0]
            xhigh = xhigh - Lx * np.round(xhigh / Lx)
            xhigh = h_N[(nn+k+1)%N, 0] + xhigh
            
            ylow = h_N[(nn+k+1)%N, 1] - h_N[(nn+k)%N, 1]
            ylow = ylow - Ly * np.round(ylow / Ly)
            ylow = h_N[(nn+k+1)%N, 1] - ylow
            
            yhigh = h_N[(nn+j)%N, 1] - h_N[(nn+k+1)%N, 1]
            yhigh = yhigh - Ly * np.round(yhigh / Ly)
            yhigh = h_N[(nn+k+1)%N, 1] + yhigh
            
            zlow = h_N[(nn+k+1)%N, 2] - h_N[(nn+k)%N, 2]
            zlow = zlow - Lz * np.round(zlow / Lz)
            zlow = h_N[(nn+k+1)%N, 2] - zlow
            
            zhigh = h_N[(nn+j)%N, 2] - h_N[(nn+k+1)%N, 2]
            zhigh = zhigh - Lz * np.round(zhigh / Lz)
            zhigh = h_N[(nn+k+1)%N, 2] + zhigh
            
            ux = (xhigh + xlow * (j - (k + 1))) / (j - k)
            uy = (yhigh + ylow * (j - (k + 1))) / (j - k)
            uz = (zhigh + zlow * (j - (k + 1))) / (j - k)
            
            ux = ux + random.gauss(0, 1) * sqrt_tau_over_mk
            uy = uy + random.gauss(0, 1) * sqrt_tau_over_mk
            uz = uz + random.gauss(0, 1) * sqrt_tau_over_mk
            
            h_N[(nn+k+1)%N, 0] = ux - Lx * np.floor(ux / Lx)
            h_N[(nn+k+1)%N, 1] = uy - Ly * np.floor(uy / Ly)
            h_N[(nn+k+1)%N, 2] = uz - Lz * np.floor(uz / Lz)
        
        energy = MetropolisB_optimized(h, h_N, N, T, mass_h, nn, j, x,
                                       yLa_h_1, yLO_h_1,yLO_h_2,yLO_h_3, yTa_h_1,yTa_h_2, yTo1_h_0, yTo1_h_1, yTo1_10_h,yTo2_h_0, yTo2_h_1, yTo2_10_h)
        
        if np.random.random() <= np.exp(-energy / T):
            h = h_N.copy()
            track_h = track_h + 1
    
    
    return x, h, track_e, track_h, track_COMe, track_COMh, se, sh, seC, shC


@numba.jit(nopython=True)
def MetropolisB_optimized(x, x_N, N, T, mass, nn, j, h,
                          gg_LA, gg_LO,gg_LO2,gg_LO3, gg_TA,gg_TA2, gg_TO_0, gg_TO_1, gg_TO_01,gg_TO2_0, gg_TO2_1, gg_TO2_01):
    # Pre-compute constants
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz
    inv_N = 1.0 / N
    
    # Self-interaction - OLD configuration (exact triangular loop as original)
    PEi = 0.0
    dum = N - 1
    for k in range(j - 1):
        ind = (nn + k + 1) % N
        ix = x[ind, 0]
        iy = x[ind, 1]
        iz = x[ind, 2]
        
        for l in range(dum):
            ind2 = (ind + 1 + l) % N
            jx = x[ind2, 0]
            jy = x[ind2, 1]
            jz = x[ind2, 2]
            
            tau = abs(ind - ind2)
            
            dx = ix - jx
            dy = iy - jy
            dz = iz - jz
            dx = dx - Lx * np.round(dx * inv_Lx)
            dy = dy - Ly * np.round(dy * inv_Ly)
            dz = dz - Lz * np.round(dz * inv_Lz)
            rtot = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            PEi = PEi - (Heff_LO(rtot, tau, gg_LO, gg_LO, wLO) + Heff_LO(rtot, tau, gg_LO2, gg_LO2, wLO2) + Heff_LO(rtot, tau, gg_LO3, gg_LO3, wLO3) +
                        Heff_AC(rtot, tau, gg_LA, gg_LA, sLA) +
                        (Heff_AC(rtot, tau, gg_TA, gg_TA, sTA1) + Heff_AC(rtot, tau, gg_TA2, gg_TA2, sTA2)+
                             Heff_TO_0(rtot, tau, gg_TO_0, gg_TO_0, wTO) ) +
                        (
                             Heff_TO_0(rtot, tau, gg_TO2_0, gg_TO2_0, wTO) ))
        
        dum = dum - 1
    
    # Cross-interaction - OLD configuration
    PEi_cross = 0.0
    for k in range(j - 1):
        ind = (nn + k + 1) % N
        ix = x[ind, 0]
        iy = x[ind, 1]
        iz = x[ind, 2]
        
        for l in range(N):
            if l != -1:
                jx = h[l, 0]
                jy = h[l, 1]
                jz = h[l, 2]
                
                tau = abs(ind - l)
                
                dx = ix - jx
                dy = iy - jy
                dz = iz - jz
                dx = dx - Lx * np.round(dx * inv_Lx)
                dy = dy - Ly * np.round(dy * inv_Ly)
                dz = dz - Lz * np.round(dz * inv_Lz)
                rtot = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                PEi_cross = PEi_cross + (Heff_LO(rtot, tau, yLO_e_1, yLO_h_1, wLO) +Heff_LO(rtot, tau, yLO_e_2, yLO_h_2, wLO2) +Heff_LO(rtot, tau, yLO_e_3, yLO_h_3, wLO3) +
                                        Heff_AC(rtot, tau, yLa_e_1, yLa_h_1, sLA) +
                                         (Heff_AC(rtot, tau, yTa_e_1, yTa_h_1, sTA1) + Heff_AC(rtot, tau, yTa_e_2, yTa_h_2, sTA2)+
                                             Heff_TO_0(rtot, tau, yTo1_e_0, yTo1_h_0, wTO) ) + 
                                        (
                                            Heff_TO_0(rtot, tau, yTo2_e_0, yTo2_h_0, wTO) ) 
                                         )
    
    # Self-interaction - NEW configuration (exact triangular loop)
    PEf = 0.0
    dum = N - 1
    for k in range(j - 1):
        ind = (nn + k + 1) % N
        ix = x_N[ind, 0]
        iy = x_N[ind, 1]
        iz = x_N[ind, 2]
        
        for l in range(dum):
            ind2 = (ind + 1 + l) % N
            jx = x_N[ind2, 0]
            jy = x_N[ind2, 1]
            jz = x_N[ind2, 2]
            
            tau = abs(ind - ind2)
            
            dx = ix - jx
            dy = iy - jy
            dz = iz - jz
            dx = dx - Lx * np.round(dx * inv_Lx)
            dy = dy - Ly * np.round(dy * inv_Ly)
            dz = dz - Lz * np.round(dz * inv_Lz)
            rtot = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            PEf = PEf - (Heff_LO(rtot, tau, gg_LO, gg_LO, wLO) + Heff_LO(rtot, tau, gg_LO2, gg_LO2, wLO2) + Heff_LO(rtot, tau, gg_LO3, gg_LO3, wLO3) +
                        Heff_AC(rtot, tau, gg_LA, gg_LA, sLA) +
                        (Heff_AC(rtot, tau, gg_TA, gg_TA, sTA1) +Heff_AC(rtot, tau, gg_TA2, gg_TA2, sTA2) + 
                             Heff_TO_0(rtot, tau, gg_TO_0, gg_TO_0, wTO) ) +
                        ( 
                             Heff_TO_0(rtot, tau, gg_TO2_0, gg_TO2_0, wTO) ))
        
        dum = dum - 1
    
    # Cross-interaction - NEW configuration
    PEf_cross = 0.0
    for k in range(j - 1):
        ind = (nn + k + 1) % N
        ix = x_N[ind, 0]
        iy = x_N[ind, 1]
        iz = x_N[ind, 2]
        
        for l in range(N):
            if l != -1:
                jx = h[l, 0]
                jy = h[l, 1]
                jz = h[l, 2]
                
                tau = abs(ind - l)
                
                dx = ix - jx
                dy = iy - jy
                dz = iz - jz
                dx = dx - Lx * np.round(dx * inv_Lx)
                dy = dy - Ly * np.round(dy * inv_Ly)
                dz = dz - Lz * np.round(dz * inv_Lz)
                rtot = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                PEf_cross = PEf_cross +  (Heff_LO(rtot, tau, yLO_e_1, yLO_h_1, wLO) +Heff_LO(rtot, tau, yLO_e_2, yLO_h_2, wLO2) +Heff_LO(rtot, tau, yLO_e_3, yLO_h_3, wLO3) +
                                        Heff_AC(rtot, tau, yLa_e_1, yLa_h_1, sLA) +
                                         (Heff_AC(rtot, tau, yTa_e_1, yTa_h_1, sTA1) + Heff_AC(rtot, tau, yTa_e_2, yTa_h_2, sTA2) +
                                             Heff_TO_0(rtot, tau, yTo1_e_0, yTo1_h_0, wTO) ) + 
                                        (
                                            Heff_TO_0(rtot, tau, yTo2_e_0, yTo2_h_0, wTO) ) 
                                         )
    
    # Coulomb difference
    C_diff = 0.0
    for k in range(j - 1):
        ind = (nn + k + 1) % N
        C_diff = C_diff + inv_N * (C(x_N[ind, 0], x_N[ind, 1], x_N[ind, 2],
                                     h[ind, 0], h[ind, 1], h[ind, 2]) -
                                   C(x[ind, 0], x[ind, 1], x[ind, 2],
                                     h[ind, 0], h[ind, 1], h[ind, 2]))
    
    EN = C_diff + 2 * (PEf + 0.5*PEf_cross - PEi - 0.5*PEi_cross)
    
    return EN

#COM moves if needed
@numba.jit(nopython=True)
def MetropolisW_optimized(x, x_New, N, T, mass, h):
    inv_N = 1.0 / N
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz
    
    # Coulomb energy difference
    sumPE = 0.0
    for i in range(N):
        PEi = C(x[i, 0], x[i, 1], x[i, 2], h[i, 0], h[i, 1], h[i, 2]) * inv_N
        PEf = C(x_New[i, 0], x_New[i, 1], x_New[i, 2], h[i, 0], h[i, 1], h[i, 2]) * inv_N
        sumPE = sumPE + (PEf - PEi)
    
    # Cross-interaction energy difference
    sumPE_cross = 0.0
    for i in range(N):
        for k1 in range( N):
            if(i!=k1):
                ix = x[i, 0]
                iy = x[i, 1]
                iz = x[i, 2]
                jx = h[k1, 0]
                jy = h[k1, 1]
                jz = h[k1, 2]
            
                tau = abs(i - k1)
            
            # Old configuration
                dx = ix - jx
                dy = iy - jy
                dz = iz - jz
                dx = dx - Lx * np.round(dx * inv_Lx)
                dy = dy - Ly * np.round(dy * inv_Ly)
                dz = dz - Lz * np.round(dz * inv_Lz)
                rtot = np.sqrt(dx*dx + dy*dy + dz*dz)
            
                PEi = (Heff_LO(rtot, tau, yLO_e_1, yLO_h_1, wLO) +
                                    Heff_AC(rtot, tau, yLa_e_1, yLa_h_1, sLA) +
                                     (Heff_AC(rtot, tau, yTa_e_1, yTa_h_1, sTA1) +
                                         Heff_TO_0(rtot, tau, yTo1_e_0, yTo1_h_0, wTO) ) + 
                                    (Heff_AC(rtot, tau, yTa_e_2, yTa_h_2, sTA2) +
                                        Heff_TO_0(rtot, tau, yTo2_e_0, yTo2_h_0, wTO) ) 
                                     )
            
            # New configuration
                ix = x_New[i, 0]
                iy = x_New[i, 1]
                iz = x_New[i, 2]
            
                dx = ix - jx
                dy = iy - jy
                dz = iz - jz
                dx = dx - Lx * np.round(dx * inv_Lx)
                dy = dy - Ly * np.round(dy * inv_Ly)
                dz = dz - Lz * np.round(dz * inv_Lz)
                rtot = np.sqrt(dx*dx + dy*dy + dz*dz)
            
                PEf = (Heff_LO(rtot, tau, yLO_e_1, yLO_h_1, wLO) +
                                    Heff_AC(rtot, tau, yLa_e_1, yLa_h_1, sLA) +
                                     (Heff_AC(rtot, tau, yTa_e_1, yTa_h_1, sTA1) +
                                         Heff_TO_0(rtot, tau, yTo1_e_0, yTo1_h_0, wTO) ) + 
                                    (Heff_AC(rtot, tau, yTa_e_2, yTa_h_2, sTA2) +
                                        Heff_TO_0(rtot, tau, yTo2_e_0, yTo2_h_0, wTO) 
                                        ) 
                                     )
            
                sumPE_cross = sumPE_cross + (PEf - PEi)
    
    return sumPE + 2 * sumPE_cross



# Parameters #
##
##


# Total number of beads
Nx = 10
Ny = 10
Nz = 25
N = Nx*Ny*Nz

#Staging Segment Length
je = int(N/10) + 1
jh = int(N/10) + 1


##NOTE that I am working in atomic units!!!
mass_e=0.20
mass_h = 0.20
ep= 4.4
ep_st = 18.6

a = 0.0

#Coulomb Potential Smoothening Parameter
Eg = 1.85/27.211386245988
ae = (1/(Eg*ep))



#Box Size
Lx = 1000
Ly = 1000
Lz = 1000


V_UC = 5373.4583
den_UC = 786.53
m_UC = 1.0

consts = np.sqrt( m_UC/ ((2*np.pi)**2))

wLO = 0.000661487

wLO2 = 0.000404242

wLO3 = 0.000257245

wTO = wLO/1.283


sLA = 0.00201

sTA1 = 0.00085

sTA2 = 0.00118



#Debye Sphere Cutoff
q = (6*np.pi**2 / V_UC)**(1/3)


pek = 1/ep - 1/ep_st

#LO Mode 1 with right weight of lattice polarizability from EPW
ae = np.sqrt(0.5/wLO)*pek*((0.00035084/0.000366)**2)
yLO_e_1 = np.sqrt(ae*(wLO**2)*np.sqrt(1/8/wLO))
yLO_h_1 = yLO_e_1

#LO Mode 2
ae2 = np.sqrt(0.5/wLO2)*pek*((0.00002/0.000286)**2)
yLO_e_2 = np.sqrt(ae2*(wLO2**2)*np.sqrt(1/8/wLO2))
yLO_h_2 = yLO_e_2

#LO Mode 3
ae3 = np.sqrt(0.5/wLO3)*pek*((0.00006/0.000228)**2)
yLO_e_3 = np.sqrt(ae3*(wLO3**2)*np.sqrt(1/8/wLO3))
yLO_h_3 = yLO_e_3


D_LA_e = 0.0113
D_LA_h = 0.00812

const_ac_LA = np.sqrt(1/(8*np.pi**2 * den_UC * sLA))

yLa_e_1 = D_LA_e*const_ac_LA
yLa_h_1 = D_LA_h*const_ac_LA


D_TA_e1 = 0.000254
D_TA_h1 = 0.000751

const_ac_TA1 = np.sqrt(1/(8*np.pi**2 * den_UC * sTA1))

yTa_e_1 = D_TA_e1*const_ac_TA1
yTa_h_1 = D_TA_h1*const_ac_TA1


D_TA_e2 = 0.000441
D_TA_h2 = 0.000336

const_ac_TA2 = np.sqrt(1/(8*np.pi**2 * den_UC * sTA2))

yTa_e_2 = D_TA_e2*const_ac_TA2
yTa_h_2 = D_TA_h2*const_ac_TA2


const_TO = np.sqrt(1/(8*np.pi**2 * den_UC * wTO))


D_TO1_e0 = 0.0
D_TO1_h0 = 0.0


yTo1_e_0 = D_TO1_e0*const_TO
yTo1_h_0 = D_TO1_h0*const_TO


D_TO2_e0 = 0.0
D_TO2_h0 = 0.0

yTo2_e_0 = D_TO2_e0*const_TO
yTo2_h_0 = D_TO2_h0*const_TO


# Number of sweeps
steps = 600000
# Averaging frequency
Nsamp = 50

#aid = str(1)
#nam="Ener1"+aid+".txt"

aid = str(sys.argv[1])
nam="EnerC"+aid+".txt"
namLO="EnerLO"+aid+".txt"
namLOc="cEnerLO"+aid+".txt"

namLO2="EnerLO2"+aid+".txt"
namLO2c="cEnerLO2"+aid+".txt"

namLO3="EnerLO3"+aid+".txt"
namLO3c="cEnerLO3"+aid+".txt"


namLA = "EnerLA"+aid+".txt"
namLAc = "cEnerLA"+aid+".txt"
namTA1 = "EnerTA1"+aid+".txt"
namTA1c = "cEnerTA1"+aid+".txt"
namTA2 = "EnerTA2"+aid+".txt"
namTA2c = "cEnerTA2"+aid+".txt"


namTO1 = "EnerTO1"+aid+".txt"
namTO1c = "cEnerTO1"+aid+".txt"

namTO2 = "EnerTO2"+aid+".txt"
namTO2c = "cEnerTO2"+aid+".txt"

nam2="x"+aid+".txt"
nam3="h"+aid+".txt"



#Initialize ring polymers on lattice
@numba.jit(nopython=True)
def init_polymer(N, T, mass, center_x, center_y, center_z, Lx, Ly, Lz):
    """
    Initialize a single polymer as a random walk.
    Bead-to-bead distance scales as thermal wavelength: σ = sqrt(1/(N*T*m))
    """
    pos = np.zeros((N, 3))

    if T > 1e-10:
        sigma = np.sqrt(1.0 / (N * T * mass))
    else:
        sigma = 5.0

    sigma = min(sigma, 50.0)

    pos[0, 0] = center_x
    pos[0, 1] = center_y
    pos[0, 2] = center_z

    for i in range(1, N):
        pos[i, 0] = pos[i-1, 0] + np.random.normal(0, sigma)
        pos[i, 1] = pos[i-1, 1] + np.random.normal(0, sigma)
        pos[i, 2] = pos[i-1, 2] + np.random.normal(0, sigma)

        pos[i, 0] = pos[i, 0] - Lx * np.floor(pos[i, 0] / Lx)
        pos[i, 1] = pos[i, 1] - Ly * np.floor(pos[i, 1] / Ly)
        pos[i, 2] = pos[i, 2] - Lz * np.floor(pos[i, 2] / Lz)

    return pos


@numba.jit(nopython=True)
def init_exciton(N, T, mass_e, mass_h, separation, Lx, Ly, Lz):
    """
    Initialize electron and hole polymers.

    separation = 0: overlapping
    separation > 0: start separated
    """
    x = np.zeros((N, 3))
    h = np.zeros((N, 3))

    cx, cy, cz = Lx/2, Ly/2, Lz/2

    x = init_polymer(N, T, mass_e, cx + separation/2, cy, cz, Lx, Ly, Lz)
    h = init_polymer(N, T, mass_h, cx - separation/2, cy, cz, Lx, Ly, Lz)

    return x, h




sep = ae + int(aid)*20.0

T = 0.0003167*1.0
print("Temperature: ", T*100/0.0003167)

x, h = init_exciton(N, T, mass_e, mass_h, sep, Lx, Ly, Lz)

#If you want to read in electron and hole positions from a previous run

#x= np.loadtxt(sys.argv[8])
#h= np.loadtxt(sys.argv[9])


# Load existing observables if they exist, otherwise start fresh
def load_or_empty(filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return np.loadtxt(filename)
    return np.array([])


Ener1_C = load_or_empty(nam)

Ener1_LO = load_or_empty(namLO)
Ener1_LOc = load_or_empty(namLOc)

Ener1_LO2 = load_or_empty(namLO2)
Ener1_LO2c = load_or_empty(namLO2c)

Ener1_LO3 = load_or_empty(namLO3)
Ener1_LO3c = load_or_empty(namLO3c)

Ener1_LA = load_or_empty(namLA)
Ener1_LAc = load_or_empty(namLAc)

Ener1_TA1 = load_or_empty(namTA1)
Ener1_TA1c = load_or_empty(namTA1c)

Ener1_TA2 = load_or_empty(namTA2)
Ener1_TA2c = load_or_empty(namTA2c)

Ener1_TO1 = load_or_empty(namTO1)
Ener1_TO2 = load_or_empty(namTO2)


count_e = 0 
count_h = 0
count_eC = 0
count_hC = 0
se = 0
sh = 0
seC = 0
shC = 0
    # Perform sweeps

for step in range(steps):

      # Perform MC moves
      x, h, count_e, count_h, count_eC, count_hC, se, sh, seC, shC =Monte_Carlo(x,h,T,count_e,count_h,count_eC,count_hC,se,sh,seC,shC)

      # Every Nsamp steps, compute expectations
      if(step%Nsamp == 0):
          print("Step: ", step, flush=True)

      if(step%Nsamp==0 and step >= 100):
          en = 0
          r_sim = np.zeros(N)

          for i in range(N):
              en = en + Est(x[i,0],x[i,1],x[i,2],h[i,0],h[i,1],h[i,2])
            # xu = (x[i,0]-h[i,0]) - Lx*np.round((x[i,0]-h[i,0]) /Lx)
            # yu = (x[i,1]-h[i,1]) - Ly*np.round((x[i,1]-h[i,1]) /Ly)
            # zu = (x[i,2]-h[i,2]) - Lz*np.round((x[i,2]-h[i,2]) /Lz)
            # r_sim[i] = np.sqrt( (xu)**2 + (yu)**2 + (zu)**2   )



          Ener1_C = np.append(Ener1_C,en/N)
          #selfe = Est_self_LO(x,yLO_e_1,wLO)
          #selfh = Est_self_LO(h,yLO_h_1,wLO)
          #selfeh =  Est_cross_LO(x,h,yLO_e_1,yLO_h_1,wLO)
          #tot = selfe+selfh+selfeh
          #print(tot)

          selfeh =  Est_cross_LO(x,h,yLO_e_1,yLO_h_1,wLO)
          Ener1_LOc = np.append(Ener1_LOc,selfeh)

          Ener1_LO = np.append(Ener1_LO, Est_self_LO(x,yLO_e_1,wLO) + Est_self_LO(h,yLO_h_1,wLO) )
          
          selfeh =  Est_cross_LO(x,h,yLO_e_2,yLO_h_2,wLO2)
          Ener1_LO2c = np.append(Ener1_LO2c,selfeh)

          Ener1_LO2 = np.append(Ener1_LO2, Est_self_LO(x,yLO_e_2,wLO2) + Est_self_LO(h,yLO_h_2,wLO2) )
          
          selfeh =  Est_cross_LO(x,h,yLO_e_3,yLO_h_3,wLO3)
          Ener1_LO3c = np.append(Ener1_LO3c,selfeh)

          Ener1_LO3 = np.append(Ener1_LO3, Est_self_LO(x,yLO_e_3,wLO3) + Est_self_LO(h,yLO_h_3,wLO3) )

          selfeh = Est_cross_AC(x,h,yLa_e_1,yLa_h_1,sLA)
          Ener1_LAc = np.append(Ener1_LAc,selfeh)


          Ener1_LA = np.append(Ener1_LA, Est_self_AC(x,yLa_e_1,sLA) + Est_self_AC(h,yLa_h_1,sLA))
          
          
          selfeh = Est_cross_AC(x,h,yTa_e_1,yTa_h_1,sTA1)
          Ener1_TA1c = np.append(Ener1_TA1c, selfeh)
          
          Ener1_TA1 = np.append(Ener1_TA1, Est_self_AC(x,yTa_e_1,sTA1) + Est_self_AC(h,yTa_h_1,sTA1))

          selfeh = Est_cross_AC(x,h,yTa_e_2,yTa_h_2,sTA2)
          Ener1_TA2c = np.append(Ener1_TA2c, selfeh)


          Ener1_TA2 = np.append(Ener1_TA2, Est_self_AC(x,yTa_e_2,sTA2) + Est_self_AC(h,yTa_h_2,sTA2) )
          

        #TO Modes don't matter to e-ph coupling at lowest order here because this material is centrosymmetric
          Ener1_TO1 = np.append(Ener1_TO1, 0.0  )
          Ener1_TO2 = np.append(Ener1_TO2, 0.0  )

          np.savetxt(nam,Ener1_C)
          np.savetxt(namLO,Ener1_LO)
          np.savetxt(namLOc,Ener1_LOc)
          
          np.savetxt(namLO2,Ener1_LO2)
          np.savetxt(namLO2c,Ener1_LO2c)
          
          np.savetxt(namLO3,Ener1_LO3)
          np.savetxt(namLO3c,Ener1_LO3c)


          np.savetxt(namLA,Ener1_LA)
          np.savetxt(namLAc,Ener1_LAc)

          np.savetxt(namTA1,Ener1_TA1)
          np.savetxt(namTA1c,Ener1_TA1c)
          np.savetxt(namTA2,Ener1_TA2)
          np.savetxt(namTA2c,Ener1_TA2c)

          np.savetxt(namTO1,Ener1_TO1)
          np.savetxt(namTO2,Ener1_TO2)
            
        #Print total energy in meV
          print(r"<E> (meV): ",(np.mean(Ener1_C[10:]) + np.mean(Ener1_LA[10:]) + np.mean(Ener1_LAc[10:]) + np.mean(Ener1_LO[10:]) + np.mean(Ener1_LOc[10:]) +np.mean(Ener1_LO2[10:]) + np.mean(Ener1_LO2c[10:]) +np.mean(Ener1_LO3[10:]) + np.mean(Ener1_LO3c[10:])+ np.mean(Ener1_TA1[10:]) +np.mean(Ener1_TA1c[10:]) + np.mean(Ener1_TA2[10:]) + np.mean(Ener1_TA2c[10:]) + np.mean(Ener1_TO1[10:]) + np.mean(Ener1_TO2[10:])  )  *1000*27.211399 )
          np.savetxt(nam2,x)
          np.savetxt(nam3,h)












