# Exciton-Phonon-PIMC
PIMC code for exciton-polaron binding energies in polar semiconductors

## Overview

This repository contains GPU-accelerated Path Integral Monte Carlo (PIMC) code for computing exciton-polaron binding energies in polar semiconductors. The code integrates out phonon degrees of freedom analytically, leaving an effective action that is sampled via PIMC on the electron and hole imaginary-time paths.

The electron-phonon coupling is treated for three phonon branches: LO (longitudinal optical), TO (transverse optical), and LA (longitudinal acoustic). All energies are extracted using the virial energy estimator.

Single-polaron (isolated electron or hole) binding energies can be obtained by turning off the Coulomb and electron-hole cross interaction terms and removing the second quasiparticle.

## Files

**ex_Pol.py** — Main PIMC simulation code for the exciton-polaron. Implements the effective electron-phonon Hamiltonian, virial energy estimator, and Monte Carlo sampling for electron and hole paths. Handles LO, TO, and LA phonon modes, as well as the Coulomb interaction.

**ac_int_gpu.py** — GPU-accelerated precomputation of interpolation grids for the LA phonon mode. The LA mode requires numerical momentum integration (Gauss-Legendre quadrature) plus analytical zero-temperature contributions, which are too expensive to evaluate on the fly during PIMC. This script precomputes the effective Hamiltonian, spatial derivative, temperature derivative, and full virial estimator on a 2D grid in (r, τ) and saves them for interpolation during the simulation.

## Virial Energy Estimator

All energies are computed using the virial estimator, which has lower variance than the thermodynamic estimator:

$$E_{\text{vir}} = \frac{1}{N} \sum_{i=1}^{N} \left\langle \bar{V}(\mathbf{x}_i, \beta/N) + \frac{1}{\beta} \frac{\partial}{\partial \beta} \bar{V}(\mathbf{x}_i, \beta/N) + \frac{1}{2} \mathbf{x}_i \cdot \frac{\partial \bar{V}(\mathbf{x}_i, \beta/N)}{\partial \mathbf{x}_i} \right\rangle$$

where V̄ is the effective potential, β is the inverse temperature, N is the number of imaginary-time slices, and the three terms correspond to the potential energy, temperature derivative, and spatial derivative contributions respectively. For the Coulomb interaction, the temperature derivative vanishes.

## Units

All calculations are performed in atomic units (Hartree for energy, Bohr radii for length). Temperature is expressed as inverse energy (β = 1/T in Hartree⁻¹).

## Requirements

- Python 3.x
- NumPy
- CuPy
- Numba (with CUDA support)
- SciPy (for validation)

## Usage

1. Precompute the LA interpolation grids:
```bash
python ac_int_gpu.py
```
This generates grid files: `H_Eff_LA_gpu.txt`, `der_spat_LA_gpu.txt`, `der_temp_LA_gpu.txt`, `Vir_LA_gpu.txt`.

2. Run the Exciton PIMC simulation:
```bash
python ex_Pol.py
```

3. Also run the single-polaron PIMC simulations by turning off one ring polymer at a time.
