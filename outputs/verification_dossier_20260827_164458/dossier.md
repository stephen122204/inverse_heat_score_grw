# Operator-verification dossier

Code commit `f6b106fc5c06`, grid convention `cell-centered`, generated 2026-08-27T16:45:01.188837+00:00.

**Verdicts: 22/22 PASS.**  Every reference value is analytic on the bounded Neumann domain.  No table compares one simulation against another, except the two self-convergence columns that isolate temporal order from spatial and kernel floors.

## kernel_verification

Neumann kernel: unit mass through the constant mode, termwise zero wall derivative (evaluated exactly at the walls), and modal damping equal to exp(-(kh)^2/2) at machine precision.

```
         h  mode  mass_error  wall_derivative  damping_error
1.0000e-02     4  0.0000e+00       1.8322e-15     1.6653e-16
1.0000e-02    12  0.0000e+00       1.5480e-14     2.2204e-16
2.0000e-02     4  0.0000e+00       1.7893e-15     1.1102e-16
2.0000e-02    12  0.0000e+00       1.2508e-14     1.6653e-16
4.0000e-02     4  0.0000e+00       1.6276e-15     1.6653e-16
4.0000e-02    12  0.0000e+00       5.3316e-15     9.7145e-17
```

## score_static

Neumann KDE score of quantile particles against the analytic score of the manufactured density (epsilon = 0; the density is bounded below by 0.5, so no floor is involved).  The error is bandwidth bias at the enforced O(h^2) rate.

```
         h  n_particles   score_L2  score_Linf
2.0000e-02         1000 8.3565e-02  1.3953e-01
2.0000e-02         4000 8.3564e-02  1.3941e-01
2.0000e-02        16000 8.3565e-02  1.3941e-01
1.0000e-02         1000 2.1230e-02  3.5866e-02
1.0000e-02         4000 2.1227e-02  3.5544e-02
1.0000e-02        16000 2.1227e-02  3.5494e-02
```

## commutation_constant

The forward multiplier, Tikhonov filter, and spectral cutoff act exactly on discrete cosine modes through the public APIs.  The cutoff column is the relative residual |A * output - mode| because 1/A is astronomically large at high modes.

```
 mode  forward_residual  tikhonov_residual  cutoff_residual_relative
    5        9.9920e-16         4.0523e-15                1.6098e-15
   25        8.8690e-16         3.2752e-14                8.3267e-15
   60        9.3760e-16         3.2842e-14                3.3751e-14
```

## operator_structure

The conservative finite-volume operator is exactly symmetric, annihilates constants, conserves mass, is negative semidefinite, and satisfies the adjoint identity.

```
  M   symmetry  nullspace  column_sums  max_eigenvalue  adjoint_residual
 50 0.0000e+00 0.0000e+00   0.0000e+00      6.3104e-15        2.8422e-14
100 0.0000e+00 0.0000e+00   0.0000e+00      3.0663e-14        1.3642e-12
200 0.0000e+00 0.0000e+00   0.0000e+00      2.3616e-14        3.6380e-12
400 0.0000e+00 0.0000e+00   0.0000e+00      6.0947e-13        1.4552e-11
```

## mms_spatial

Forced manufactured solution, spatial refinement at fixed dt = 1e-4: second-order convergence against the exact solution.

```
  M         dt  rel_error      order
 25 1.0000e-04 2.9884e-05        NaN
 50 1.0000e-04 7.4751e-06 1.9992e+00
100 1.0000e-04 1.8690e-06 1.9998e+00
200 1.0000e-04 4.6725e-07 2.0000e+00
```

## mms_temporal

Forced manufactured solution, temporal refinement at fixed M = 400: second-order self-convergence against the dt-reference.  The vs-exact column floors at the spatial error, as it must.

```
  M         dt  rel_error_vs_exact  rel_error_vs_dt_ref  order_self
400 5.0000e-03          2.1760e-07           2.1877e-07         NaN
400 2.5000e-03          1.1437e-07           5.4533e-08  2.0042e+00
400 1.2500e-03          1.1377e-07           1.3473e-08  2.0171e+00
400 6.2500e-04          1.1591e-07           3.2078e-09  2.0704e+00
```

## particle_N_refinement

Exact-score density particles at fixed physical bandwidth: the particle-count error converges to the analytically predicted kernel-bias floor and the reconstruction mass is exact by construction.

```
         h  n_particles  rel_error  mass_error  predicted_bias_floor  error_over_floor
2.0000e-02          500 6.0423e-03  4.4409e-16            5.8695e-03        1.0294e+00
2.0000e-02         1000 6.0422e-03  6.6613e-16            5.8695e-03        1.0294e+00
2.0000e-02         2000 6.0422e-03  6.6613e-16            5.8695e-03        1.0294e+00
2.0000e-02         4000 6.0422e-03  6.6613e-16            5.8695e-03        1.0294e+00
2.0000e-02         8000 6.0422e-03  4.4409e-16            5.8695e-03        1.0294e+00
1.0000e-02          500 1.6545e-03  4.4409e-16            1.4772e-03        1.1200e+00
1.0000e-02         1000 1.6545e-03  6.6613e-16            1.4772e-03        1.1200e+00
1.0000e-02         2000 1.6544e-03  2.2204e-16            1.4772e-03        1.1200e+00
1.0000e-02         4000 1.6544e-03  6.6613e-16            1.4772e-03        1.1200e+00
1.0000e-02         8000 1.6544e-03  6.6613e-16            1.4772e-03        1.1200e+00
```

## particle_dt_refinement

Time-step refinement at fixed h, N, M.  Consecutive Richardson differences give the clean explicit-Euler first order, and the truth error's ratio to the analytic kernel-bias floor approaches one as dt shrinks — the residual floor discrepancy is temporal, nothing else.

```
         h  n_particles   M         dt  rel_error_vs_truth  error_over_floor  richardson_diff  richardson_order
2.0000e-02         4000 400 8.0000e-03          6.5589e-03        1.1175e+00       3.8579e-04               NaN
2.0000e-02         4000 400 4.0000e-03          6.2133e-03        1.0586e+00       1.9322e-04        9.9761e-01
2.0000e-02         4000 400 2.0000e-03          6.0422e-03        1.0294e+00       9.6688e-05        9.9880e-01
2.0000e-02         4000 400 1.0000e-03          5.9572e-03        1.0149e+00       4.8364e-05        9.9940e-01
2.0000e-02         4000 400 5.0000e-04          5.9148e-03        1.0077e+00       2.4187e-05        9.9970e-01
2.0000e-02         4000 400 2.5000e-04          5.8936e-03        1.0041e+00       1.2095e-05        9.9985e-01
2.0000e-02         4000 400 1.2500e-04          5.8830e-03        1.0023e+00              NaN               NaN
```

## particle_M_refinement

Grid refinement at fixed physical bandwidth leaves the error unchanged: the endpoint-era h = c * dx coupling that confounded the archived grid study is gone.

```
         h  n_particles         dt   M  rel_error  mass_error
2.0000e-02         4000 2.0000e-03 100 6.0864e-03  4.4409e-16
2.0000e-02         4000 2.0000e-03 200 6.0511e-03  4.4409e-16
2.0000e-02         4000 2.0000e-03 400 6.0422e-03  6.6613e-16
2.0000e-02         4000 2.0000e-03 800 6.0400e-03  2.2204e-16
```

## particle_M_coupled_control

The paired control reinstates the endpoint-era coupling h = 8 dx: the error now falls with the grid exactly as the archived curve did, tracking the shrinking kernel-bias floor.  Together with the fixed-h table this proves the archived grid 'convergence' was the bandwidth coupling.

```
bandwidth_rule   M          h  rel_error  predicted_bias_floor
         h=8dx 100 8.0000e-02 8.2638e-02            8.2472e-02
         h=8dx 200 4.0000e-02 2.3037e-02            2.2865e-02
         h=8dx 400 2.0000e-02 6.0422e-03            5.8695e-03
         h=8dx 800 1.0000e-02 1.6522e-03            1.4772e-03
```

## Verdicts

- PASS  kernel_mass<=1e-11
- PASS  kernel_wall<=1e-9
- PASS  kernel_damping<=1e-11
- PASS  score_finite
- PASS  score_h_order_Linf_in_[1.8,2.2]
- PASS  score_h_order_L2_in_[1.8,2.2]
- PASS  commutation<=1e-9
- PASS  op_symmetry==0
- PASS  op_nullspace<=1e-8
- PASS  op_conservation<=1e-8
- PASS  op_dissipative<=1e-8
- PASS  op_adjoint<=1e-7
- PASS  mms_spatial_order>=1.9
- PASS  mms_temporal_order>=1.9
- PASS  particle_N_floor_ratio_in_[0.7,1.5]
- PASS  particle_mass<=1e-12
- PASS  particle_N_spread_h=0.02<=1e-3
- PASS  particle_N_spread_h=0.01<=1e-3
- PASS  particle_dt_richardson_order_in_[0.9,1.1]
- PASS  particle_floor_ratio_at_finest_dt<=1.01
- PASS  particle_M_grid_independent(spread<=5%)
- PASS  coupled_control_error_falls(M100/M800>=3)
