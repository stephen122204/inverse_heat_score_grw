# Resonance kill test (exploratory, 2026-09-01)

## Second-order formula validation (grid-stable)

Purified fully-spectral solver (sine-projected flux derivative, no
finite differences), convergence-checked across (M, modes, dt) =
(2048, 220, 2.5e-4) and (4096, 300, 1.25e-4):

- a=0.10, n=3, h=0.028: measured/predicted = 1.0024 (both grids)
- a=0.50, n=3, h=0.028: 1.0683 (both grids; third-order onset)
- a-sweep at n=3 (hybrid run): 1.001 / 1.002 / 1.010 at a = 0.05 /
  0.10 / 0.20, exact a^2 scaling.

Prediction: a^2 b(T), b(T) = alpha m k^2 phi(1-m phi) e^{-2 alpha k^2 T}
(e^{gamma_2k T} - e^{2 gamma_k T})/(gamma_2k - 2 gamma_k).

## C1 cascade (continuum, hybrid run at (2048, 220, 5e-4))

a=0.5, n=3, h=0.028: total continuum E2 = 1.99 percent (particle
method: 2.53 percent). Error coefficients: 1.55e-2 (3pi, matching the
linear filter deficit 1.52e-2), 1.59e-2 (6pi; exact-formula prediction
1.48e-2), 1.76e-2 (9pi, third harmonic largest), 8.9e-3 (12pi).

## Particle-reconstruction signature (gate item iv): PRESENT

run_campaign_density on the same case (N=4000, production path):
E2 = 2.53 percent; error coefficients 3.22e-2 (3pi; linear deficit plus
the particle layer), 1.41e-2 (6pi; formula predicts 1.48e-2, within
5 percent), 1.31e-2 (9pi), 5.1e-3 (12pi). The cascade is in the actual
method's output, not only the continuum model.

## The (n=6, h=0.014) regime is numerically unreachable (and why)

In the purified solver this case blows up (2e8 at the coarse grid,
1e10 refined - growing with refinement); the earlier hybrid ratio 1.33
was finite-difference dissipation masking roundoff amplification.
Independent confirmation of the boundary formula: h*(delta = 1e-16) =
sqrt(2 alpha T/(e ln 1e16)) = 0.0141, so h = 0.014 sits exactly at the
machine-precision instability boundary and roundoff is amplified to
O(1) through the physics. Consequence: the asymptotic resonance regime
(k^2 ~ 1/h, h below ~0.016) cannot be probed in double precision; the
resonance theorem (gate item ii) must carry the asymptotics
analytically, with numerics confined to the accessible window where the
formula is now validated.
