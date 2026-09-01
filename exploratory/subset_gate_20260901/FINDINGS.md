# End-to-end subset gate (exploratory, 2026-09-01)

Production drivers on the full G1+G2 closure blocks, bandwidth_clean H and
VB05, and the VB05 oracle row. 79 of 79 rows completed, none failed. The
VB05 rows exercised the 8x/16x de-crimed data gate and the SVD Tikhonov
branch through the production path for the first time.

## Closure pilot verdicts (the representation evidence)

- All 8 reference-resolution pairs (G1, G2 x both closures x both kinds)
  pass the 1e-4 gate at BOTH archived times. G2 regularized q: 1.9e-4 at
  the reg->unreg bridge... see gates JSON for exact pair values.
- h-bridge: the distance from the regularized reference to the
  unregularized wrong-limit reference falls as h^2 (fitted exponent 1.994
  on both cases, both closures). The finite-bandwidth dynamics approach
  the wrong continuum law at second order.
- Four-field decomposition in U reconciles at machine precision (1e-16).
  At final time the wrong-transport component is 95 percent or more of the
  total error (G1: 6.53e-2 of 6.87e-2; G2: 5.29e-2 of 5.37e-2, absolute
  L2), identical across closures; particle discretization is ~50x smaller
  and refines; score regularization ~1000x smaller.
- Analytic anchor: measured frozen-left offset -0.093974 vs predicted
  -0.093982 (8e-6 agreement).
- Carrier refinement gate: U passes (last reductions 1.62-1.88, second
  order); binned q FAILS (plateau ~0.15 relative, reduction 1.05). The
  q decomposition shows why: the one-carrier-per-jump binned field has an
  O(1)-per-cell sawtooth, so its discretization component (0.23-0.30)
  swamps transport (0.11-0.29). Per protocol Section 8, no q-convergence
  claim is made; the field-level story runs through U, where binning
  integrates away. This limitation must be reported, not repaired.

## Initial-rate anchor (representation_rate_20260901)

For G1, the analytic defect norm ||D[g]||_2 = 2.1038e-1 and the measured
separation of the unregularized reference from the true heat carrier obeys
e(tau) = tau * ||D[g]|| to 0.08 percent at tau = 0.0125, with the ratio
rising linearly in tau (the O(tau^2) term). The short-time separation
theorem's constant is the measured constant.
