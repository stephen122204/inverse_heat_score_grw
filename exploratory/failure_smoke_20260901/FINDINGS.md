# Injected-failure smoke (exploratory, 2026-09-01)

One failure injected per driver family through mock patches of the
underlying runners, with the drivers and result contract running for real.
All 14 checks passed at code commit 78c4bf6:

- bandwidth_clean: the poisoned row is retained as failed with its step and
  message, the driver completes the remaining rows, and resume treats the
  failed row as attempted instead of re-running it.
- noise_paired: with one bandwidth broken in a block, both arms record the
  failure, no bandwidth selection is made from the partial residual curve,
  and the block is reported as incomplete.
- lambda_noise: an injected monotonicity violation becomes a failed row
  labeled failed_monotonicity (never Morozov), the oracle row is untouched,
  and the failure count is surfaced in the summary.
- closure: an injected carrier positivity loss is a failed row and the
  refinement verdict fails on the incomplete ladder rather than passing
  vacuously.
- transition_table: a legacy-era failure is retained while the canonical
  era completes.

Machine-readable record with file hashes: outputs/smoke_record.json
(regenerated on each run; outputs stay local).
