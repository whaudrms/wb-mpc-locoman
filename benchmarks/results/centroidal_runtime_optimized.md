# Centroidal HPIPM condensed-QP runtime optimization

14 nodes, B2 + arm 4 joints, same targets/weights/constraints/regularization/tolerances.
BLAS/OpenMP one thread. No horizon reduction, no relaxed constraints and no approximation change.
Times are local observations, not a comparison against an NLP solver or a deadline guarantee.

## Before/after, initial matched step numbers

Steps 1–11, excluding step 3 (instrumented baseline). Startup excluded.

| Phase | Before ms | After ms |
|---|---:|---:|
| assembly | 24.40 | 4.60 |
| condense | 15.46 | 9.53 |
| setup | 1.28 | 1.21 |
| solve | 4.57 | 4.17 |
| reported | 45.80 | 19.61 |
| total | 60.33 | 26.06 |

Before full-loop timing stops before the next-state integration; after includes it.
`reported` is the original solve_time boundary: assembly, condensing, setup, and solve.
Full MPC timing also includes reference update, residual evaluation and prediction reconstruction.

## Optimized trajectory

| Interval | Successful steps | QP processing mean ms | Full step mean ms | Full step P95 ms | Full step max ms |
|---|---:|---:|---:|---:|---:|
| all | 97 | 27.20 | 33.73 | 58.25 | 80.50 |
| initial_1_31 | 31 | 19.64 | 26.11 | 26.47 | 26.55 |
| late_90_96 | 7 | 55.49 | 62.00 | 76.78 | 80.50 |

All 97 feasible steps succeeded. Step 97 remains rejected (HPIPM minimum-step status);
the earlier independent feasibility check identified that saved QP as infeasible.
The failing attempt is present in raw records and excluded from successful-step statistics.
Compiled/interpreted data were compared at all 98 evaluation points (atol 1e-8, rtol 1e-9).
Old/new condensing matrices and reconstruction maps agreed for all 98 frozen QPs at the same tolerances.

## Implementation

- Cache Opti variable/parameter expressions and CSC sparsity metadata.
- Cache the state integration function instead of constructing it for each horizon node.
- Vectorize local-equality classification; materialize constraint values once for stage slicing.
- Reuse Hessian-times-basis products; retain equality reconstruction checks and QR fallback.
- Compile CasADi model/Jacobian and constraint evaluation to C; use a content-addressed cache.
- Set BLAS threads explicitly before numerical libraries load.

First O2 compilation took 276.77 s; the generated C source was approximately 30 MB.
The measured cached initialization (code generation/hash/load) took 1.59 s, outside MPC timings.
A new robot, horizon or symbolic graph can trigger another compilation. Changing numerical
state, targets or parameter values does not invalidate the cache.
Set qp_compile_data=False to use the optimized interpreted path without a C compiler.

Reproduce: `env -u PYTHONPATH python benchmarks/profile_centroidal_runtime.py --verify`.
Use `--no-compile` to measure the interpreted path with the other optimizations retained.
