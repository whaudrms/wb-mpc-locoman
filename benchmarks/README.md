# Centroidal condensed-QP solver comparison

This benchmark compares OSQP, qpOASES and **HPIPM dense QP** on the same frozen
linearized/condensed QPs. It does not select a new controller backend. A separate OCP-versus-condensed
comparison is described below.

## Reproduce

Use the `wb-mpc` conda environment, from the repository root. The optional native
libraries are built in a separate directory; the conda installation is untouched.
The pinned official sources are [HPIPM](https://github.com/giaf/hpipm) and
[BLASFEO](https://github.com/giaf/blasfeo). The default build uses AVX2/FMA and was
tested on an Intel Core i7-12700H; select appropriate targets on other processors.

```bash
bash benchmarks/build_hpipm.sh /tmp/wb_mpc_hpipm_bench

env PYTHONPATH=/tmp/wb_mpc_hpipm_bench/hpipm/interfaces/python/hpipm_python \
  LD_LIBRARY_PATH=/tmp/wb_mpc_hpipm_bench/install/hpipm/lib:/tmp/wb_mpc_hpipm_bench/install/blasfeo/lib \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python benchmarks/compare_centroidal_solvers.py --repeats 3
```

The explicit `PYTHONPATH` replaces ROS's incompatible Pinocchio path. Results
are written to `benchmarks/results/centroidal_solvers.json`. The 187 MB frozen
QP dataset is kept in `/tmp/centroidal_solver_cases.pkl`, outside the repository.
Pass a new `--dataset` path to collect fresh QPs after changing the controller.
Only load datasets generated locally by this script (the format is pickle).
`--cpu` selects one allowed CPU; default is the first CPU in the affinity mask.

## Method

- Generate one qpOASES predicted-state trajectory using `main.py` parameters:
  B2 + 4 arm joints, 14 nodes, trot, 0.015–0.08 s intervals. Every solver receives
  the same numeric QP at each step; competing solver solutions do not change the
  next benchmark QP.
- Test three sequential replays of steps 0–97. Steps 0–96 are feasible; step 97
  is an independently checked infeasible case, kept separate from feasible-case
  timing summaries. Rotate solver order between repetitions.
- Also test cold starts independently on steps 0, 20, 40, 54, 80, 94, 95 and 96.
- Match existing controller policies in replay: OSQP rebuilds/equilibrates each
  QP; qpOASES reuses its active set; HPIPM reuses buffers but cold-starts its
  primal/dual iterates (`warm_start=0`) because reduced coordinates can change.
  The cold-start comparison removes cross-QP reuse for every solver.
- Apply the same zero-row cleanup to all three solvers. Identically zero rows
  satisfied within 1e-10 are omitted, avoiding redundant `0 <= 0` barrier rows.
  This only tolerates roundoff in fixed-state bounds; nonzero coefficients are
  not truncated. Infinite bounds are disabled using HPIPM masks, not big-M.
- Retain original cost weights, regularization and all nontrivial constraints.
  Report native success AND reconstructed original-QP violation <=1e-3 AND
  normalized stationarity <=1e-6. Report objective differences and complementary
  slackness separately; solver-native tolerance definitions are not identical.
- Report wall-clock setup/data-conversion and solve times separately. Matrix
  evaluation, linearization and condensing are common costs, measured once
  during dataset generation and excluded from solver-only timing. Build/import,
  file I/O and quality checks are outside solver timing. Timing includes failed
  attempts as well as successful ones.

HPIPM uses `robust`, `iter_max=100`, `tol_stat=1e-7`, `tol_eq=1e-8`,
`tol_ineq=1e-8`, `tol_comp=1e-8`. OSQP and qpOASES use current production options
from `args.py` and the centroidal class. Full settings and pinned dependency
commits are recorded in the JSON. An HPIPM minimum-step/iteration-limit return
is a failed solve, not by itself an infeasibility certificate.

Analytic conversion tests (including HPIPM inequality masks and equality-dual
sign) are in `tests/test_qp_solver_benchmark.py`. Run with the same environment:
`python -m unittest discover -s tests -p test_qp_solver_benchmark.py -v`.

## HPIPM without condensing vs full condensing + HPIPM

Use the same library/Python environment and frozen dataset:

```bash
env PYTHONPATH=/tmp/wb_mpc_hpipm_bench/hpipm/interfaces/python/hpipm_python \
  LD_LIBRARY_PATH=/tmp/wb_mpc_hpipm_bench/install/hpipm/lib:/tmp/wb_mpc_hpipm_bench/install/blasfeo/lib \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python benchmarks/compare_hpipm_condensing.py --repeats 3
```

Results: [report](results/hpipm_condensing.md), [raw JSON](results/hpipm_condensing.json).
The OCP adapter retains all 938 state/input variables and stage dynamics.
The condensed path re-runs the production stagewise reduction to 154 variables
for every measured QP. Both use HPIPM robust mode, identical tolerances and cold
primal iterates, reusing workspaces. The OCP conversion uses next-state block
inversion, not horizon-wide state elimination. Stage-local equalities are passed
as equal lower/upper general constraints. The full reduction additionally
eliminates these equalities and scales reduced-coordinate curvature; stability
differences cannot be attributed to state elimination alone.

Total time includes formulation preparation, canonicalization, solver data
updates, native solve, solution extraction and primal reconstruction. Accuracy
checks, imports, dataset I/O and common model assembly are excluded. Common
model assembly measured during dataset collection is reported separately;
adding it is an estimate, not a new full MPC-loop measurement. Failed solves
remain in all-attempt statistics; paired-success results compare only identical
QPs solved successfully by both paths. The objective is compared in the original
coordinates, including the constant lost through affine substitution.

The analytic OCP test checks nonidentity next-state dynamics, cross cost terms,
an active one-sided bound, dual reconstruction and workspace reuse. These adapters now share `optimization/hpipm_backend.py` with the controller.
Select `solver="HPIPM"` and `qp_condesed=True/False` in `main.py` to use them.

## Optimized MPC runtime

`env -u PYTHONPATH python benchmarks/profile_centroidal_runtime.py --verify`
executes the actual HPIPM + condensed MPC path headlessly, checks compiled versus
interpreted data and reports per-phase and full-step latency. `--no-compile`
retains Python/caching optimizations but disables native data evaluation.
Initialization/compilation and numerical verification are outside step timings.
See [results](results/centroidal_runtime_optimized.md). The older formulation
comparison reports describe the implementation measured at that time.
