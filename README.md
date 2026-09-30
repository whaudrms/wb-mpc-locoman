# Whole-Body MPC for Loco-Manipulation

Official code for the paper: **Whole-Body Inverse Dynamics MPC for Legged Loco-Manipulation**, IEEE Robotics and Automation Letters (RA-L) 2025. Lukas Molnar, Jin Cheng, Gabriele Fadini, Dongho Kang, Fatemeh Zargarbashi, Stelian Coros. *ETH Zurich*

[<u>Paper</u>](https://ieeexplore.ieee.org/document/11266934) | [<u>arXiv</u>](https://arxiv.org/abs/2511.19709) | [<u>Video</u>](https://www.youtube.com/watch?v=glWWE-754mI&t=16s) | [<u>Website</u>](https://lukasmolnar.github.io/wb-mpc-locoman/)

<p float="left">
  <img src="utils/media/b2_z1_tracking.gif" width="49%" />
  <img src="utils/media/b2_z1_pulling.gif" width="49%" />
</p>

## Installation

Create conda environment:

```bash
conda create -f environment.yaml
conda activate wb-mpc
```

## Usage

Run the main script:

```bash
python main.py
```

Within the script, the following parameters are defined:
- Robot: Model and dynamics (whole-body or centroidal, see table below)
- Targets: Base velocity, and arm end-effector velocity/force
- OCP: Number of nodes and time discretization
- Gait: Type, period and swing parameters
- Solver: Type ("fatrop", "ipopt", "osqp", or "qp"), warm-starting, code-compilation

## Optimal Control Problem

### Dynamics

The table below shows the available dynamics models (whole-body and centroidal variants). See the paper for detailed benchmarking results.

For certain models, the argument `include_base` determines whether the base variable is part of the input (set in `args.py`). If it is included, the dynamics are ensured through a path constraint on each node. If it is not included, the base dynamics propagate through the state transition function.

![alt text](utils/media/dynamics_models.png)

### Parameters

The optimization parameters fall into the following categories:
- Initial state: `x_init`
- Tracking targets: `base_vel_des`, `arm_vel_des`, `arm_force_des`
- Gait schedule: `contact_schedule` (0 or 1), `swing_schedule` (phase between 0 and 1)
- Tunable parameters:
    - `Q_diag`, `R_diag`: Diagonals of the weight matrices
    - `swing_period`, `swing_height`, `swing_vel_limits`: Swing trajectory params
    - `dt_min`, `dt_max`: Initial and final time step sizes of the geometric series
    - `n_contacts`: Number of stance feet (eg. 2 for trot)

## Solvers

### Single convex QP: whole-body RNEA

Select the following configuration in `main.py`:

```python
dynamics = "whole_body_rnea_qp"
solver = "OSQP"
qp_condesed = False
```

`OCPWholeBodyRNEAQP` builds **one affine horizon QP per MPC update** and
calls OSQP once. There is no SQP loop or line search. OSQP still performs
its own numerical iterations to solve that one QP; `max_iter` controls those
iterations, not the number of horizon QPs.

The states are configuration/velocity differences relative to the measured
initial state. Inputs are whole-body accelerations, end-effector forces and
joint torques. This mode uses explicit acceleration and torque variables at
**every node**, regardless of `tau_nodes`, and requires `include_acc=True`.
It keeps the existing quadratic tracking cost, linearizes RNEA, manifold
Euler integration and end-effector velocity constraints once around a shifted
previous trajectory, and uses the inscribed friction pyramid
`abs(fx) + abs(fy) <= mu*fz`, `fz >= 0`. Contact schedules and time steps remain
parameters. Joint position/velocity bounds also apply at the terminal node.

`build_qp()` returns numeric sparse `P, q, A, l, u` for the correction to the
reference trajectory. CasADi is used for model/Jacobian evaluation, not for
an NLP solve. Q/R weights must be nonnegative. Solver settings, correction
regularization and the accepted affine constraint residual are in
`SOLVER_ARGS["qp"]`. Failed/non-finite QP solutions and solutions exceeding
`max_qp_violation` raise an error before updating the prediction.

The log distinguishes `affine CV` (the solved QP) from `nonlinear CV`
(the original nonlinear constraints evaluated at the reconstructed solution).
This is a local approximation: a feasible QP does not guarantee nonlinear
feasibility. The existing `main.py` loop advances to the **predicted state**;
it is not an independent contact simulation or hardware validation. Hard
end-effector targets are retained and may become infeasible; there is no
implicit target relaxation or fallback NLP solve.

Run regression tests from the repository root in the `wb-mpc` environment:

```bash
python -m unittest discover -s tests -v
```

If ROS injects an incompatible Pinocchio through `PYTHONPATH`, run with
`env -u PYTHONPATH python ...` in that environment.

### Single convex QP: centroidal velocity

Select the solver and condensing independently in `main.py`:

```python
dynamics = "centroidal_vel_qp"
solver = "qpOASE"     # "qpOASE" (or "qpOASES"), "OSQP", "HPIPM"
qp_condesed = True    # Python True / False; parameter spelling is intentional
```

| Solver | `qp_condesed = False` | `qp_condesed = True` |
|---|---|---|
| `qpOASE` | Original QP, dense active-set solve | Condensed QP, dense active-set solve |
| `OSQP` | Original sparse QP | Condensed QP |
| `HPIPM` | Original OCP QP; states and inputs retained | Condensed dense QP |

Every combination evaluates the model/Jacobian once and solves one horizon QP,
without an SQP loop, line search, or automatic switch to another solver.
The default 14-node trot case has 938 original variables and 154 condensed variables.
The uncondensed qpOASES path can be very slow at this size; its dense active-set
solve exceeded 130 seconds in a headless smoke check and was stopped. All six
combinations were validated at four nodes, including a contact switch.
The selected backend and condensing method appear in the solve log.
`ocp.qp_backend` and `ocp.qp_condense_equalities` expose the selected settings.

Solver options stay in `SOLVER_ARGS["qp"]` in `args.py`: `qpoases_opts` for
qpOASES, `opts` for OSQP, and `hpipm_opts` / `hpipm_mode` for HPIPM.
qpOASES uses `nWSR` for its iteration limit, OSQP uses `max_iter`, and HPIPM uses
`iter_max`. The legacy `solver="qp"` still uses the model's default backend
(qpOASES for centroidal, OSQP for RNEA); `qp_condesed` controls its reduction too.
For nonlinear dynamics, keep `fatrop`, `ipopt`, or lowercase `osqp` (the SQP path).
For the original RNEA QP configuration select `whole_body_rnea_qp`, `OSQP`, and
`qp_condesed=False`.

HPIPM is optional and is loaded only when selected. A project-local build is
discovered automatically, so no HPIPM-specific PYTHONPATH/LD_LIBRARY_PATH is needed:

```bash
bash benchmarks/build_hpipm.sh .deps/hpipm
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python main.py
```

The current workspace already contains the tested native runtime in `.deps/hpipm`
(ignored by Git). New checkouts need the build command above. The build targets
AVX2/FMA by default; see `benchmarks/README.md` for other CPUs. To use a different
build, set `HPIPM_ROOT` or `SOLVER_ARGS["qp"]["hpipm_root"]`. The build script sets
shared-library SONAMEs needed by the loader; rebuild older benchmark-only builds
with the updated script before selecting them as a runtime root.

HPIPM with condensing disabled retains all states and inputs and converts each
transition to `x_next = A*x + B*u + b`. It uses HPIPM's OCP solver. Local equality
constraints are passed as identical lower/upper bounds. This path failed to
converge at steps 94–96 in the saved benchmark; full condensing solved those QPs.
Both paths reject failed solutions before updating the trajectory. Condensing
can cost more than the solve time it saves; see the
[HPIPM comparison](benchmarks/results/hpipm_condensing.md).

The qpOASES adapter supplies a full symmetric dense Hessian and dense constraint
matrix, reconstructing both triangles from the stored upper-triangular `P`.
It reuses the CasADi solver function when dimensions match and recreates it when
the condensed dimensions change. CasADi's qpOASES interface hot-starts repeated
calls with updated Hessian/constraint matrices. Defaults use inactive initial
bounds, full linear-independence tests, Cholesky refactorisation at each active-
set iteration, and up to three iterative-refinement steps. The latter settings
avoid the premature hot-start infeasibility seen with default factor updates.
Failed, iteration-limited, or non-finite results are rejected before updating the prediction;
there is no automatic retry with another solver.

The physical state is **`[h/m, q]`**, where `h` contains CoM linear and angular
momentum expressed in world axes. The optimization state is
`[delta(h/m), delta_q]`, with configuration differences on the robot manifold.
The `include_base` option in `DYN_ARGS["centroidal_vel_qp"]` selects:

- `True` (default): inputs `[v_base, v_joints, forces]`, with the linearized
  consistency constraint `A_G(q) v = m * (h/m)`.
- `False`: inputs `[v_joints, forces]`; base velocity is recovered from the
  centroidal map, and that expression is differentiated with the other equations.

The QP reuses `centroidal_vel`'s dynamics equations directly, including
`delta_q_next = delta_q + dt*v`, centroidal momentum dynamics, momentum closure
and torque estimates. The nonlinear terms are linearized once per MPC update.
Physical configuration reconstruction still uses the robot manifold.

Warm starting calls the original NLP implementation: previous delta states and
velocities are reused at the same node index; force guesses are reset from the
desired support forces and current contact schedule. There is no trajectory
time shift, chart re-anchoring or base-velocity projection. The first reference
uses the original Opti initial values. `warm_start=False` disables previous-value
reuse. The physical prediction snapshot remains available for diagnostics.

This retains the CoM-based model and full robot kinematics; it is not the exact
affine reduced model using angular momentum about a fixed world origin.

As in `centroidal_vel`, the first `tau_nodes` input nodes use a **quasi-static
Jacobian/gravity torque estimate**. There are no torque or acceleration decision
variables, so these bounds do not guarantee full RNEA torque feasibility.
Joint position and velocity bounds apply at input nodes 0..N-1, with no added
terminal joint constraint, matching the original centroidal NLP. Friction keeps
the inscribed linear pyramid so the subproblem remains a convex QP. The default
14-node model has 938 variables and 1,700 constraint rows (previously 1,716).
The original NLP has 1,532 rows; the remaining 168 extra rows are pyramid faces.
The arm target extracts `q` after the six momentum entries of the initial state.

Output accelerations are reconstructed after the solve: joint accelerations
use velocity finite differences, and base acceleration uses centroidal dynamics.
The last input node uses the preceding joint-velocity interval because there is
no terminal velocity input. They are output estimates, not optimization inputs.

With `qp_condesed=True`, centroidal QP uses **stagewise condensing** in
`optimization/qp_condensing.py`. At each node a small SVD eliminates local
momentum consistency, foot/arm velocity and fixed-force equalities. The
delta-coordinate/momentum transition Jacobian then propagates the next state as an
affine function of the remaining free inputs. This avoids factoring the entire
horizon equality matrix. The full trajectory correction is `offset + basis @ y`;
the quadratic objective and all inequalities are transformed into these free
coordinates. Diagonal normalization of their cost curvature improves numerical
conditioning without changing the objective or adding a penalty.

This is an algebraically equivalent version of the same linearized QP, not a
single-rigid-body approximation: full kinematics, joint bounds and the existing
torque estimates remain. The default 14-node trot case has 938 original variables
and 154 free variables after condensation. Both `include_base` settings are
supported. Transition rows are recorded during OCP construction so a vanishing
Jacobian entry cannot change the detected dynamics structure.

If a stage input block loses row rank, a transition is singular, or the stage
structure is unsupported, the implementation falls back to the general
rank-revealing QR reduction. It retains compatibility constraints and rejects
inconsistent equalities; this fallback does not add an optimization solve.
`qp_condensing_method` reports `none`, `stagewise`, or `global_qr`, and
`qp_condensing_fallback_reason` records the reason when QR is needed.
`last_qp` stores the original numeric problem; `last_solver_qp` stores the reduced
problem. Residual acceptance is checked on the original QP after reconstruction.
RNEA QP remains uncondensed when `qp_condesed=False`; explicit condensing uses the general QR reduction.

For reproducible CPU timing, the headless validation used one BLAS/OpenMP
thread. The same setting can be used when running the example:

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python main.py
```

Timing attributes (seconds) separate `qp_condense_time`, `qp_setup_time`, and
`qp_solver_time`. `qp_build_time` includes matrix assembly and condensation;
`solve_time` includes assembly, condensation, backend setup/update and the solve,
but excludes final residual evaluation and prediction reconstruction.

The QP runtime now caches fixed CasADi variable/parameter expressions, sparse
matrix layouts and state-integration functions. Stagewise condensing vectorizes
constraint classification and avoids repeated sparse slicing. Equality checks,
QR fallback and solver tolerances are retained.

`main.py` also exposes:

```python
blas_threads = 1        # Set before importing NumPy/CasADi.
qp_compile_data = True  # Native C evaluation for model/Jacobian and constraints.
qp_profile = True      # Print QP phase times in detailed mode.
log_mode = "original"  # Default: original NLP logs; "detailed": full MPC + QP phases.
```

The first native build needs a C compiler (`cc`); a previous version of the
14-node model took about 4 min 37 s to compile. It is cached under `.deps/qp_codegen`; this workspace's 14-node model
is already compiled. Subsequent initialization generates/hashes the code and
loads the cached library (about 1.6 s in the check). Robot/horizon/graph changes
may require a new build. Numerical state/target updates reuse the cache.
`qp_compile_data=False` disables native compilation while keeping other runtime
optimizations. This flag is separate from the legacy NLP `compile_solver` flag.

`main.py` uses one common MPC loop and timing boundary for FATROP, IPOPT and QP:
parameter/warm-start update → input packing → optimization → nonlinear constraint
evaluation → solution retraction → next-state calculation. Initialization,
compilation/loading, console output and visualization are outside the interval.
`log_mode="original"` (default) restores the original NLP output: `Solve time (ms)`,
`CV (inf norm)`, and average/std solve-time statistics. For NLP this measures only
`solver_function`, excluding input packing, as in the original repository. For QP
it uses `ocp.solve_time`, including QP assembly/condensing/setup and solving;
these legacy solve-time values do not share an identical comparison boundary.
They are saved as `ocp.solve_times` (seconds).

`log_mode="detailed"` selects the full MPC/optimization logs and QP diagnostics
(with phase timings controlled by `qp_profile`). Both modes retain all timing
arrays and the same common full-step measurement boundary.
The primary comparable statistic is **Avg full MPC step (ms)** in detailed mode. Per-step values
are exposed as `ocp.mpc_step_times`. `ocp.optimization_times` and QP phase logs
remain diagnostic sub-timings; they are not the full MPC latency.

The saved [optimization measurements](benchmarks/results/centroidal_runtime_optimized.md)
predate the integration/warm-start/terminal-bound alignment and are historical,
not performance claims for the current formulation. Regenerate measurements and
frozen QP datasets after this change; do not reuse the old `/tmp` dataset.
`python benchmarks/profile_centroidal_runtime.py --verify` measures the current
QP path. The original nonlinear friction cone and the QP pyramid still differ;
C compilation and solver accuracy also need matching for a full NLP comparison.

When explicitly selecting OSQP, centroidal QP defaults to 50 equilibration passes (`scaling=50`) to improve
conditioning between momentum and kinematic rows. It updates the ADMM penalty
every 25 iterations (`adaptive_rho_interval=25`) and uses `eps_prim_inf=1e-8`,
`eps_dual_inf=1e-8` to avoid premature infeasibility certificates in condensed
coordinates. Solution tolerances and original-QP residual acceptance remain
unchanged. Its OSQP workspace is rebuilt
for the current Jacobian each update: simply updating the first workspace can
stall even when a fresh solve of the same QP converges. Previous-node warm-start
values are used as the linearization reference. This is matrix scaling
and one QP solve, not repeated linearization. `scaling` can be overridden in
`SOLVER_ARGS["qp"]["opts"]`; RNEA QP retains its existing settings and workspace
update behavior.
The same nonlinear-feasibility and predicted-state feedback limitations above
apply to this model.

Validation: the regression suite covers both `include_base` settings,
finite-difference Jacobian checks, QP convexity, equality-condensation
exactness (including affine defects and rank-loss fallback), momentum scaling,
rotated-base arm targets, full-horizon output,
contact switches, single solver calls, qpOASES symmetric-Hessian conversion,
solver failure handling, backend selection, and the existing RNEA mode.

Historical limitation before the integration/warm-start alignment: with 14 nodes,
trot, base velocity `[0.1, 0, 0, 0, 0, 0]` and arm velocity `[0.1, 0, -0.2]`,
the predicted-state rollout became affine-QP infeasible around `t = 1.455 s`
(zero-based step 97). That result has not been re-established for the current
formulation. The earlier stagewise implementation completed steps 0 through 96 without QR fallback,
then stopped at step 97, with both OSQP and qpOASES. Infeasibility was also
confirmed on the original,
uncondensed QP by HiGHS dual-simplex and interior-point feasibility checks
with presolve disabled. The solver stops without applying that failed
solution. The example is not a validated 200-step locomotion controller;
longer operation may require task relaxation or a different reference/target
policy, as well as validation against nonlinear contact dynamics.

### Interior-Point: Fatrop and Ipopt

The interior-point solvers **Fatrop** and **Ipopt** are supported, which directly solve the constrained nonlinear optimization problem until convergence.

As described in the paper, **Fatrop** exploits the block-sparse structure of stage-wise constriants, and shows a >10x speedup over **Ipopt** (higher speedup for longer horizons). The solver is warm-started with the MPC solution from the previous iteration. 

### Sequential Quadratic Programming: OSQP

Instead of solving the full NLP, it is converted to a Sequential Quadruatic Program (SQP). Each SQP iteration is solved with OSQP, and the solution is updated using the Armijo line-search method.


### Code-generation

Currently code-generation is only supported for **Fatrop**, since it showed the most promising results in terms of solve-time and convergence. See the `/codegen` folder for how to generate C code for the solver and compile it to a shared library.

For hardware deployment the shared library can be loaded with `casadi::external` in C++. This allows for straight forward deployment without having to reformulate the optimization problem in C++. It also allows for real-time parameter tuning.

## Citation

If you use this code in your research, please cite our paper:
```bibtex
@ARTICLE{11266934,
  author={Molnar, Lukas and Cheng, Jin and Fadini, Gabriele and Kang, Dongho and Zargarbashi, Fatemeh and Coros, Stelian},
  journal={IEEE Robotics and Automation Letters}, 
  title={Whole-Body Inverse Dynamics MPC for Legged Loco-Manipulation}, 
  year={2026},
  volume={11},
  number={1},
  pages={898-905},
  keywords={Dynamics;Robots;Robot kinematics;Manipulator dynamics;Legged locomotion;Force;Quadrupedal robots;Foot;Real-time systems;Planning;Legged Robots;Mobile Manipulation;Whole-Body Motion Planning and Control},
  doi={10.1109/LRA.2025.3636005}}
```

## Contact

Feel free to open an issue or discussion if you encounter any problems or have questions about this project.

For collaborations, feedback, or further inquiries, please reach out to:

- Lukas Molnar: [lukas.molnar@bluewin.ch](mailto:lukas.molnar@bluewin.ch).

We welcome contributions and are happy to support the community in building upon this work!