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

Select the following pair in `main.py`:

```python
dynamics = "whole_body_rnea_qp"
solver = "qp"
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

The current `main.py` defaults select this pair:

```python
dynamics = "centroidal_vel_qp"
solver = "qp"
```

This model shares the numeric QP backend in `optimization/affine_qp.py` with
`whole_body_rnea_qp`. Each MPC update evaluates the model/Jacobian once and
solves one horizon QP, without an SQP loop or line search.

The physical state is **`[h/m, q]`**, where `h` contains CoM linear and angular
momentum expressed in world axes. The optimization state is
`[delta(h/m), delta_q]`, with configuration differences on the robot manifold.
The `include_base` option in `DYN_ARGS["centroidal_vel_qp"]` selects:

- `True` (default): inputs `[v_base, v_joints, forces]`, with the linearized
  consistency constraint `A_G(q) v = m * (h/m)`.
- `False`: inputs `[v_joints, forces]`; base velocity is recovered from the
  centroidal map, and that expression is differentiated with the other equations.

The momentum dynamics, manifold Euler integration, foot/arm velocity tasks
and torque estimates are linearized around a time-shifted, re-anchored
prediction. Momentum is interpolated linearly and configuration is interpolated
on the manifold. This retains the CoM-based model and full robot kinematics;
it is **not** the exact affine reduced model using angular momentum about a
fixed world origin.

As in `centroidal_vel`, the first `tau_nodes` input nodes use a **quasi-static
Jacobian/gravity torque estimate**. There are no torque or acceleration decision
variables, so these bounds do not guarantee full RNEA torque feasibility.
Joint velocity bounds apply to input nodes, joint position bounds include the
terminal state, and friction uses the same inscribed linear pyramid as RNEA QP.
The arm target extracts `q` after the six momentum entries of the initial state.

Output accelerations are reconstructed after the solve: joint accelerations
use velocity finite differences, and base acceleration uses centroidal dynamics.
The last input node uses the preceding joint-velocity interval because there is
no terminal velocity input. They are output estimates, not optimization inputs.

Before the solver call, centroidal QP eliminates affine equalities with a
rank-revealing QR: the full trajectory correction is `offset + basis @ y`.
It minimizes the same quadratic objective over `y` with the transformed
inequalities. This is an algebraically equivalent QP, not another approximation
or an additional optimization solve. It handles zero/redundant equality rows
and rejects inconsistent ones. `last_qp` stores the original numeric problem;
`last_solver_qp` stores the reduced problem. Residual acceptance is checked on
the original full QP after reconstruction. This dense QR adds preparation cost
but avoids the dual-convergence stalls observed on the full centroidal QP.
RNEA QP continues to use its original uncondensed solve.

Centroidal QP defaults to 50 OSQP equilibration passes (`scaling=50`) to improve
conditioning between momentum and kinematic rows. Its OSQP workspace is rebuilt
for the current Jacobian each update: simply updating the first workspace can
stall even when a fresh solve of the same QP converges. The shifted physical
trajectory is still used as the linearization reference. This is matrix scaling
and one QP solve, not repeated linearization. `scaling` can be overridden in
`SOLVER_ARGS["qp"]["opts"]`; RNEA QP retains its existing settings and workspace
update behavior.
The same nonlinear-feasibility and predicted-state feedback limitations above
apply to this model.

Validation: the regression suite covers both `include_base` settings,
finite-difference Jacobian checks, QP convexity, equality-condensation
exactness, momentum scaling, rotated-base arm targets, full-horizon output,
contact switches, single solver calls, and the existing RNEA mode.

Known limitation of the current example: with 14 nodes, trot, base velocity
`[0.1, 0, 0, 0, 0, 0]` and arm velocity `[0.1, 0, -0.2]`, the predicted-state
rollout becomes affine-QP infeasible around `t = 1.455 s` (zero-based step 97).
This was also confirmed on the original, uncondensed QP with an independent
linear feasibility check. The solver stops without applying that failed
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