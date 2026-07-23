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
- Solver: Type ("fatrop", "ipopt", or "osqp"), warm-starting, code-compilation

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

### Interior-Point: Fatrop and Ipopt

The interior-point solvers **Fatrop** and **Ipopt** are supported, which directly solve the constrained nonlinear optimization problem until convergence.

As described in the paper, **Fatrop** exploits the block-sparse structure of stage-wise constriants, and shows a >10x speedup over **Ipopt** (higher speedup for longer horizons). The solver is warm-started with the MPC solution from the previous iteration. 

### Sequential Quadratic Programming: OSQP

Instead of solving the full NLP, it is converted to a Sequential Quadruatic Program (SQP). Each SQP iteration is solved with OSQP, and the solution is updated using the Armijo line-search method.


### Code-generation

Currently code-generation is only supported for **Fatrop**, since it showed the most promising results in terms of solve-time and convergence. See the `/codegen` folder for how to generate C code for the solver and compile it to a shared library.

For hardware deployment the shared library can be loaded with `casadi::external` in C++. This allows for straight forward deployment without having to reformulate the optimization problem in C++. It also allows for real-time parameter tuning.

## MuJoCo MPPI comparison

`compare_mujoco.py` runs the existing whole-body RNEA MPC and a
RTWholeBodyMPPI-style controller as separate closed-loop episodes on the same
MuJoCo B2-Z1 plant. The original `main.py` and OCP classes are unchanged.

The MPPI core retains the reference controller's action semantics: it samples
joint-position commands, MuJoCo's affine PD actuators generate rollout torques,
and the exponentially weighted action trajectory is shifted for the next
iteration. Its task layer consumes the same base velocity, arm end-effector
velocity, gait timing, swing height, and swing velocity references as the OCP.

Run a short comparison:

```bash
python compare_mujoco.py --controller both --steps 20
```

Useful options include:

```bash
python compare_mujoco.py \
  --base-vel 0.1 0 0 0 0 0 \
  --arm-vel 0.1 0 -0.2 \
  --gait trot \
  --gait-period 0.8 \
  --samples 128 \
  --temperature 0.1
```

For MPPI tuning, the main tracking and exploration parameters are also
available from the command line:

```bash
python compare_mujoco.py \
  --controller mppi \
  --ee-velocity-weight 75000 \
  --base-velocity-weight-scale 1.0 \
  --joint-position-weight-scale 1.0 \
  --arm-posture-weight-scale 1.0 \
  --arm-velocity-weight-scale 1.0 \
  --noise-scale 1.0 \
  --arm-noise-scale 1.0
```

The EE global-velocity target is evaluated once from the measured state at the
start of each MPPI update and held fixed across that rollout horizon, matching
the existing OCP target construction. The simulation comparison defaults use
a velocity-tracking profile selected from a 120-step sweep: joint-position
regularization 100, joint-velocity regularization 0, base-velocity weight
10000, EE-velocity weight 75000, torque weight 0.001, and arm exploration
noise 0.03 rad.
The joint-position cost tracks the time-varying gait and integrated arm
nominal trajectory used as the sampling center; it no longer pulls every
rollout state back to the fixed stand keyframe.
The 128-sample default favors balanced base/EE tracking in simulation; it is
not intended to meet the 15 ms real-time control budget.

The runner records the applied actuator torque, tracking errors, gait-contact
mismatch, command clipping, constraint violation, and solve time in
`results/mujoco_mpc_mppi.npz`.

Replay MPC and MPPI side by side after the timed simulation:

```bash
python compare_mujoco.py --controller both --steps 200 --visualize
```

Save comparison plots and a synchronized MuJoCo GIF:

```bash
python compare_mujoco.py \
  --controller both \
  --steps 200 \
  --plot \
  --record
```

The summary plot is saved to `results/mujoco_mpc_mppi.png`. A second
`results/mujoco_mpc_mppi_tracking.png` plot compares every base/EE velocity
component and base height against its dashed reference; measured values are
solid. The default replay output is `results/mujoco_mpc_mppi.gif`. On a
headless machine, select MuJoCo's EGL backend before recording:

```bash
MUJOCO_GL=egl python compare_mujoco.py --controller both --record
```

Rendering is performed from the recorded state trajectories after both
episodes finish, so visualization overhead is excluded from controller solve
times.

The shared default task has zero arm end-effector force. A nonzero arm-force
comparison requires an explicit MuJoCo interaction object or applied-wrench
model and is rejected by the MPPI adapter rather than silently changing the
task.

Run the MuJoCo bridge and MPPI regression tests with:

```bash
python -m unittest tests/test_mujoco_comparison.py -v
```

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
