"""Contact-implicit MPC with a Pinocchio-compatible boundary."""

from dataclasses import dataclass
import time

import numpy as np
from pydrake.common.eigen_geometry import Quaternion
from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.multibody.tree import JacobianWrtVariable
from pyidto import (
    ProblemDefinition, SolverParameters, TrajectoryOptimizer,
    TrajectoryOptimizerSolution, TrajectoryOptimizerStats,
)

from .model import StateAdapter, build_model, joint_names


@dataclass
class MPCConfig:
    arm_joints: int = 4
    nodes: int = 14
    dt: float = 0.04
    iterations: int = 10
    initial_iterations: int = 10
    threads: int = 4
    stiffness: float = 30000.
    smoothing: float = 0.002
    max_base_residual: float = 5.0
    max_refinements: int = 2

    def __post_init__(self):
        joint_names(self.arm_joints)
        if not isinstance(self.max_refinements, int) or self.max_refinements < 0:
            raise ValueError("max_refinements must be a nonnegative integer")
        for name in ("nodes", "iterations", "initial_iterations", "threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("dt", "stiffness", "smoothing", "max_base_residual"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


class B2Z1MPC:
    """solve(q_pin, v_pin, time, ...) -> diagnostics; command(time) -> qj,vj,tauj.

    Only one caller may use an instance at a time. The C++ solver retains
    pointers to the diagram/plant, so both are owned for this object's lifetime.
    Cartesian arm velocity is converted into a *soft* joint trajectory target.
    Nonzero arm force requests are rejected: they need a manipulation contact
    model and an IDTO objective/constraint extension.
    """

    def __init__(self, config=None, pin_joint_names=None):
        self.config = config or MPCConfig()
        cfg = self.config
        builder, self.plant, self.scene_graph, _ = build_model(cfg.dt, cfg.arm_joints)
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        self.adapter = StateAdapter(self.plant, pin_joint_names or joint_names(cfg.arm_joints))
        self.q0, self.v0 = self.adapter.standing_state(
            self.plant_context, cfg.stiffness, cfg.smoothing)
        self.q0_pin, self.v0_pin = self.adapter.to_pin(self.q0, self.v0)
        self.problem = self._problem()
        self.params = SolverParameters()
        self.params.max_iterations = cfg.iterations
        self.params.num_threads = cfg.threads
        self.params.verbose = False
        self.params.normalize_quaternions = True
        self.params.equality_constraints = True
        self.params.contact_stiffness = cfg.stiffness
        self.params.smoothing_factor = cfg.smoothing
        self.params.dissipation_velocity = 0.1
        self.params.stiction_velocity = 0.05
        self.params.friction_coefficient = 0.8
        self.optimizer = TrajectoryOptimizer(self.diagram, self.plant, self.problem, self.params)
        self.warm_start = self.optimizer.CreateWarmStart(
            [self.q0.copy() for _ in range(cfg.nodes + 1)])
        self.solution = None
        self.solution_time = None
        self.last_diagnostics = None

    def _problem(self):
        cfg, ad = self.config, self.adapter
        nq, nv = self.plant.num_positions(), self.plant.num_velocities()
        p = ProblemDefinition()
        p.num_steps = cfg.nodes
        p.q_init, p.v_init = self.q0, self.v0
        wq = np.zeros(nq)
        wq[ad.qb:ad.qb + 4] = [2000, 8000, 8000, 2000]
        wq[ad.qb + 4:ad.qb + 7] = [500, 500, 4000]
        wq[ad.qj] = [30 if name in joint_names(0) else 100 for name in ad.names]
        wv = np.ones(nv)
        wv[ad.vb:ad.vb + 6] = [40, 40, 40, 150, 150, 150]
        wr = np.full(nv, 1e-4)
        wr[ad.vb:ad.vb + 6] = 1.0
        p.Qq, p.Qv, p.R = np.diag(wq), np.diag(wv), np.diag(wr)
        p.Qf_q, p.Qf_v = 5 * p.Qq, 5 * p.Qv
        p.q_nom = [self.q0.copy() for _ in range(cfg.nodes + 1)]
        p.v_nom = [self.v0.copy() for _ in range(cfg.nodes + 1)]
        return p

    def _nominal(self, q, base_velocity, arm_velocity):
        ad, cfg = self.adapter, self.config
        quat = Quaternion(ad.unit_quaternion(q[ad.qb:ad.qb + 4]))
        yaw = RollPitchYaw(quat).yaw_angle()
        base_R = RotationMatrix.MakeZRotation(yaw).matrix()
        # Planar body-frame command: vx, vy, yaw rate.
        linear_world = base_R @ base_velocity[:3]
        arm_indices = [i for i, name in enumerate(ad.names) if name.startswith("joint")]
        arm_qi, arm_vi = ad.qj[arm_indices], ad.vj[arm_indices]
        arm_ref = q[arm_qi].copy()
        qs, vs = [], []
        for k in range(cfg.nodes + 1):
            qn, vn = self.q0.copy(), self.v0.copy()
            qn[ad.qb:ad.qb + 4] = RollPitchYaw(
                0., 0., yaw + base_velocity[5] * k * cfg.dt).ToQuaternion().wxyz()
            if qn[ad.qb:ad.qb + 4] @ q[ad.qb:ad.qb + 4] < 0:
                qn[ad.qb:ad.qb + 4] *= -1
            qn[ad.qb + 4:ad.qb + 6] = q[ad.qb + 4:ad.qb + 6] + linear_world[:2] * k * cfg.dt
            vn[ad.vb:ad.vb + 3] = [0, 0, base_velocity[5]]
            vn[ad.vb + 3:ad.vb + 6] = linear_world
            if len(arm_indices):
                qn[arm_qi] = arm_ref
                self.plant.SetPositions(self.plant_context, qn)
                # gripperCenter is a fixed joint at an offset on gripperStator;
                # use its child body's frame to retain the original EE point.
                frame = self.plant.GetJointByName("gripperCenter").frame_on_child()
                J = self.plant.CalcJacobianTranslationalVelocity(
                    self.plant_context, JacobianWrtVariable.kV, frame, np.zeros(3),
                    self.plant.world_frame(), self.plant.world_frame())
                Rn = RotationMatrix(Quaternion(qn[ad.qb:ad.qb + 4])).matrix()
                Jarm = J[:, arm_vi]
                target = Rn @ arm_velocity
                # Base transport motion is already represented by vn. The arm
                # joints add only the requested relative end-effector motion.
                qd = Jarm.T @ np.linalg.solve(Jarm @ Jarm.T + 1e-4 * np.eye(3), target)
                qd = np.clip(qd, -ad.velocity_limits[arm_indices], ad.velocity_limits[arm_indices])
                next_ref = np.clip(arm_ref + cfg.dt * qd,
                                   ad.lower[arm_indices], ad.upper[arm_indices])
                vn[arm_vi] = (next_ref - arm_ref) / cfg.dt
                arm_ref = next_ref
            qs.append(qn)
            vs.append(vn)
        return qs, vs

    def _sample(self, elapsed):
        cfg, ad = self.config, self.adapter
        if self.solution is None:
            raise RuntimeError("Call solve before sampling a command")
        t = np.clip(elapsed / cfg.dt, 0., cfg.nodes)
        i = min(int(t), cfg.nodes - 1)
        alpha = t - i
        q1, q2 = self.solution[0][i:i + 2].copy()
        if q1[ad.qb:ad.qb + 4] @ q2[ad.qb:ad.qb + 4] < 0:
            q2[ad.qb:ad.qb + 4] *= -1
        q = (1 - alpha) * q1 + alpha * q2
        q[ad.qb:ad.qb + 4] = ad.unit_quaternion(q[ad.qb:ad.qb + 4])
        v = (1 - alpha) * self.solution[1][i] + alpha * self.solution[1][i + 1]
        tau = self.solution[2][i].copy()  # piecewise constant generalized force
        return q, v, tau

    def solve(self, q_pin, v_pin, current_time, base_vel_des=None,
              arm_vel_des=None, arm_force_des=None):
        call_started = time.perf_counter()
        ad, cfg = self.adapter, self.config
        if not np.isfinite(current_time):
            raise ValueError("current_time must be finite")
        if self.solution_time is not None and current_time < self.solution_time:
            raise ValueError("MPC timestamps must be monotonic")
        base = ad._vector(np.zeros(6) if base_vel_des is None else base_vel_des, 6, "base_vel_des")
        arm = ad._vector(np.zeros(3) if arm_vel_des is None else arm_vel_des, 3, "arm_vel_des")
        force = ad._vector(np.zeros(3) if arm_force_des is None else arm_force_des, 3, "arm_force_des")
        if np.any(force != 0):
            raise NotImplementedError("Nonzero arm_force_des requires an IDTO manipulation contact model")
        if np.any(base[[2, 3, 4]] != 0):
            raise ValueError("Planar locomotion supports base vx, vy and yaw rate only")
        if cfg.arm_joints == 0 and np.any(arm != 0):
            raise ValueError("Arm velocity requires active arm joints")
        q, v = ad.to_drake(q_pin, v_pin)
        reference = self.q0 if self.solution is None else self.solution[0][0]
        if q[ad.qb:ad.qb + 4] @ reference[ad.qb:ad.qb + 4] < 0:
            q[ad.qb:ad.qb + 4] *= -1
        q_nom, v_nom = self._nominal(q, base, arm)
        self.optimizer.ResetInitialConditions(q, v)
        self.optimizer.UpdateNominalTrajectory(q_nom, v_nom)
        if self.solution is None:
            guess = [q.copy() for _ in range(cfg.nodes + 1)]
        else:
            elapsed = current_time - self.solution_time
            # Hold the last knot outside the old horizon instead of extrapolating.
            guess = [self._sample(elapsed + k * cfg.dt)[0] for k in range(cfg.nodes + 1)]
            for item in guess:
                if item[ad.qb:ad.qb + 4] @ q[ad.qb:ad.qb + 4] < 0:
                    item[ad.qb:ad.qb + 4] *= -1
            guess[0] = q.copy()
        if self.solution is not None and self.warm_start.Delta < self.params.Delta0:
            self.warm_start = self.optimizer.CreateWarmStart(guess)
        else:
            self.warm_start.set_q(guess)
        first = self.solution is None
        # params() returns a copy in this binding; use a separate initial solver.
        optimizer = self.optimizer
        if first:
            self.params.max_iterations = cfg.initial_iterations
            self.problem.q_init, self.problem.v_init = q, v
            self.problem.q_nom, self.problem.v_nom = q_nom, v_nom
            optimizer = TrajectoryOptimizer(self.diagram, self.plant, self.problem, self.params)
        started = time.perf_counter()
        total_iterations = 0
        expected = ((cfg.nodes + 1, self.plant.num_positions()),
                    (cfg.nodes + 1, self.plant.num_velocities()),
                    (cfg.nodes, self.plant.num_velocities()))
        for refinement in range(cfg.max_refinements + 1):
            result, stats = TrajectoryOptimizerSolution(), TrajectoryOptimizerStats()
            optimizer.SolveFromWarmStart(self.warm_start, result, stats)
            total_iterations += len(stats.iteration_times)
            candidate = (np.asarray(result.q), np.asarray(result.v), np.asarray(result.tau))
            if any(a.shape != shape or not np.isfinite(a).all()
                   for a, shape in zip(candidate, expected)):
                raise RuntimeError("IDTO returned an invalid trajectory")
            base_residual = float(np.max(np.abs(candidate[2][:, ad.vb:ad.vb + 6])))
            if base_residual <= cfg.max_base_residual:
                break
            if refinement < cfg.max_refinements and self.warm_start.Delta < self.params.Delta0:
                self.warm_start = optimizer.CreateWarmStart(list(candidate[0]))
        duration = time.perf_counter() - started
        self.params.max_iterations = cfg.iterations
        qj, vj, tauj = candidate[0][:, ad.qj], candidate[1][:, ad.vj], candidate[2][:, ad.vj]
        diagnostics = {
            "solve_ms": (time.perf_counter() - call_started) * 1000,
            "optimizer_ms": duration * 1000,
            "iterations": total_iterations,
            "refinements": refinement,
            "trust_radius": self.warm_start.Delta,
            "base_residual_max": base_residual,
            "joint_position_violation": float(max(0, np.max(ad.lower - qj), np.max(qj - ad.upper))),
            "joint_velocity_violation": float(max(0, np.max(np.abs(vj) - ad.velocity_limits))),
            "joint_torque_violation": float(max(0, np.max(np.abs(tauj) - ad.effort_limits))),
        }
        if (diagnostics["base_residual_max"] > cfg.max_base_residual
                or diagnostics["joint_position_violation"] > 1e-3
                or diagnostics["joint_velocity_violation"] > 1e-2
                or diagnostics["joint_torque_violation"] > 1e-2):
            raise RuntimeError(f"IDTO trajectory failed feasibility checks: {diagnostics}")
        self.solution = candidate
        self.solution_time = float(current_time)
        self.last_diagnostics = diagnostics
        return diagnostics.copy()

    def command(self, current_time):
        if not np.isfinite(current_time):
            raise ValueError("current_time must be finite")
        if self.solution_time is None:
            raise RuntimeError("Call solve before command")
        elapsed = current_time - self.solution_time
        if elapsed < -1e-9 or elapsed > self.config.nodes * self.config.dt + 1e-9:
            raise RuntimeError("MPC trajectory is not valid at the requested timestamp")
        q, v, tau = self._sample(elapsed)
        ad = self.adapter
        # Limits here bound the command, not the optimizer's predicted motion.
        return (np.clip(q[ad.qj], ad.lower, ad.upper),
                np.clip(v[ad.vj], -ad.velocity_limits, ad.velocity_limits),
                np.clip(tau[ad.vj], -ad.effort_limits, ad.effort_limits))
